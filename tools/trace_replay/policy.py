"""Cache replacement policies for MoE expert caching.

Implements pluggable eviction policies:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- LRU-K (K=2 backward distance)
- Hybrid (Recency + Frequency decay)
- Cost-Aware (Layer-priority & tier placement)
- Belady's Oracle (Optimal Offline Eviction - upper bound)
"""

from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import Collection, Dict, List, Optional, Set, Tuple


class ReplacementPolicy(ABC):
    """Abstract base class for expert cache replacement policies."""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def on_access(self, key: Tuple[int, int], step: int) -> None:
        """Record an access to expert key=(layer_idx, expert_idx) at time step."""
        pass

    @abstractmethod
    def select_eviction(self, candidates: Collection[Tuple[int, int]], step: int) -> Tuple[int, int]:
        """Select one key from candidates to evict."""
        pass

    def on_evict(self, key: Tuple[int, int]) -> None:
        """Notification that key was evicted from cache."""
        pass

    def should_admit(
        self, key: Tuple[int, int], victim: Tuple[int, int], step: int
    ) -> bool:
        """Decide whether `key` is worth displacing `victim` from the tier.

        Default True: every policy without an opinion behaves as before, admitting
        unconditionally. Overriding this is how a policy becomes scan-resistant --
        see TinyLFUAdmissionPolicy for why that matters when a single pass touches
        more distinct experts than the tier can hold.
        """
        return True

    def reset(self) -> None:
        """Reset policy state for a new simulation run."""
        pass


class LRUPolicy(ReplacementPolicy):
    """Least Recently Used policy."""

    def __init__(self) -> None:
        self.last_used: Dict[Tuple[int, int], int] = {}

    def name(self) -> str:
        return "LRU"

    def on_access(self, key: Tuple[int, int], step: int) -> None:
        self.last_used[key] = step

    def select_eviction(self, candidates: Collection[Tuple[int, int]], step: int) -> Tuple[int, int]:
        return min(candidates, key=lambda k: self.last_used.get(k, -1))

    def on_evict(self, key: Tuple[int, int]) -> None:
        self.last_used.pop(key, None)

    def reset(self) -> None:
        self.last_used.clear()


class LFUPolicy(ReplacementPolicy):
    """Least Frequently Used policy with tie-breaker on recency."""

    def __init__(self) -> None:
        self.freq: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        self.last_used: Dict[Tuple[int, int], int] = {}

    def name(self) -> str:
        return "LFU"

    def on_access(self, key: Tuple[int, int], step: int) -> None:
        self.freq[key] += 1
        self.last_used[key] = step

    def select_eviction(self, candidates: Collection[Tuple[int, int]], step: int) -> Tuple[int, int]:
        # Evict candidate with min frequency; tie break by oldest access
        return min(candidates, key=lambda k: (self.freq.get(k, 0), self.last_used.get(k, -1)))

    def on_evict(self, key: Tuple[int, int]) -> None:
        self.freq.pop(key, None)
        self.last_used.pop(key, None)

    def reset(self) -> None:
        self.freq.clear()
        self.last_used.clear()


class LRUKPolicy(ReplacementPolicy):
    """LRU-K policy (default K=2): tracks backward distance of the K-th previous access."""

    def __init__(self, k: int = 2) -> None:
        self.k = k
        self.history: Dict[Tuple[int, int], collections.deque[int]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.k)
        )

    def name(self) -> str:
        return f"LRU-{self.k}"

    def on_access(self, key: Tuple[int, int], step: int) -> None:
        self.history[key].append(step)

    def select_eviction(self, candidates: Collection[Tuple[int, int]], step: int) -> Tuple[int, int]:
        def k_distance(k_key: Tuple[int, int]) -> float:
            hist = self.history.get(k_key)
            if not hist or len(hist) < self.k:
                # Fewer than K accesses -> infinite backward distance (highest eviction priority)
                first_access = hist[0] if hist else -1
                return -1e12 + first_access
            # K-th backward distance: timestamp of K-th previous access
            return float(hist[0])

        return min(candidates, key=k_distance)

    def on_evict(self, key: Tuple[int, int]) -> None:
        self.history.pop(key, None)

    def reset(self) -> None:
        self.history.clear()


class HybridRecencyFrequencyPolicy(ReplacementPolicy):
    """Hybrid policy balancing recency and frequency with exponential decay.

    Score = decayed_freq + alpha * recency_score
    """

    def __init__(self, decay_factor: float = 0.999, alpha: float = 0.5) -> None:
        self.decay_factor = decay_factor
        self.alpha = alpha
        self.score: Dict[Tuple[int, int], float] = collections.defaultdict(float)
        self.last_used: Dict[Tuple[int, int], int] = {}

    def name(self) -> str:
        return "RecencyFrequency-Hybrid"

    def on_access(self, key: Tuple[int, int], step: int) -> None:
        prev_step = self.last_used.get(key, step)
        delta = step - prev_step
        decay = (self.decay_factor ** delta) if delta > 0 else 1.0
        self.score[key] = self.score[key] * decay + 1.0
        self.last_used[key] = step

    def select_eviction(self, candidates: Collection[Tuple[int, int]], step: int) -> Tuple[int, int]:
        def calc_combined_score(k: Tuple[int, int]) -> float:
            lu = self.last_used.get(k, 0)
            delta = step - lu
            decay = self.decay_factor ** delta
            cur_freq_score = self.score.get(k, 0.0) * decay
            recency_score = 1.0 / (1.0 + delta)
            return cur_freq_score + self.alpha * recency_score

        return min(candidates, key=calc_combined_score)

    def on_evict(self, key: Tuple[int, int]) -> None:
        self.score.pop(key, None)
        self.last_used.pop(key, None)

    def reset(self) -> None:
        self.score.clear()
        self.last_used.clear()


class CostAwarePolicy(ReplacementPolicy):
    """Layer-cost aware policy.

    Experts in earlier layers (e.g. layers 0..10) have higher latency sensitivity
    because their misses stall the entire pipeline early. Experts in early layers
    get a priority boost to remain in cache.
    """

    def __init__(self, total_layers: int = 43, early_boost: float = 2.0) -> None:
        self.total_layers = total_layers
        self.early_boost = early_boost
        self.last_used: Dict[Tuple[int, int], int] = {}

    def name(self) -> str:
        return "CostAware-LayerPriority"

    def on_access(self, key: Tuple[int, int], step: int) -> None:
        self.last_used[key] = step

    def select_eviction(self, candidates: Collection[Tuple[int, int]], step: int) -> Tuple[int, int]:
        def retention_value(k: Tuple[int, int]) -> float:
            # Score on AGE (how long since last use), never on the absolute timestamp:
            # multiplying a raw step counter by a weight makes the score depend on where
            # step 0 sits rather than on the recency gap, and collapses to 0 at step 0.
            layer_idx, _ = k
            age = step - self.last_used.get(k, -1)
            layer_weight = 1.0 + self.early_boost * ((self.total_layers - layer_idx) / self.total_layers)
            # Higher weight => more valuable to keep => discount its apparent age.
            return age / layer_weight

        # Evict the entry with the greatest weight-adjusted age.
        return max(candidates, key=retention_value)

    def on_evict(self, key: Tuple[int, int]) -> None:
        self.last_used.pop(key, None)

    def reset(self) -> None:
        self.last_used.clear()


class BeladyPolicy(ReplacementPolicy):
    """Belady's Optimal Replacement Policy (Oracle / OPT).

    Requires full future access sequence. Evicts the expert whose next access is furthest
    in the future (or infinity if never accessed again).
    """

    def __init__(self, future_accesses: Optional[List[Tuple[int, int]]] = None) -> None:
        self.future_map: Dict[Tuple[int, int], collections.deque[int]] = collections.defaultdict(collections.deque)
        if future_accesses:
            self.set_future_accesses(future_accesses)

    def name(self) -> str:
        return "Belady-Oracle"

    def set_future_accesses(self, future_accesses: List[Tuple[int, int]]) -> None:
        self._future_set = True
        self.future_map.clear()
        for step, key in enumerate(future_accesses):
            self.future_map[key].append(step)

    def on_access(self, key: Tuple[int, int], step: int) -> None:
        q = self.future_map.get(key)
        if q:
            while q and q[0] <= step:
                q.popleft()

    def select_eviction(self, candidates: Collection[Tuple[int, int]], step: int) -> Tuple[int, int]:
        if not getattr(self, "_future_set", False):
            # Without the future map every key looks "never reused again" and Belady
            # silently degrades into an arbitrary choice that scores WORSE than LRU --
            # while still being labelled an oracle. Fail loudly instead. Belady is
            # inherently offline; it cannot be used on an incremental access stream.
            raise RuntimeError(
                "BeladyPolicy requires set_future_accesses() before use. It is an "
                "offline oracle and cannot serve an online/incremental access stream."
            )

        def next_access_step(k: Tuple[int, int]) -> int:
            q = self.future_map.get(k)
            if q:
                while q and q[0] <= step:
                    q.popleft()
            if not q:
                return 999_999_999
            return q[0]

        return max(candidates, key=next_access_step)

    def on_evict(self, key: Tuple[int, int]) -> None:
        pass

    def reset(self) -> None:
        self._future_set = False
        self.future_map.clear()


class TinyLFUAdmissionPolicy(ReplacementPolicy):
    """Frequency-gated admission over a base eviction policy (a TinyLFU variant).

    The problem this solves: at the target configuration a single K=5 verification
    pass touches ~456 distinct experts against a 339-expert VRAM tier. Under any
    pure recency policy the pass sweeps the tier clean every time, so the hot
    experts that Zipf routing makes worth keeping are evicted by one-off experts
    that will not be seen again. The tier thrashes by construction and contributes
    almost nothing (~23 hits per pass under LRU).

    The fix is admission control rather than a smarter eviction order: when the tier
    is full, only displace the victim if the newcomer has been seen MORE often than
    the victim. One-off experts then stream through the lower tier without
    disturbing residency, while genuinely hot experts still earn their way in.

    Frequency is counted with periodic halving (the "aging" step in TinyLFU), so an
    expert that was hot early cannot hold a slot forever -- without this, frequency
    policies calcify around whatever appeared first.
    """

    def __init__(
        self,
        base_policy: Optional[ReplacementPolicy] = None,
        aging_period: int = 100_000,
    ) -> None:
        self.base = base_policy if base_policy is not None else LRUPolicy()
        self.freq: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        self.aging_period = aging_period
        self._accesses = 0

    def name(self) -> str:
        return f"TinyLFU-admit({self.base.name()})"

    def on_access(self, key: Tuple[int, int], step: int) -> None:
        self.freq[key] += 1
        self._accesses += 1
        if self.aging_period > 0 and self._accesses % self.aging_period == 0:
            # Halve every counter so recent behaviour dominates old behaviour.
            for k in list(self.freq):
                self.freq[k] >>= 1
                if self.freq[k] == 0:
                    del self.freq[k]
        self.base.on_access(key, step)

    def select_eviction(
        self, candidates: Collection[Tuple[int, int]], step: int
    ) -> Tuple[int, int]:
        return self.base.select_eviction(candidates, step)

    def should_admit(
        self, key: Tuple[int, int], victim: Tuple[int, int], step: int
    ) -> bool:
        # Strictly greater: ties favour the incumbent, since displacing costs a
        # transfer and an equally-hot victim is no improvement.
        return self.freq.get(key, 0) > self.freq.get(victim, 0)

    def on_evict(self, key: Tuple[int, int]) -> None:
        self.base.on_evict(key)

    def reset(self) -> None:
        self.freq.clear()
        self._accesses = 0
        self.base.reset()
