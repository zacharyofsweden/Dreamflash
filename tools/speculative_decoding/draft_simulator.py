"""Speculative Decoding Engine for MoE Expert Streaming.

Simulates drafting K candidate tokens and verifying them in one target-model pass:
- Deduplicates expert reads across draft candidates within a pass.
- Serves those reads against a REAL two-tier LRU cache (the same CacheSimulator the
  trace replay uses), so the hit rate is an emergent property of capacity and access
  pattern rather than a dial the caller sets.
- Charges wall-clock time through CostModel, which accounts for VRAM, PCIe, SSD,
  kernel launch, and the draft model itself.

Two things this still does NOT model, and they bound how much its output is worth:
the expert access stream is synthetic (see `expert_locality_overlap` below), and no
real routing trace exists in this repo to replace it with.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trace_replay"))

from cost_model import CostModel
from model import FLASH, Shape
from policy import LFUPolicy, LRUPolicy, ReplacementPolicy
from simulator import CacheSimulator, CacheTier, SimulationStats


@dataclass
class CandidateToken:
    token_idx: int
    draft_id: int
    accesses: List[Tuple[int, int]]  # (layer_idx, expert_idx)
    is_accepted: bool = False


@dataclass
class SpeculativeStats:
    total_generated_tokens: int = 0
    total_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    target_verification_passes: int = 0
    raw_expert_accesses: int = 0

    # Unique-per-pass expert reads, split by the tier that served them. These are
    # summed over all passes.
    vram_experts: int = 0
    host_experts: int = 0
    ssd_experts: int = 0

    # Accumulated wall clock from the cost model.
    serialized_seconds: float = 0.0
    ideal_seconds: float = 0.0

    bytes_per_expert: float = 0.0
    cost_model: CostModel = field(default_factory=CostModel)
    shape: Shape = FLASH

    @property
    def unique_experts_fetched(self) -> int:
        """Unique experts that had to come from somewhere other than VRAM."""
        return self.host_experts + self.ssd_experts

    @property
    def acceptance_rate(self) -> float:
        return (
            self.accepted_draft_tokens / self.total_draft_tokens
            if self.total_draft_tokens > 0
            else 0.0
        )

    @property
    def cache_hit_rate(self) -> float:
        """Emergent combined hit rate over unique per-pass expert reads.

        An OUTPUT of the simulation, not an input.
        """
        total = self.vram_experts + self.host_experts + self.ssd_experts
        return (self.vram_experts + self.host_experts) / total if total > 0 else 0.0

    @property
    def accepted_tok_s(self) -> float:
        """Accepted user-visible tok/s, serialized (pessimistic) end of the range."""
        if self.serialized_seconds <= 0:
            return 0.0
        return self.total_generated_tokens / self.serialized_seconds

    @property
    def accepted_tok_s_ideal(self) -> float:
        """Accepted tok/s assuming transfer hides perfectly behind compute."""
        if self.ideal_seconds <= 0:
            return 0.0
        return self.total_generated_tokens / self.ideal_seconds

    # Retained under the old name so existing callers keep working; reports the
    # conservative end deliberately.
    @property
    def effective_accepted_tok_s(self) -> float:
        return self.accepted_tok_s

    @property
    def baseline_tok_s(self) -> float:
        """Non-speculative cold-cache decode under the SAME cost model."""
        return self.cost_model.cold_baseline_tok_s(
            self.shape.routed_bytes_per_token(), self.shape.n_layer
        )

    @property
    def speedup_factor(self) -> float:
        base = self.baseline_tok_s
        return self.accepted_tok_s / base if base > 0 else 0.0


class SpeculativeEngine:
    """Simulates speculative candidate generation and target MoE batch verification."""

    def __init__(
        self,
        draft_k: int = 4,
        acceptance_prob: float = 0.65,
        expert_locality_overlap: float = 0.80,
        shape: Shape = FLASH,
        ssd_gbps: Optional[float] = None,
        seed: int = 42,
        cost_model: Optional[CostModel] = None,
        zipf_s: float = 1.2,
        policy_factory: Optional[Callable[[], ReplacementPolicy]] = None,
        draft_branches: int = 1,
    ) -> None:
        self.draft_k = draft_k
        self.acceptance_prob = acceptance_prob
        # Tree-structured drafting (Medusa/EAGLE style). With draft_branches=1 this
        # is a linear chain: one candidate per position, and the first rejection ends
        # the pass. With b>1 the drafter proposes b alternatives at each position and
        # the target verifies the whole tree in a single pass, so the position
        # advances if ANY branch matches.
        #
        # MEASURED RESULT: on this workload widening the tree LOSES, badly. At the
        # assumed 0.80 overlap, b=2 drops throughput from 14.84 to 9.23 tok/s. The
        # appeal was that siblings should share experts and so cost little extra I/O,
        # but they do not: each sibling keeps 80% of the base token's experts and
        # replaces a DIFFERENT 20%, so distinct reads per pass grow faster (+87%)
        # than accepted tokens (+41%). Only at >=0.99 overlap does b=2 win.
        #
        # Kept because it is the honest way to record that negative result, and
        # because the conclusion flips if real routing overlap turns out to be very
        # high. Leave at 1 unless you have measured overlap and it is extreme.
        if draft_branches < 1:
            raise ValueError("draft_branches must be >= 1")
        self.draft_branches = draft_branches
        # Routing skew for the base token of each pass. Matches the default in
        # trace_replay/trace.py on purpose: this engine previously drew base experts
        # UNIFORMLY, so the two tools in this repo disagreed about the access
        # distribution and reported wildly different hit rates for the same cache.
        # As with trace.py, this exponent is an uncited assumption -- hit rate is
        # highly sensitive to it, so vary it before trusting any conclusion.
        self.zipf_s = zipf_s
        weights = [1.0 / (i ** zipf_s) for i in range(1, shape.n_expert + 1)]
        total_w = sum(weights)
        self._expert_probs = [w / total_w for w in weights]
        self._expert_pool = list(range(shape.n_expert))
        # SYNTHETIC: the probability that a draft candidate routes to the same expert
        # as the pass's base token. This constant is what produces the cross-candidate
        # deduplication the engine reports -- the saving is injected here, not measured.
        self.expert_locality_overlap = expert_locality_overlap
        self.shape = shape
        self.rng = random.Random(seed)

        # A factory, not an instance: policies carry state, so each run needs a fresh
        # one or residency history leaks between runs.
        #
        # LFU rather than LRU by default. A verification pass touches ~456 distinct
        # experts against a 339-expert VRAM tier, so any pure-recency policy is swept
        # clean every pass and the tier contributes almost nothing (~23 hits/pass vs
        # ~105 under LFU). Since VRAM hits are the only ones that avoid PCIe entirely,
        # that difference is worth ~8% end to end.
        self.policy_factory = policy_factory or LFUPolicy

        # `ssd_gbps` is a convenience override for the common case. Applying it
        # unconditionally would silently discard the ssd_read_bps of a caller-supplied
        # cost model -- which made SSD bandwidth appear to have no effect at all.
        # replace() rather than assignment so we never mutate the caller's object.
        base = cost_model if cost_model is not None else CostModel()
        self.cost_model = (
            replace(base, ssd_read_bps=ssd_gbps * 1e9) if ssd_gbps is not None else base
        )

    def _sample_experts(self) -> List[int]:
        """Draw n_expert_used distinct experts under the Zipf popularity weights."""
        selected: List[int] = []
        while len(selected) < self.shape.n_expert_used:
            choice = self.rng.choices(self._expert_pool, weights=self._expert_probs, k=1)[0]
            if choice not in selected:
                selected.append(choice)
        return selected

    def run_simulation(
        self,
        target_token_count: int = 512,
        vram_capacity: int = 339,
        host_capacity: int = 1744,
        warmup_tokens: int = 512,
    ) -> SpeculativeStats:
        """Run until `target_token_count` accepted tokens have been measured.

        Capacities are in experts. The defaults are what plan_budget() yields for the
        12 GiB + 16 GiB target box at 100K context -- themselves derived from
        placeholder constants, see README.

        `warmup_tokens` are generated first and drive the cache but contribute nothing
        to the reported stats. Without this the measurement is dominated by compulsory
        first-touch misses: the model has 11,008 routed experts and one pass touches
        ~700 of them, so a short run spends most of its time filling a cold cache and
        reports a hit rate well below steady state (36% vs 42%, and 12.4 vs 14.0
        tok/s at the default configuration). Measuring a cold cache and calling it
        throughput is the same error as measuring page cache and calling it disk
        bandwidth.

        The 512-token default matches the repo's own rule that a valid speed claim
        needs at least 512 generated tokens.
        """
        stats = SpeculativeStats(
            bytes_per_expert=self.shape.bytes_per_routed_expert(),
            cost_model=self.cost_model,
            shape=self.shape,
        )

        cache = CacheSimulator(
            vram_capacity=vram_capacity,
            host_capacity=host_capacity,
            shape=self.shape,
            policy=self.policy_factory(),
        )
        cache_stats = SimulationStats(bytes_per_expert=stats.bytes_per_expert)

        tokens_produced = 0
        step_idx = 0
        warm = max(0, warmup_tokens)

        # Phase 1 warms the cache and is excluded from stats; phase 2 is measured.
        # The measured count is tracked separately rather than derived from the
        # total: a pass emits several tokens at once, so warm-up overshoots its
        # target by a variable amount and subtracting a fixed `warm` would leave
        # the measured window short of what was asked for.
        while stats.total_generated_tokens < target_token_count:
            measuring = tokens_produced >= warm
            if measuring:
                stats.target_verification_passes += 1

            k_candidates: List[CandidateToken] = []
            accepted_in_pass = 0

            # Base expert set for candidate 0 in this pass, drawn with the same Zipf
            # skew trace_replay uses so both tools model one access distribution.
            base_pass_experts: List[List[int]] = []
            for _ in range(self.shape.n_layer):
                base_pass_experts.append(self._sample_experts())

            for d_i in range(self.draft_k):
                position_accepted = False

                # Siblings at this tree position. All of them are verified, so all of
                # their expert reads are charged, whether or not they are accepted.
                for b_i in range(self.draft_branches):
                    accesses = []
                    for l_i in range(self.shape.n_layer):
                        prev_layer_experts = base_pass_experts[l_i]
                        cur_layer_experts: List[int] = []

                        for exp in prev_layer_experts:
                            if self.rng.random() < self.expert_locality_overlap:
                                cur_layer_experts.append(exp)
                            else:
                                new_exp = self.rng.randint(0, self.shape.n_expert - 1)
                                while new_exp in cur_layer_experts:
                                    new_exp = self.rng.randint(0, self.shape.n_expert - 1)
                                cur_layer_experts.append(new_exp)

                        for exp in cur_layer_experts:
                            accesses.append((l_i, exp))

                    accepted = self.rng.random() < self.acceptance_prob
                    position_accepted = position_accepted or accepted
                    k_candidates.append(
                        CandidateToken(
                            token_idx=step_idx,
                            draft_id=d_i * self.draft_branches + b_i,
                            accesses=accesses,
                            is_accepted=accepted,
                        )
                    )

                if position_accepted and accepted_in_pass == d_i:
                    # At least one sibling matched, so the accepted path extends.
                    accepted_in_pass += 1
                else:
                    # No branch at this position matched: the tree is cut here and
                    # everything deeper is discarded.
                    break

            if measuring:
                stats.total_draft_tokens += len(k_candidates)
                stats.accepted_draft_tokens += accepted_in_pass

            # The target verifies all candidates together, so an expert touched by
            # several candidates is read once. Rejected candidates' reads still cost
            # -- they were fetched before the rejection was known.
            unique_experts_in_pass: Set[Tuple[int, int]] = set()
            for cand in k_candidates:
                for acc in cand.accesses:
                    unique_experts_in_pass.add(acc)
                    if measuring:
                        stats.raw_expert_accesses += 1

            pass_vram = pass_host = pass_ssd = 0
            for key in sorted(unique_experts_in_pass):
                tier = cache.access(key, step_idx, cache_stats)
                if tier is CacheTier.VRAM:
                    pass_vram += 1
                elif tier is CacheTier.HOST:
                    pass_host += 1
                else:
                    pass_ssd += 1

            if measuring:
                stats.vram_experts += pass_vram
                stats.host_experts += pass_host
                stats.ssd_experts += pass_ssd

            serialized, ideal = self.cost_model.pass_time_seconds(
                draft_k=len(k_candidates),  # every verified node costs draft compute
                vram_experts=pass_vram,
                host_experts=pass_host,
                ssd_experts=pass_ssd,
                bytes_per_expert=stats.bytes_per_expert,
                n_layers=self.shape.n_layer,
            )
            total_accepted_pass = accepted_in_pass + 1
            tokens_produced += total_accepted_pass
            step_idx += total_accepted_pass

            if measuring:
                stats.serialized_seconds += serialized
                stats.ideal_seconds += ideal
                stats.total_generated_tokens += total_accepted_pass

        return stats
