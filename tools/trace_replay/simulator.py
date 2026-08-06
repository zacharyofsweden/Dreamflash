"""Multi-tier expert cache simulator for trace replay evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional

from model import FLASH, Shape
from policy import BeladyPolicy, ReplacementPolicy
from trace import Trace


class CacheTier(Enum):
    """Which tier served an access. Determines what bandwidth the bytes cross."""
    VRAM = "vram"   # already resident; no transfer
    HOST = "host"   # crosses PCIe
    SSD = "ssd"     # crosses SSD, then PCIe


@dataclass
class SimulationStats:
    """Statistics gathered during trace simulation."""
    total_accesses: int = 0
    vram_hits: int = 0
    host_hits: int = 0
    ssd_misses: int = 0

    vram_evictions: int = 0
    host_evictions: int = 0
    vram_admission_rejections: int = 0
    """Times a newcomer was judged not worth displacing the VRAM victim."""

    bytes_per_expert: float = 0.0

    @property
    def total_hits(self) -> int:
        return self.vram_hits + self.host_hits

    @property
    def vram_hit_rate(self) -> float:
        return self.vram_hits / self.total_accesses if self.total_accesses > 0 else 0.0

    @property
    def combined_hit_rate(self) -> float:
        return self.total_hits / self.total_accesses if self.total_accesses > 0 else 0.0

    @property
    def ssd_bytes_transferred(self) -> float:
        return self.ssd_misses * self.bytes_per_expert

    @property
    def pcie_bytes_transferred(self) -> float:
        # Every host hit and SSD miss must cross PCIe to VRAM
        return (self.host_hits + self.ssd_misses) * self.bytes_per_expert


class CacheSimulator:
    """Simulator for a two-tier (VRAM + Host RAM) expert cache system."""

    def __init__(
        self,
        vram_capacity: int,
        host_capacity: int = 0,
        shape: Shape = FLASH,
        policy: Optional[ReplacementPolicy] = None,
    ) -> None:
        self.vram_capacity = max(0, vram_capacity)
        self.host_capacity = max(0, host_capacity)
        self.shape = shape
        self.policy = policy

        self.vram_cache: Set[Tuple[int, int]] = set()
        self.host_cache: Set[Tuple[int, int]] = set()

    def run(self, trace: Trace) -> SimulationStats:
        """Run the simulation over a given trace and return performance metrics."""
        access_list = list(trace.iter_accesses())
        bytes_per_exp = self.shape.bytes_per_routed_expert()

        # If using Belady policy, initialize future access map
        if isinstance(self.policy, BeladyPolicy):
            flat_accesses = [a.key for a in access_list]
            self.policy.set_future_accesses(flat_accesses)
        elif self.policy:
            self.policy.reset()

        stats = SimulationStats(bytes_per_expert=bytes_per_exp)
        self.vram_cache.clear()
        self.host_cache.clear()

        for step, access in enumerate(access_list):
            self.access(access.key, step, stats)

        return stats

    def access(
        self, key: Tuple[int, int], step: int, stats: SimulationStats
    ) -> "CacheTier":
        """Serve one expert access against the live cache, updating residency and stats.

        Exposed separately from run() so callers that generate accesses incrementally
        (the speculative engine) can drive the same cache rather than approximating
        it with an independent hit-rate parameter.
        """
        stats.total_accesses += 1

        if key in self.vram_cache:
            # Tier 1 HIT
            stats.vram_hits += 1
            if self.policy:
                self.policy.on_access(key, step)
            return CacheTier.VRAM

        if key in self.host_cache:
            # Tier 2 HIT (Host RAM)
            stats.host_hits += 1
            if self.policy:
                self.policy.on_access(key, step)

            # Move from Host RAM to VRAM
            self.host_cache.remove(key)
            self._insert_into_vram(key, step, stats)
            return CacheTier.HOST

        # SSD MISS
        stats.ssd_misses += 1
        if self.policy:
            self.policy.on_access(key, step)

        # Fetch into VRAM (or Host RAM if VRAM capacity is 0)
        if self.vram_capacity > 0:
            self._insert_into_vram(key, step, stats)
        elif self.host_capacity > 0:
            self._insert_into_host(key, step, stats)
        return CacheTier.SSD

    def reset(self) -> None:
        """Drop all residency. Policy state is left to the caller."""
        self.vram_cache.clear()
        self.host_cache.clear()

    def _insert_into_vram(
        self, key: Tuple[int, int], step: int, stats: SimulationStats
    ) -> None:
        if self.vram_capacity == 0:
            return

        if len(self.vram_cache) >= self.vram_capacity and key not in self.vram_cache:
            # Evict from VRAM
            if self.policy:
                evict_key = self.policy.select_eviction(self.vram_cache, step)
            else:
                evict_key = next(iter(self.vram_cache))

            # Admission control: a scan-resistant policy may decline to displace the
            # victim, leaving the newcomer in the lower tier instead. Without this,
            # a pass whose working set exceeds the tier evicts everything worth
            # keeping on every pass.
            if self.policy and not self.policy.should_admit(key, evict_key, step):
                stats.vram_admission_rejections += 1
                if self.host_capacity > 0:
                    self._insert_into_host(key, step, stats)
                return

            self.vram_cache.remove(evict_key)
            stats.vram_evictions += 1

            # Move evicted key down to Host RAM tier if host capacity > 0
            if self.host_capacity > 0:
                self._insert_into_host(evict_key, step, stats)
            elif self.policy:
                self.policy.on_evict(evict_key)

        self.vram_cache.add(key)

    def _insert_into_host(
        self, key: Tuple[int, int], step: int, stats: SimulationStats
    ) -> None:
        if self.host_capacity == 0:
            return

        if len(self.host_cache) >= self.host_capacity and key not in self.host_cache:
            # Evict from Host RAM
            if self.policy:
                evict_key = self.policy.select_eviction(self.host_cache, step)
            else:
                evict_key = next(iter(self.host_cache))

            self.host_cache.remove(evict_key)
            stats.host_evictions += 1
            if self.policy:
                self.policy.on_evict(evict_key)

        self.host_cache.add(key)
