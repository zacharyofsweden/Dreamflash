"""Multi-tier expert cache simulator for trace replay evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

from model import FLASH, Shape
from policy import BeladyPolicy, ReplacementPolicy
from trace import Trace


@dataclass
class SimulationStats:
    """Statistics gathered during trace simulation."""
    total_accesses: int = 0
    vram_hits: int = 0
    host_hits: int = 0
    ssd_misses: int = 0

    vram_evictions: int = 0
    host_evictions: int = 0

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
            key = access.key
            stats.total_accesses += 1

            if key in self.vram_cache:
                # Tier 1 HIT
                stats.vram_hits += 1
                if self.policy:
                    self.policy.on_access(key, step)

            elif key in self.host_cache:
                # Tier 2 HIT (Host RAM)
                stats.host_hits += 1
                if self.policy:
                    self.policy.on_access(key, step)

                # Move from Host RAM to VRAM
                self.host_cache.remove(key)
                self._insert_into_vram(key, step, stats)

            else:
                # SSD MISS
                stats.ssd_misses += 1
                if self.policy:
                    self.policy.on_access(key, step)

                # Fetch into VRAM (or Host RAM if VRAM capacity is 0)
                if self.vram_capacity > 0:
                    self._insert_into_vram(key, step, stats)
                elif self.host_capacity > 0:
                    self._insert_into_host(key, step, stats)

        return stats

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
