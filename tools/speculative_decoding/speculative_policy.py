"""Speculative-Aware Cache Replacement Policy for MoE expert caching.

Prioritizes experts predicted by draft candidate trees, pinning them in VRAM/Host RAM
prior to target model verification passes to prevent preemption.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Collection, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trace_replay"))

from policy import ReplacementPolicy


class SpeculativeAwarePolicy(ReplacementPolicy):
    """Speculative-aware replacement policy that pins predicted draft candidate experts."""

    def __init__(self, base_policy: ReplacementPolicy, speculative_boost: float = 1e6) -> None:
        self.base_policy = base_policy
        self.speculative_boost = speculative_boost
        self.pinned_draft_experts: Set[Tuple[int, int]] = set()

    def name(self) -> str:
        return f"Speculative-{self.base_policy.name()}"

    def set_draft_predictions(self, predicted_experts: Collection[Tuple[int, int]]) -> None:
        """Register experts predicted by draft candidate trees for the upcoming verification pass."""
        self.pinned_draft_experts = set(predicted_experts)

    def clear_draft_predictions(self) -> None:
        self.pinned_draft_experts.clear()

    def on_access(self, key: Tuple[int, int], step: int) -> None:
        self.base_policy.on_access(key, step)

    def select_eviction(self, candidates: Collection[Tuple[int, int]], step: int) -> Tuple[int, int]:
        # Filter out pinned draft experts if possible
        unpinned = [c for c in candidates if c not in self.pinned_draft_experts]
        if unpinned:
            return self.base_policy.select_eviction(unpinned, step)
        # If all candidates are pinned, fall back to base policy
        return self.base_policy.select_eviction(candidates, step)

    def on_evict(self, key: Tuple[int, int]) -> None:
        self.base_policy.on_evict(key)

    def reset(self) -> None:
        self.pinned_draft_experts.clear()
        self.base_policy.reset()
