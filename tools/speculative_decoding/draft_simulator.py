"""Speculative Decoding Engine for MoE Expert Streaming.

Simulates drafting candidate tokens (K tokens) and batch verification in target model:
- Deduplicates expert reads across draft candidate tokens within a batch pass.
- Computes acceptance rates, effective decode throughput, and SSD bandwidth savings.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Set, Tuple

from model import FLASH, Shape


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
    unique_experts_fetched: int = 0
    raw_expert_accesses: int = 0

    bytes_per_expert: float = 0.0
    ssd_bandwidth_bps: float = 6.0e9

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / self.total_draft_tokens if self.total_draft_tokens > 0 else 0.0

    @property
    def effective_accepted_tok_s(self) -> float:
        """Effective accepted user-visible tokens per second under speculative verification."""
        if self.target_verification_passes == 0:
            return 0.0

        avg_unique_experts_per_pass = self.unique_experts_fetched / self.target_verification_passes
        pass_bytes = avg_unique_experts_per_pass * self.bytes_per_expert
        pass_time_sec = pass_bytes / self.ssd_bandwidth_bps if self.ssd_bandwidth_bps > 0 else 1.0

        accepted_per_pass = 1.0 + (self.accepted_draft_tokens / self.target_verification_passes)
        return accepted_per_pass / pass_time_sec if pass_time_sec > 0 else 0.0

    @property
    def speedup_factor(self) -> float:
        """Speedup factor compared to baseline cold autoregressive decoding."""
        baseline_cold_bpt = FLASH.routed_bytes_per_token()
        baseline_tok_s = self.ssd_bandwidth_bps / baseline_cold_bpt if baseline_cold_bpt > 0 else 1.0
        return self.effective_accepted_tok_s / baseline_tok_s if baseline_tok_s > 0 else 1.0


class SpeculativeEngine:
    """Simulates speculative decoding candidate generation and target MoE batch verification."""

    def __init__(
        self,
        draft_k: int = 4,
        acceptance_prob: float = 0.65,
        expert_locality_overlap: float = 0.80,
        shape: Shape = FLASH,
        ssd_gbps: float = 6.0,
        seed: int = 42,
    ) -> None:
        self.draft_k = draft_k
        self.acceptance_prob = acceptance_prob
        self.expert_locality_overlap = expert_locality_overlap
        self.shape = shape
        self.ssd_bandwidth_bps = ssd_gbps * 1e9
        self.rng = random.Random(seed)

    def run_simulation(
        self,
        target_token_count: int = 200,
        cache_hit_rate: float = 0.70,
    ) -> SpeculativeStats:
        stats = SpeculativeStats(
            bytes_per_expert=self.shape.bytes_per_routed_expert(),
            ssd_bandwidth_bps=self.ssd_bandwidth_bps,
        )

        tokens_produced = 0
        step_idx = 0

        while tokens_produced < target_token_count:
            stats.target_verification_passes += 1

            k_candidates: List[CandidateToken] = []
            accepted_in_pass = 0

            # Base expert set for candidate 0 in this pass
            base_pass_experts: List[List[int]] = []
            for l_i in range(self.shape.n_layer):
                base_pass_experts.append(self.rng.sample(range(self.shape.n_expert), self.shape.n_expert_used))

            for d_i in range(self.draft_k):
                accesses = []
                for l_i in range(self.shape.n_layer):
                    prev_layer_experts = base_pass_experts[l_i]
                    cur_layer_experts = []

                    for exp in prev_layer_experts:
                        if self.rng.random() < self.expert_locality_overlap:
                            cur_layer_experts.append(exp)
                        else:
                            # Pick new expert
                            new_exp = self.rng.randint(0, self.shape.n_expert - 1)
                            while new_exp in cur_layer_experts:
                                new_exp = self.rng.randint(0, self.shape.n_expert - 1)
                            cur_layer_experts.append(new_exp)

                    for exp in cur_layer_experts:
                        accesses.append((l_i, exp))

                accepted = self.rng.random() < self.acceptance_prob
                k_candidates.append(CandidateToken(
                    token_idx=step_idx,
                    draft_id=d_i,
                    accesses=accesses,
                    is_accepted=accepted,
                ))

                if accepted and accepted_in_pass == d_i:
                    accepted_in_pass += 1
                else:
                    break

            stats.total_draft_tokens += len(k_candidates)
            stats.accepted_draft_tokens += accepted_in_pass

            # Target model verifies candidate tokens in parallel
            unique_experts_in_pass: Set[Tuple[int, int]] = set()
            for cand in k_candidates:
                for acc in cand.accesses:
                    unique_experts_in_pass.add(acc)
                    stats.raw_expert_accesses += 1

            # Experts served from VRAM/RAM cache don't require SSD reads
            ssd_miss_count = sum(
                1 for _ in unique_experts_in_pass if self.rng.random() >= cache_hit_rate
            )
            stats.unique_experts_fetched += ssd_miss_count

            total_accepted_pass = accepted_in_pass + 1
            tokens_produced += total_accepted_pass
            stats.total_generated_tokens += total_accepted_pass
            step_idx += total_accepted_pass

        return stats
