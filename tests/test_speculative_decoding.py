"""Unit tests for speculative decoding engine and MoE batch verification."""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "tools" / "roofline"))
sys.path.insert(0, str(ROOT_DIR / "tools" / "trace_replay"))
sys.path.insert(0, str(ROOT_DIR / "tools" / "speculative_decoding"))

from draft_simulator import SpeculativeEngine
from model import FLASH
from policy import LRUPolicy
from speculative_policy import SpeculativeAwarePolicy


class TestSpeculativeDecoding(unittest.TestCase):

    def test_speculative_candidate_verification(self) -> None:
        """Test speculative candidate generation and acceptance accounting."""
        engine = SpeculativeEngine(draft_k=4, acceptance_prob=0.80, seed=123)
        stats = engine.run_simulation(target_token_count=50, cache_hit_rate=0.70)

        self.assertGreater(stats.total_generated_tokens, 0)
        self.assertGreater(stats.target_verification_passes, 0)
        self.assertGreater(stats.acceptance_rate, 0.0)
        self.assertGreater(stats.effective_accepted_tok_s, 0.0)

    def test_expert_deduplication_reduces_raw_fetches(self) -> None:
        """Test that batch verification deduplicates expert reads across candidate tokens."""
        engine = SpeculativeEngine(draft_k=5, acceptance_prob=0.75, expert_locality_overlap=0.85, seed=42)
        stats = engine.run_simulation(target_token_count=100, cache_hit_rate=0.0)

        # Raw accesses = num_tokens * 258 accesses
        # Unique fetched experts should be substantially less than raw accesses due to batch deduplication
        self.assertLess(stats.unique_experts_fetched, stats.raw_expert_accesses)

    def test_speculative_policy_pinning(self) -> None:
        """Test that SpeculativeAwarePolicy pins predicted draft candidate experts."""
        base_policy = LRUPolicy()
        spec_policy = SpeculativeAwarePolicy(base_policy=base_policy)

        # Access experts (0,1), (0,2), (0,3)
        spec_policy.on_access((0, 1), step=1)
        spec_policy.on_access((0, 2), step=2)
        spec_policy.on_access((0, 3), step=3)

        # Pin (0,1) as a predicted draft expert
        spec_policy.set_draft_predictions([(0, 1)])

        # Select eviction among candidates [(0,1), (0,2), (0,3)]
        # LRU would normally pick (0,1) because step=1. But (0,1) is pinned, so it must pick (0,2)!
        evicted = spec_policy.select_eviction([(0, 1), (0, 2), (0, 3)], step=4)
        self.assertEqual(evicted, (0, 2))

    def test_higher_acceptance_boosts_throughput(self) -> None:
        """Test that higher draft acceptance rate monotonically increases effective tok/s."""
        eng_low = SpeculativeEngine(draft_k=4, acceptance_prob=0.40, seed=999)
        stats_low = eng_low.run_simulation(target_token_count=100, cache_hit_rate=0.70)

        eng_high = SpeculativeEngine(draft_k=4, acceptance_prob=0.85, seed=999)
        stats_high = eng_high.run_simulation(target_token_count=100, cache_hit_rate=0.70)

        self.assertGreater(
            stats_high.effective_accepted_tok_s,
            stats_low.effective_accepted_tok_s,
            f"Higher acceptance ({stats_high.effective_accepted_tok_s:.2f}) should beat low ({stats_low.effective_accepted_tok_s:.2f})",
        )


if __name__ == "__main__":
    unittest.main()
