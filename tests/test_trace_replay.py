"""Unit tests for offline trace replay and cache replacement simulator."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add search paths for tools
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "tools" / "roofline"))
sys.path.insert(0, str(ROOT_DIR / "tools" / "trace_replay"))

from model import FLASH
from policy import (
    BeladyPolicy,
    CostAwarePolicy,
    HybridRecencyFrequencyPolicy,
    LFUPolicy,
    LRUKPolicy,
    LRUPolicy,
)
from roofline import Machine, plan_budget
from simulator import CacheSimulator
from trace import Trace, generate_synthetic_trace


class TestTraceReplay(unittest.TestCase):

    def test_trace_serialization(self) -> None:
        """Test saving and loading trace to/from JSONL."""
        t1 = generate_synthetic_trace(n_tokens=10, n_layers=2, n_experts=8, n_expert_used=2, seed=123)
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as tmp:
            tmp_path = tmp.name

        try:
            t1.save_jsonl(tmp_path)
            t2 = Trace.load_jsonl(tmp_path, n_layers=2, n_experts=8, n_expert_used=2)
            self.assertEqual(t1.total_tokens(), t2.total_tokens())
            self.assertEqual(t1.total_accesses(), t2.total_accesses())
            for token1, token2 in zip(t1.tokens, t2.tokens):
                self.assertEqual(token1.token_idx, token2.token_idx)
                self.assertEqual(token1.accesses, token2.accesses)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_lru_policy_eviction(self) -> None:
        """Test that LRU evicts the least recently accessed candidate."""
        p = LRUPolicy()
        p.on_access((0, 1), step=1)
        p.on_access((0, 2), step=2)
        p.on_access((0, 3), step=3)
        p.on_access((0, 1), step=4)  # (0,1) refreshed

        evicted = p.select_eviction([(0, 1), (0, 2), (0, 3)], step=5)
        self.assertEqual(evicted, (0, 2))

    def test_lfu_policy_eviction(self) -> None:
        """Test that LFU evicts candidate with lowest access frequency."""
        p = LFUPolicy()
        p.on_access((0, 1), step=1)
        p.on_access((0, 1), step=2)
        p.on_access((0, 2), step=3)

        evicted = p.select_eviction([(0, 1), (0, 2)], step=4)
        self.assertEqual(evicted, (0, 2))

    def test_belady_optimality_over_lru(self) -> None:
        """Test that Belady's Oracle achieves >= hit rate than LRU on any access sequence."""
        trace = generate_synthetic_trace(n_tokens=50, n_layers=4, n_experts=32, n_expert_used=4, zipf_s=1.0, seed=42)

        sim_lru = CacheSimulator(vram_capacity=8, host_capacity=0, shape=FLASH, policy=LRUPolicy())
        stats_lru = sim_lru.run(trace)

        sim_belady = CacheSimulator(vram_capacity=8, host_capacity=0, shape=FLASH, policy=BeladyPolicy())
        stats_belady = sim_belady.run(trace)

        self.assertGreaterEqual(
            stats_belady.combined_hit_rate,
            stats_lru.combined_hit_rate,
            f"Belady hit rate ({stats_belady.combined_hit_rate:.4f}) should be >= LRU ({stats_lru.combined_hit_rate:.4f})",
        )

    def test_multi_tier_stats_accounting(self) -> None:
        """Test two-tier VRAM + Host RAM accounting and byte counts."""
        # 1 token, 1 layer, accesses (0,1), (0,2), (0,3)
        t = Trace(n_layers=1, n_experts=10, n_expert_used=3)
        t.add_token(0, [(0, 1), (0, 2), (0, 3)])
        # Repeat same accesses
        t.add_token(1, [(0, 1), (0, 2), (0, 3)])

        # VRAM cap = 2, Host RAM cap = 2
        sim = CacheSimulator(vram_capacity=2, host_capacity=2, shape=FLASH, policy=LRUPolicy())
        stats = sim.run(t)

        # Total accesses = 6
        self.assertEqual(stats.total_accesses, 6)
        # Token 0 accesses (0,1), (0,2), (0,3) -> 3 SSD misses
        # VRAM capacity is 2, so (0,1) gets demoted to Host RAM when (0,3) arrives.
        # Token 1 accesses (0,1), (0,2), (0,3) -> each item hit in Host RAM as it cycles through.
        self.assertEqual(stats.ssd_misses, 3)
        self.assertEqual(stats.host_hits, 3)
        self.assertEqual(stats.vram_hits, 0)
        self.assertAlmostEqual(stats.combined_hit_rate, 3.0 / 6.0)

    def test_zero_capacity_behavior(self) -> None:
        """Test cache simulator with 0 capacity."""
        t = generate_synthetic_trace(n_tokens=5, n_layers=2, n_experts=8, n_expert_used=2)
        sim = CacheSimulator(vram_capacity=0, host_capacity=0, shape=FLASH, policy=LRUPolicy())
        stats = sim.run(t)
        self.assertEqual(stats.vram_hits, 0)
        self.assertEqual(stats.host_hits, 0)
        self.assertEqual(stats.ssd_misses, stats.total_accesses)
        self.assertEqual(stats.combined_hit_rate, 0.0)

    def test_reproducibility(self) -> None:
        """Test that synthetic trace generation is deterministic with fixed seed."""
        t1 = generate_synthetic_trace(n_tokens=20, seed=999)
        t2 = generate_synthetic_trace(n_tokens=20, seed=999)
        self.assertEqual(t1.total_accesses(), t2.total_accesses())
        for tok1, tok2 in zip(t1.tokens, t2.tokens):
            self.assertEqual(tok1.accesses, tok2.accesses)


if __name__ == "__main__":
    unittest.main()
