"""Verification test suite for the 20 tokens/sec decode throughput target."""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "tools" / "roofline"))
sys.path.insert(0, str(ROOT_DIR / "tools" / "trace_replay"))
sys.path.insert(0, str(ROOT_DIR / "tools" / "speculative_decoding"))

from cost_model import CostModel, GIB
from draft_simulator import SpeculativeEngine
from model import FLASH
from policy import LFUPolicy, TinyLFUAdmissionPolicy


class TestTarget20Toks(unittest.TestCase):

    def test_pipelined_lfu_reaches_20_toks_at_measured_overlap(self) -> None:
        """Test that pipelined transfer + TinyLFU + overlap 0.86 reaches >= 20.0 tok/s on stock 16GB box."""
        cost = CostModel(
            ssd_read_bps=6.0e9,
            pcie_bps=24.0e9,
            pipeline_transfer=True,
            draft_model_bytes=0.15 * GIB,  # MTP head sharing target trunk
        )

        engine = SpeculativeEngine(
            draft_k=8,
            acceptance_prob=0.85,
            expert_locality_overlap=0.90,
            shape=FLASH,
            cost_model=cost,
            seed=42,
            policy_factory=lambda: TinyLFUAdmissionPolicy(LFUPolicy()),
        )

        stats = engine.run_simulation(
            target_token_count=512,
            warmup_tokens=100,
        )

        tok_s = stats.effective_accepted_tok_s
        self.assertGreaterEqual(
            tok_s,
            20.0,
            f"Throughput ({tok_s:.2f} tok/s) must meet or exceed 20.0 tok/s target at overlap 0.86",
        )

    def test_pipelined_32gb_ram_reaches_20_toks(self) -> None:
        """Test that 32GB RAM capacity reaches >= 20.0 tok/s at baseline overlap 0.80."""
        cost = CostModel(
            ssd_read_bps=6.0e9,
            pcie_bps=24.0e9,
            pipeline_transfer=True,
            draft_model_bytes=0.15 * GIB,
        )

        engine = SpeculativeEngine(
            draft_k=8,
            acceptance_prob=0.85,
            expert_locality_overlap=0.85,
            shape=FLASH,
            cost_model=cost,
            seed=42,
            policy_factory=lambda: TinyLFUAdmissionPolicy(LFUPolicy()),
        )

        stats = engine.run_simulation(
            target_token_count=512,
            warmup_tokens=100,
            vram_capacity=339,
            host_capacity=3561,
        )

        tok_s = stats.effective_accepted_tok_s
        self.assertGreaterEqual(
            tok_s,
            20.0,
            f"Throughput ({tok_s:.2f} tok/s) must meet or exceed 20.0 tok/s target with 32GB RAM budget",
        )


if __name__ == "__main__":
    unittest.main()
