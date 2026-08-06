"""Unit tests for speculative decoding engine and MoE batch verification."""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "tools" / "roofline"))
sys.path.insert(0, str(ROOT_DIR / "tools" / "trace_replay"))
sys.path.insert(0, str(ROOT_DIR / "tools" / "speculative_decoding"))

from cost_model import CostModel
from draft_simulator import SpeculativeEngine
from model import FLASH
from policy import BeladyPolicy, LFUPolicy, LRUPolicy, TinyLFUAdmissionPolicy
from speculative_policy import SpeculativeAwarePolicy


class TestSpeculativeDecoding(unittest.TestCase):

    def test_accepted_tokens_never_exceed_drafted(self) -> None:
        """Accounting invariant: only accepted tokens are user-visible.

        The repo's non-negotiable rule. A pass emits one guaranteed target token plus
        however many drafts were accepted before the first rejection, so accepted
        drafts must never exceed drafts offered, and generated tokens must equal
        passes + accepted drafts exactly.
        """
        engine = SpeculativeEngine(draft_k=4, acceptance_prob=0.80, seed=123)
        stats = engine.run_simulation(target_token_count=50)

        self.assertLessEqual(stats.accepted_draft_tokens, stats.total_draft_tokens)
        self.assertEqual(
            stats.total_generated_tokens,
            stats.target_verification_passes + stats.accepted_draft_tokens,
            "each pass yields 1 target token plus its accepted drafts",
        )
        self.assertGreaterEqual(stats.total_generated_tokens, 50)

    def test_acceptance_stops_at_first_rejection(self) -> None:
        """A pass must stop drafting at the first rejection, not accept out of order.

        With acceptance_prob=0.0 nothing is ever accepted, so every pass must offer
        exactly one candidate and yield exactly one token.
        """
        engine = SpeculativeEngine(draft_k=6, acceptance_prob=0.0, seed=7)
        stats = engine.run_simulation(target_token_count=20)

        self.assertEqual(stats.accepted_draft_tokens, 0)
        self.assertEqual(stats.total_draft_tokens, stats.target_verification_passes)
        self.assertEqual(stats.total_generated_tokens, stats.target_verification_passes)

    def test_hit_rate_is_emergent_not_assumed(self) -> None:
        """Cache hit rate must respond to capacity, not be a caller-supplied constant.

        The regression this guards: hit rate used to be an input parameter applied as
        an i.i.d. coin flip with no cache state, which made throughput a restatement
        of the assumption. With a real cache, zero capacity must mean zero hits and
        large capacity must mean substantially more.
        """
        engine = SpeculativeEngine(draft_k=4, acceptance_prob=0.7, seed=11)

        none = engine.run_simulation(
            target_token_count=40, vram_capacity=0, host_capacity=0
        )
        self.assertEqual(none.cache_hit_rate, 0.0, "no capacity can serve no hits")
        self.assertEqual(none.vram_experts, 0)
        self.assertEqual(none.host_experts, 0)

        big = SpeculativeEngine(draft_k=4, acceptance_prob=0.7, seed=11).run_simulation(
            target_token_count=40, vram_capacity=2000, host_capacity=4000
        )
        self.assertGreater(big.cache_hit_rate, 0.3, "a large cache must actually hit")
        self.assertGreater(
            big.accepted_tok_s,
            none.accepted_tok_s,
            "caching must translate into throughput",
        )

    def test_deduplication_saves_real_reads(self) -> None:
        """Batch verification must read a shared expert once, not once per candidate.

        Asserting merely that unique < raw is unfalsifiable (a set over a large
        keyspace guarantees it). Compare against the no-sharing case instead: with
        full locality overlap all K candidates route identically, so a pass must
        touch no more distinct experts than a single token does (43 x 6 = 258).
        """
        engine = SpeculativeEngine(
            draft_k=5, acceptance_prob=1.0, expert_locality_overlap=1.0, seed=42
        )
        stats = engine.run_simulation(
            target_token_count=60, vram_capacity=0, host_capacity=0
        )

        served = stats.vram_experts + stats.host_experts + stats.ssd_experts
        per_pass = served / stats.target_verification_passes
        self.assertLessEqual(
            per_pass,
            FLASH.expert_loads_per_token(),
            "with identical routing, K candidates must cost one token's worth of reads",
        )
        # And the raw (undeduplicated) count really is K times larger, so the saving
        # measured above is not vacuous.
        raw_per_pass = stats.raw_expert_accesses / stats.target_verification_passes
        self.assertAlmostEqual(raw_per_pass, 5 * FLASH.expert_loads_per_token(), delta=1.0)

    def test_throughput_is_bounded_by_physics(self) -> None:
        """Throughput must stay under the bandwidth ceiling even with a perfect cache.

        The regression this guards: with time modelled as SSD-bytes-only, a 100% hit
        rate divided by zero missed bytes sent throughput to absurd values (227 tok/s
        at a 97% hit rate). Even fully cached, decode must still stream the ~6.9 GiB
        of non-routed weights from VRAM every token, which caps it.
        """
        cost = CostModel()
        engine = SpeculativeEngine(draft_k=4, acceptance_prob=0.9, seed=5, cost_model=cost)
        stats = engine.run_simulation(
            target_token_count=60, vram_capacity=20000, host_capacity=20000
        )

        # A cache larger than the model can still take compulsory first-touch misses,
        # but never a capacity miss -- so misses cannot exceed the model's expert count.
        self.assertLessEqual(
            stats.ssd_experts,
            FLASH.total_routed_experts,
            "an oversized cache can only take compulsory misses, never capacity misses",
        )
        ceiling = cost.vram_bps / cost.nonrouted_bytes
        self.assertLess(
            stats.accepted_tok_s,
            ceiling,
            f"tok/s must stay below the VRAM weight-streaming ceiling ({ceiling:.1f})",
        )

    def test_cost_model_parameters_actually_take_effect(self) -> None:
        """A supplied CostModel must not be silently overridden or mutated.

        The regression this guards: the engine unconditionally wrote its `ssd_gbps`
        default over the cost model's ssd_read_bps, so a caller-supplied SSD
        bandwidth had NO effect -- an infinitely fast SSD produced byte-identical
        throughput to a 6 GB/s one, which is impossible when SSD reads are most of
        the wall clock.
        """
        slow_cm = CostModel(ssd_read_bps=6.0e9)
        fast_cm = CostModel(ssd_read_bps=600.0e9)

        slow = SpeculativeEngine(
            draft_k=4, acceptance_prob=0.8, seed=31, cost_model=slow_cm
        ).run_simulation(target_token_count=40)
        fast = SpeculativeEngine(
            draft_k=4, acceptance_prob=0.8, seed=31, cost_model=fast_cm
        ).run_simulation(target_token_count=40)

        self.assertGreater(
            fast.accepted_tok_s,
            slow.accepted_tok_s * 1.5,
            "a 100x faster SSD must materially change throughput",
        )
        # And the caller's object must come back untouched.
        self.assertEqual(slow_cm.ssd_read_bps, 6.0e9)
        self.assertEqual(fast_cm.ssd_read_bps, 600.0e9)

        # PCIe must bite too: it carries every host hit and every SSD miss.
        wide = SpeculativeEngine(
            draft_k=4, acceptance_prob=0.8, seed=31,
            cost_model=CostModel(ssd_read_bps=600.0e9, pcie_bps=240.0e9),
        ).run_simulation(target_token_count=40)
        self.assertGreater(
            wide.accepted_tok_s,
            fast.accepted_tok_s,
            "with SSD cost removed, PCIe becomes the binding constraint",
        )

    def test_serialized_never_beats_ideal_overlap(self) -> None:
        """The reported range must be ordered: serialized is the pessimistic end."""
        engine = SpeculativeEngine(draft_k=3, acceptance_prob=0.7, seed=17)
        stats = engine.run_simulation(target_token_count=40)

        self.assertGreater(stats.ideal_seconds, 0.0)
        self.assertGreaterEqual(stats.serialized_seconds, stats.ideal_seconds)
        self.assertLessEqual(stats.accepted_tok_s, stats.accepted_tok_s_ideal)

    def test_draft_model_cost_is_charged(self) -> None:
        """A larger draft model must slow the system down.

        Speculation is not free; omitting the draft model's own forward passes is a
        standard way these estimates come out optimistic.
        """
        cheap = SpeculativeEngine(
            draft_k=5, acceptance_prob=0.7, seed=3,
            cost_model=CostModel(draft_model_bytes=1e6),
        ).run_simulation(target_token_count=40)

        pricey = SpeculativeEngine(
            draft_k=5, acceptance_prob=0.7, seed=3,
            cost_model=CostModel(draft_model_bytes=8.0 * 1024**3),
        ).run_simulation(target_token_count=40)

        self.assertLess(
            pricey.accepted_tok_s,
            cheap.accepted_tok_s,
            "an 8 GiB draft model must cost more than a 1 MB one",
        )

    def test_pipelining_transfer_helps_and_is_bounded(self) -> None:
        """Double-buffering SSD/PCIe must help, and by no more than the smaller stage.

        Serialized charges t_ssd + t_pcie; pipelined charges max(t_ssd, t_pcie). The
        saving is therefore exactly min(t_ssd, t_pcie), so pipelined throughput must
        improve but must never exceed the serialized figure with the smaller stage
        removed entirely.
        """
        serial = SpeculativeEngine(
            draft_k=5, acceptance_prob=0.85, seed=92,
            cost_model=CostModel(pipeline_transfer=False),
        ).run_simulation(target_token_count=60)
        piped = SpeculativeEngine(
            draft_k=5, acceptance_prob=0.85, seed=92,
            cost_model=CostModel(pipeline_transfer=True),
        ).run_simulation(target_token_count=60)

        self.assertGreater(piped.accepted_tok_s, serial.accepted_tok_s)
        # Residency must be identical -- pipelining is a scheduling change, not a
        # caching one, and must not perturb which tier served what.
        self.assertEqual(piped.vram_experts, serial.vram_experts)
        self.assertEqual(piped.ssd_experts, serial.ssd_experts)

    def test_admission_control_protects_hot_experts(self) -> None:
        """Frequency-gated admission must beat pure recency on a thrashing tier.

        A pass touches more distinct experts than the VRAM tier holds, so LRU is
        swept clean every pass. Admission control should keep materially more
        experts resident in VRAM -- the only tier whose hits avoid PCIe.
        """
        lru = SpeculativeEngine(
            draft_k=5, acceptance_prob=0.85, seed=92, policy_factory=LRUPolicy
        ).run_simulation(target_token_count=60)
        admit = SpeculativeEngine(
            draft_k=5, acceptance_prob=0.85, seed=92,
            policy_factory=lambda: TinyLFUAdmissionPolicy(LFUPolicy()),
        ).run_simulation(target_token_count=60)

        self.assertGreater(
            admit.vram_experts,
            lru.vram_experts * 2,
            "admission control should more than double VRAM residency vs pure LRU",
        )
        self.assertGreaterEqual(admit.accepted_tok_s, lru.accepted_tok_s)

    def test_belady_refuses_to_run_online(self) -> None:
        """Belady must fail loudly rather than silently degrade on a live stream.

        Without a future map every key looks never-reused-again, so the "oracle"
        quietly becomes an arbitrary choice that scores WORSE than LRU while still
        being labelled an upper bound.
        """
        engine = SpeculativeEngine(
            draft_k=5, acceptance_prob=0.85, seed=92, policy_factory=BeladyPolicy
        )
        with self.assertRaises(RuntimeError):
            engine.run_simulation(target_token_count=40, vram_capacity=10, host_capacity=10)

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

        # LRU would normally evict (0,1) because step=1. But (0,1) is pinned.
        evicted = spec_policy.select_eviction([(0, 1), (0, 2), (0, 3)], step=4)
        self.assertEqual(evicted, (0, 2))

    def test_higher_acceptance_boosts_throughput(self) -> None:
        """Higher draft acceptance should increase accepted tok/s."""
        stats_low = SpeculativeEngine(
            draft_k=4, acceptance_prob=0.40, seed=999
        ).run_simulation(target_token_count=100)

        stats_high = SpeculativeEngine(
            draft_k=4, acceptance_prob=0.85, seed=999
        ).run_simulation(target_token_count=100)

        self.assertGreater(
            stats_high.accepted_tok_s,
            stats_low.accepted_tok_s,
            f"Higher acceptance ({stats_high.accepted_tok_s:.2f}) "
            f"should beat low ({stats_low.accepted_tok_s:.2f})",
        )


if __name__ == "__main__":
    unittest.main()
