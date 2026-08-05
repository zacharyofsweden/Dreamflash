"""Tests for the Dreamflash roofline model.

These are consistency tests, not physics. They catch transcription errors in
the model geometry and algebra errors in the bandwidth math -- the two failure
modes that would silently invalidate every downstream engineering decision.

Run:  python -m pytest tests/ -q     (or: python tests/test_roofline.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "roofline"))

from model import BPW, FLASH, PRO  # noqa: E402
from roofline import (  # noqa: E402
    GB,
    GIB,
    Machine,
    attainable_tok_s,
    compute,
    plan_budget,
)


def approx(a, b, tol=0.02):
    return abs(a - b) <= tol * abs(b)


def test_bpw_matches_known_gguf_values():
    """Standard GGUF bits-per-weight, derived from block layout."""
    assert approx(BPW["IQ2_XXS"], 2.0625, 1e-9)
    assert approx(BPW["Q2_K"], 2.625, 1e-9)
    assert approx(BPW["Q4_K"], 4.5, 1e-9)
    assert approx(BPW["Q8_0"], 8.5, 1e-9)


def test_routed_param_count_reconciles_with_model_card():
    """284 B total published. Our derived routed count must leave a plausible
    non-routed remainder -- if this drifts, a shape constant is wrong."""
    routed = FLASH.total_routed_params
    assert approx(routed, 277e9, 0.02), f"routed={routed/1e9:.1f}B"
    nonrouted = FLASH.nonrouted_params
    # Non-routed = attention + shared experts + embed + output + norms.
    # Must be positive and a small fraction of the whole.
    assert 0 < nonrouted < 0.05 * FLASH.total_params
    assert approx(nonrouted, 7e9, 0.10)


def test_pro_reconciles_too():
    """Same check for Pro guards against a Flash-specific fluke."""
    routed = PRO.total_routed_params
    assert routed < PRO.total_params
    assert 0 < PRO.nonrouted_params < 0.10 * PRO.total_params


def test_expert_size():
    """6.75 MiB is the atomic unit of the entire streaming design."""
    b = FLASH.bytes_per_routed_expert()
    assert approx(b / 1024 / 1024, 6.75, 1e-6)


def test_expert_loads_per_token():
    assert FLASH.expert_loads_per_token() == 43 * 6 == 258


def test_cold_bytes_per_token():
    assert approx(FLASH.routed_bytes_per_token() / 1e9, 1.826, 0.01)


def test_higher_precision_costs_proportionally_more():
    q2 = FLASH.bytes_per_routed_expert("IQ2_XXS", "Q2_K")
    q4 = FLASH.bytes_per_routed_expert("Q4_K", "Q4_K")
    assert q4 > q2
    # Q4_K everywhere is 4.5bpw vs (2*2.0625+2.625)/3 = 2.25bpw -> exactly 2x
    assert approx(q4 / q2, 2.0, 1e-6)


def test_attainable_is_monotonic_in_hit_rate():
    m = Machine()
    prev = 0.0
    for h in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
        t = attainable_tok_s(FLASH, m, h)
        assert t > prev
        prev = t


def test_zero_hit_rate_ceiling():
    """With no cache the SSD alone permits ~3.3 tok/s. This is the number that
    makes milestones M4 (1 tok/s) and M5 (3 tok/s) reachable without any cache
    at all -- and M6+ impossible without one."""
    m = Machine()
    t = attainable_tok_s(FLASH, m, 0.0)
    assert approx(t, 3.29, 0.02)


def test_required_hit_rate_for_target():
    m = Machine()
    b = plan_budget(FLASH, m, 100_000, 20_000.0)
    rl = compute(FLASH, m, b, target_tok_s=10.0)
    assert approx(rl.required_hit_rate_ssd, 0.671, 0.02)


def test_pcie_is_not_the_binding_constraint():
    """Central architectural finding: at 24 GB/s, PCIe 4.0 x16 can carry the
    full 18.3 GB/s cold-cache demand. So the design must optimize for SSD
    reads, NOT for PCIe transfers -- and a host-RAM cache tier is valuable
    even though every RAM hit still crosses PCIe."""
    m = Machine()
    b = plan_budget(FLASH, m, 100_000, 20_000.0)
    rl = compute(FLASH, m, b, target_tok_s=10.0)
    assert rl.required_hit_rate_vram < 0, "PCIe should have slack at 10 tok/s"
    assert rl.required_hit_rate_ssd > rl.required_hit_rate_vram


def test_faster_ssd_relaxes_required_hit_rate():
    """Sanity: the recommended hardware upgrade must actually help."""
    slow = Machine(ssd_read_bps=6.0 * GB)
    fast = Machine(ssd_read_bps=12.0 * GB)
    b_s = plan_budget(FLASH, slow, 100_000, 20_000.0)
    b_f = plan_budget(FLASH, fast, 100_000, 20_000.0)
    assert (compute(FLASH, fast, b_f, 10.0).required_hit_rate_ssd
            < compute(FLASH, slow, b_s, 10.0).required_hit_rate_ssd)


def test_more_vram_increases_cache_capacity():
    small = Machine(vram_bytes=12 * GIB)
    big = Machine(vram_bytes=24 * GIB)
    b_s = plan_budget(FLASH, small, 100_000, 20_000.0)
    b_b = plan_budget(FLASH, big, 100_000, 20_000.0)
    assert b_b.vram_expert_cache > b_s.vram_expert_cache
    assert (compute(FLASH, big, b_b, 10.0).cache_capacity_experts
            > compute(FLASH, small, b_s, 10.0).cache_capacity_experts)


def test_long_context_eats_expert_cache():
    """100K context is not free: KV state competes with the expert cache for
    the same VRAM. This is the tension behind milestone M9."""
    m = Machine()
    short = plan_budget(FLASH, m, 4_096, 20_000.0)
    long_ = plan_budget(FLASH, m, 100_000, 20_000.0)
    assert long_.vram_expert_cache < short.vram_expert_cache


def test_nonrouted_weights_must_fit_in_vram():
    """If non-routed weights alone exceeded VRAM the whole design collapses --
    they are touched every layer every token and cannot be streamed."""
    m = Machine()
    b = plan_budget(FLASH, m, 100_000, 20_000.0)
    assert b.nonrouted_vram < m.vram_bytes
    assert b.vram_expert_cache > 0, "no VRAM left for experts"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
