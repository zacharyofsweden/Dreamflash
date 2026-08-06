"""Trace Replay CLI tool.

Compares expert replacement policies (LRU, LFU, LRU-K, Hybrid, Cost-Aware, Belady Oracle)
on a given or synthetic trace, reporting hit rates, SSD & PCIe traffic, and eviction counts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure parent path is in sys.path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "roofline"))

from model import FLASH
from policy import (
    BeladyPolicy,
    CostAwarePolicy,
    HybridRecencyFrequencyPolicy,
    LFUPolicy,
    LRUKPolicy,
    LRUPolicy,
    ReplacementPolicy,
)
from roofline import Machine, plan_budget
from simulator import CacheSimulator
from trace import Trace, generate_synthetic_trace


def run_comparison(
    trace: Trace,
    vram_capacity: int,
    host_capacity: int,
    policies: Optional[List[ReplacementPolicy]] = None,
) -> Dict[str, dict]:
    if policies is None:
        policies = [
            LRUPolicy(),
            LFUPolicy(),
            LRUKPolicy(k=2),
            HybridRecencyFrequencyPolicy(),
            CostAwarePolicy(total_layers=FLASH.n_layer),
            BeladyPolicy(),
        ]

    results = {}
    for policy in policies:
        sim = CacheSimulator(
            vram_capacity=vram_capacity,
            host_capacity=host_capacity,
            shape=FLASH,
            policy=policy,
        )
        stats = sim.run(trace)
        results[policy.name()] = {
            "vram_hits": stats.vram_hits,
            "host_hits": stats.host_hits,
            "ssd_misses": stats.ssd_misses,
            "vram_hit_rate": stats.vram_hit_rate,
            "combined_hit_rate": stats.combined_hit_rate,
            "ssd_gb": stats.ssd_bytes_transferred / 1e9,
            "pcie_gb": stats.pcie_bytes_transferred / 1e9,
            "vram_evictions": stats.vram_evictions,
            "host_evictions": stats.host_evictions,
        }
    return results


def print_report(
    results: Dict[str, dict],
    vram_cap: int,
    host_cap: int,
    total_experts: int,
    total_accesses: int,
    trace_source: str = "unknown",
) -> str:
    lines = []
    a = lines.append

    a("=" * 80)
    a("OFFLINE TRACE REPLAY POLICY COMPARISON")
    a(f"Trace source: {trace_source}")
    a(f"Total experts in model: {total_experts:,} | Total trace accesses: {total_accesses:,}")
    a(f"VRAM Capacity: {vram_cap:,} experts | Host RAM Capacity: {host_cap:,} experts")
    a(f"Combined Capacity: {vram_cap + host_cap:,} / {total_experts:,} experts "
      f"({(vram_cap + host_cap) / total_experts * 100:.1f}%)")
    a("=" * 80)
    a(f"{'Policy':<25} | {'VRAM Hit%':<10} | {'Combined Hit%':<15} | {'SSD (GB)':<10} | {'PCIe (GB)':<10}")
    a("-" * 80)

    lru_hit_rate = results.get("LRU", {}).get("combined_hit_rate", 0.0)
    belady_hit_rate = results.get("Belady-Oracle", {}).get("combined_hit_rate", 0.0)
    gap = (belady_hit_rate - lru_hit_rate) * 100.0

    for name, r in results.items():
        vram_pct = r["vram_hit_rate"] * 100.0
        comb_pct = r["combined_hit_rate"] * 100.0
        a(f"{name:<25} | {vram_pct:9.2f}% | {comb_pct:14.2f}% | {r['ssd_gb']:9.2f} | {r['pcie_gb']:9.2f}")

    a("-" * 80)
    a(f"Belady (see caveat): {belady_hit_rate * 100:.2f}% | LRU baseline: {lru_hit_rate * 100:.2f}%")
    a(f"Headroom (Belady - LRU): {gap:+.2f}% percentage points")
    if host_cap > 0:
        a("CAVEAT: Belady is optimal only for a single-tier cache. This simulation is two-tier")
        a("and applies Belady independently within VRAM and within host RAM, so it is a strong")
        a("heuristic here, NOT a proven upper bound -- another policy can beat it. LRU is a")
        a("conventional baseline, not a lower bound (CostAware scores below it).")
    a("=" * 80)

    return "\n".join(lines)


def sweep_zipf_report(
    tokens: int,
    vram_cap: int,
    host_cap: int,
    exponents: List[float],
) -> str:
    """Report hit rate across routing-skew exponents.

    The single most misleading thing this tool can do is print one hit rate from one
    arbitrary `zipf_s` and let it be read as a property of DeepSeek-V4-Flash. Nothing
    in this repo justifies any particular exponent, and the answer swings from
    "hopeless" to "trivial" across the plausible range. Until a real routing trace
    exists (Phase 0), the range IS the finding.
    """
    lines = []
    a = lines.append

    a("=" * 80)
    a("ROUTING-SKEW SENSITIVITY SWEEP (synthetic)")
    a(f"VRAM {vram_cap:,} + host {host_cap:,} experts | {tokens} tokens per point")
    a("=" * 80)
    a(f"{'zipf_s':<10} | {'LRU hit%':<12} | {'Belady hit%':<14} | {'10 tok/s viable?'}")
    a("-" * 80)

    # From roofline: 10 tok/s needs a 67.1% hit rate at 6 GB/s.
    required = 0.671
    rows = []

    for s in exponents:
        trace = generate_synthetic_trace(
            n_tokens=tokens,
            n_layers=FLASH.n_layer,
            n_experts=FLASH.n_expert,
            n_expert_used=FLASH.n_expert_used,
            distribution="zipf",
            zipf_s=s,
        )
        # Only the bracketing policies are needed per point; running all six here
        # multiplies the sweep cost for numbers the sweep does not report.
        results = run_comparison(
            trace, vram_cap, host_cap, policies=[LRUPolicy(), BeladyPolicy()]
        )
        lru = results.get("LRU", {}).get("combined_hit_rate", 0.0)
        belady = results.get("Belady-Oracle", {}).get("combined_hit_rate", 0.0)
        verdict = "YES" if lru >= required else ("oracle only" if belady >= required else "no")
        rows.append((s, lru, belady))
        a(f"{s:<10.2f} | {lru*100:11.2f}% | {belady*100:13.2f}% | {verdict}")

    a("-" * 80)
    lo = min(r[1] for r in rows)
    hi = max(r[1] for r in rows)
    a(f"LRU hit rate spans {lo*100:.1f}% to {hi*100:.1f}% across this range "
      f"({(hi-lo)*100:.1f} points).")
    a(f"10 tok/s at 6 GB/s requires {required*100:.1f}%.")

    default_row = next((r for r in rows if abs(r[0] - 1.2) < 1e-9), None)
    if default_row is not None and abs(default_row[1] - required) < 0.05:
        a("")
        a(f"NOTE: the default exponent (1.2) yields {default_row[1]*100:.2f}%, within a")
        a(f"whisker of the {required*100:.1f}% the target requires -- the one value in this")
        a("range that makes the north star look barely achievable. Treat a default that")
        a("lands exactly on the threshold as a number to re-derive, not to rely on.")
    a("")
    a("The exponent is an ASSUMPTION with no citation anywhere in this repo, and the")
    a("verdict on the north-star target flips inside the plausible range. Treat the")
    a("span above as the honest state of knowledge until a measured routing trace")
    a("exists -- that measurement, not more policy work, is the critical path.")
    a("=" * 80)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=200, help="Number of synthetic tokens")
    parser.add_argument("--vram-gb", type=float, default=12.0)
    parser.add_argument("--ram-gb", type=float, default=16.0)
    parser.add_argument("--ctx", type=int, default=100_000)
    parser.add_argument("--distribution", choices=["zipf", "uniform", "repeating"], default="zipf")
    parser.add_argument("--zipf-s", type=float, default=1.2)
    parser.add_argument("--trace-file", type=str, default=None, help="Path to JSONL trace file")
    parser.add_argument("--output-json", type=str, default=None, help="Save summary to JSON file")
    parser.add_argument(
        "--sweep-zipf",
        action="store_true",
        help="Sweep the routing-skew exponent and report the RANGE of hit rates rather "
        "than a single point estimate from an uncited constant",
    )
    args = parser.parse_args()

    m = Machine(
        vram_bytes=args.vram_gb * (1024**3),
        ram_bytes=args.ram_gb * (1024**3),
    )
    budget = plan_budget(FLASH, m, args.ctx, 20_000.0)

    per_expert = FLASH.bytes_per_routed_expert()
    vram_cap = max(0, int(budget.vram_expert_cache // per_expert))
    host_cap = max(0, int(budget.host_expert_cache // per_expert))

    if args.sweep_zipf:
        print(sweep_zipf_report(
            tokens=args.tokens,
            vram_cap=vram_cap,
            host_cap=host_cap,
            exponents=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        ))
        return

    if args.trace_file:
        trace = Trace.load_jsonl(args.trace_file)
        trace_source = f"REAL trace file: {args.trace_file}"
    else:
        trace_source = (
            f"SYNTHETIC -- generated, not measured "
            f"(distribution={args.distribution}"
            + (f", zipf_s={args.zipf_s}" if args.distribution == "zipf" else "")
            + "). Hit rates below are a property of this generator, "
            "NOT of DeepSeek-V4-Flash."
        )
        trace = generate_synthetic_trace(
            n_tokens=args.tokens,
            n_layers=FLASH.n_layer,
            n_experts=FLASH.n_expert,
            n_expert_used=FLASH.n_expert_used,
            distribution=args.distribution,
            zipf_s=args.zipf_s,
        )

    results = run_comparison(trace, vram_cap, host_cap)
    report_text = print_report(
        results,
        vram_cap,
        host_cap,
        FLASH.total_routed_experts,
        trace.total_accesses(),
        trace_source=trace_source,
    )
    print(report_text)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trace_source": trace_source,
                    "vram_capacity": vram_cap,
                    "host_capacity": host_cap,
                    "total_accesses": trace.total_accesses(),
                    "results": results,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
