"""Trace Replay CLI tool.

Compares expert replacement policies (LRU, LFU, LRU-K, Hybrid, Cost-Aware, Belady Oracle)
on a given or synthetic trace, reporting hit rates, SSD & PCIe traffic, and eviction counts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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
) -> Dict[str, dict]:
    policies: List[ReplacementPolicy] = [
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
    args = parser.parse_args()

    m = Machine(
        vram_bytes=args.vram_gb * (1024**3),
        ram_bytes=args.ram_gb * (1024**3),
    )
    budget = plan_budget(FLASH, m, args.ctx, 20_000.0)

    per_expert = FLASH.bytes_per_routed_expert()
    vram_cap = max(0, int(budget.vram_expert_cache // per_expert))
    host_cap = max(0, int(budget.host_expert_cache // per_expert))

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
