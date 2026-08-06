"""Speculative Decoding CLI Evaluation Tool.

Evaluates speculative candidate verification across draft window lengths (K) and
acceptance probabilities (alpha), reporting accepted tok/s.

Cache hit rate is NOT an input here -- it emerges from the configured VRAM/host
capacities and the access pattern. Throughput comes from cost_model.CostModel,
which charges VRAM, PCIe, SSD, kernel launch, and the draft model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "roofline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trace_replay"))

from cost_model import CostModel
from draft_simulator import SpeculativeEngine
from model import FLASH
from roofline import Machine, plan_budget


def evaluate_speculative_matrix(
    ssd_gbps: float = 6.0,
    target_tokens: int = 200,
    vram_capacity: int = 339,
    host_capacity: int = 1744,
) -> str:
    lines = []
    a = lines.append

    cost = CostModel(ssd_read_bps=ssd_gbps * 1e9)
    baseline = cost.cold_baseline_tok_s(FLASH.routed_bytes_per_token(), FLASH.n_layer)

    a("=" * 96)
    a("DREAMFLASH SPECULATIVE DECODING EVALUATION (MoE Batch Verification)")
    a(f"Target Model: {FLASH.name} | SSD: {ssd_gbps:.1f} GB/s | PCIe: {cost.pcie_bps/1e9:.1f} GB/s"
      f" | VRAM: {cost.vram_bps/1e9:.0f} GB/s")
    a(f"Cache: {vram_capacity:,} VRAM + {host_capacity:,} host experts "
      f"({(vram_capacity + host_capacity) / FLASH.total_routed_experts * 100:.1f}% of model)")
    a(f"Non-speculative cold baseline under the same cost model: {baseline:.2f} tok/s")
    a("ASSUMED -- NOT MEASURED. Every bandwidth above is a placeholder; the expert")
    a("access stream is synthetic. See the caveats below the table.")
    a("=" * 96)
    a(f"{'K':<4} | {'Accept':<8} | {'Hit%':<7} | {'tok/s (serial)':<15} | "
      f"{'tok/s (ideal)':<14} | {'vs base':<8} | {'30?'}")
    a("-" * 96)

    draft_ks = [1, 2, 3, 4, 5, 6]
    accept_probs = [0.50, 0.65, 0.75, 0.85]
    rows = []

    for k in draft_ks:
        for alpha in accept_probs:
            engine = SpeculativeEngine(
                draft_k=k,
                acceptance_prob=alpha,
                shape=FLASH,
                ssd_gbps=ssd_gbps,
                seed=42 + k * 10,
            )
            stats = engine.run_simulation(
                target_token_count=target_tokens,
                vram_capacity=vram_capacity,
                host_capacity=host_capacity,
            )

            tok_s = stats.accepted_tok_s
            met = "  YES" if tok_s >= 30.0 else "  No"
            rows.append((k, alpha, tok_s, stats.accepted_tok_s_ideal, stats.cache_hit_rate))

            a(f"{k:<4} | {alpha*100:6.1f}% | {stats.cache_hit_rate*100:6.2f}% | "
              f"{tok_s:13.2f} | {stats.accepted_tok_s_ideal:12.2f} | "
              f"{stats.speedup_factor:7.2f}x | {met}")

    a("-" * 96)
    a("Summary:")

    best_k, best_alpha, best_tok_s, best_ideal, best_hit = max(rows, key=lambda r: r[2])
    met_rows = [r for r in rows if r[2] >= 30.0]
    if met_rows:
        a(f"  {len(met_rows)} of {len(rows)} (K, alpha) combinations reach 30 tok/s "
          f"at the serialized end.")
    else:
        a(f"  NO (K, alpha) combination reaches 30 tok/s at the serialized end. "
          f"Best: {best_tok_s:.2f} tok/s.")
    a(f"  Best in matrix: K={best_k}, acceptance={best_alpha*100:.0f}% -> "
      f"{best_tok_s:.2f} tok/s serialized, {best_ideal:.2f} tok/s with ideal overlap,")
    a(f"  at an emergent {best_hit*100:.1f}% cache hit rate.")
    a("")
    a("  READ BEFORE QUOTING ANY NUMBER ABOVE:")
    a("  * 'serial' charges transfer and compute back to back; 'ideal' assumes transfer")
    a("    hides perfectly behind compute. Real hardware is between. Quote the serial")
    a("    end -- overlap efficiency here is unmeasured.")
    a("  * Hit% is an OUTPUT (a real LRU cache over the access stream), not an input.")
    a("    It is still only as meaningful as the SYNTHETIC access stream that produced")
    a("    it: cross-candidate expert overlap is injected by a hardcoded 0.80 constant,")
    a("    and no real DeepSeek-V4 routing trace exists in this repo.")
    a("  * 'vs base' compares against non-speculative cold decode under this SAME cost")
    a("    model, so it isolates speculation rather than crediting it with the cache.")
    a("  * Acceptance probability remains an input. It is a property of the draft model")
    a("    on real prompts and nothing here measures it.")
    a("=" * 96)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssd-gbps", type=float, default=6.0)
    parser.add_argument("--target-tokens", type=int, default=200)
    parser.add_argument("--vram-gb", type=float, default=12.0)
    parser.add_argument("--ram-gb", type=float, default=16.0)
    parser.add_argument("--ctx", type=int, default=100_000)
    parser.add_argument(
        "--vram-experts", type=int, default=None, help="Override VRAM cache capacity"
    )
    parser.add_argument(
        "--host-experts", type=int, default=None, help="Override host cache capacity"
    )
    args = parser.parse_args()

    # Derive capacities from the byte budget, exactly as replay.py does.
    m = Machine(vram_bytes=args.vram_gb * (1024**3), ram_bytes=args.ram_gb * (1024**3))
    budget = plan_budget(FLASH, m, args.ctx, 20_000.0)
    per_expert = FLASH.bytes_per_routed_expert()

    vram_cap = (
        args.vram_experts
        if args.vram_experts is not None
        else max(0, int(budget.vram_expert_cache // per_expert))
    )
    host_cap = (
        args.host_experts
        if args.host_experts is not None
        else max(0, int(budget.host_expert_cache // per_expert))
    )

    print(
        evaluate_speculative_matrix(
            ssd_gbps=args.ssd_gbps,
            target_tokens=args.target_tokens,
            vram_capacity=vram_cap,
            host_capacity=host_cap,
        )
    )


if __name__ == "__main__":
    main()
