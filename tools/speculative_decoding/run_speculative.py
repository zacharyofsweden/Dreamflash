"""Speculative Decoding CLI Evaluation Tool.

Evaluates speculative candidate verification performance across draft window lengths (K)
and acceptance probabilities (alpha), reporting accepted tok/s and speedup factors.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "roofline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trace_replay"))

from draft_simulator import SpeculativeEngine
from model import FLASH


def evaluate_speculative_matrix(
    ssd_gbps: float = 6.0,
    target_tokens: int = 200,
    cache_hit_rate: float = 0.70,
) -> str:
    lines = []
    a = lines.append

    a("=" * 85)
    a("DREAMFLASH SPECULATIVE DECODING EVALUATION (MoE Batch Verification)")
    a(f"Target Model: {FLASH.name} | SSD Bandwidth: {ssd_gbps:.1f} GB/s | Cache Hit Rate: {cache_hit_rate*100:.1f}%")
    a("=" * 85)
    a("ASSUMED -- NOT MEASURED. Cache hit rate and acceptance probability are both inputs.")
    a(f"{'Draft Length (K)':<18} | {'Accept Prob':<12} | {'Accepted tok/s':<16} | {'vs cold':<10} | {'30 tok/s Met?'}")
    a("-" * 85)

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
                cache_hit_rate=cache_hit_rate,
            )

            tok_s = stats.effective_accepted_tok_s
            speedup = stats.speedup_factor
            met = "  YES (TARGET MET)" if tok_s >= 30.0 else "  No"
            rows.append((k, alpha, tok_s))

            a(f"{k:<18} | {alpha*100:10.1f}% | {tok_s:14.2f} tok/s | {speedup:8.2f}x | {met}")

    a("-" * 85)
    a("Summary:")
    a("  Speculative batch verification deduplicates expert reads across draft candidate trees.")

    best_k, best_alpha, best_tok_s = max(rows, key=lambda r: r[2])
    met = [r for r in rows if r[2] >= 30.0]
    if met:
        lo_alpha = min(r[1] for r in met)
        lo_k = min(r[0] for r in met)
        a(f"  {len(met)} of {len(rows)} (K, alpha) combinations reach 30 tok/s at a "
          f"{cache_hit_rate*100:.0f}% assumed hit rate,")
        a(f"  the weakest requiring K >= {lo_k} and acceptance >= {lo_alpha*100:.0f}%.")
    else:
        a(f"  NO (K, alpha) combination reaches 30 tok/s at a {cache_hit_rate*100:.0f}% "
          f"assumed hit rate.")
    a(f"  Best in matrix: K={best_k}, acceptance={best_alpha*100:.0f}% -> {best_tok_s:.2f} tok/s.")
    a("  The 'vs cold' column baselines against the 3.29 tok/s cold-cache ceiling (no cache")
    a("  at all), so most of that multiple reflects the assumed cache, not speculation.")
    a("")
    a("  MODEL SCOPE -- READ BEFORE QUOTING ANY NUMBER ABOVE:")
    a("  Wall-clock time here is modelled as SSD read time for cache-missed experts and")
    a("  NOTHING ELSE. Compute, PCIe transfer, attention/KV, kernel launch, the draft model")
    a("  itself, and verification are all charged at zero cost. Throughput therefore reduces")
    a("  to roughly accepted_tokens / (1 - cache_hit_rate), and cache_hit_rate is a free")
    a("  input parameter (--cache-hit-rate), not a measured quantity. These figures are an")
    a("  UPPER BOUND on an I/O-bound-only machine, not a predicted decode speed.")
    a("=" * 85)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssd-gbps", type=float, default=6.0)
    parser.add_argument("--target-tokens", type=int, default=200)
    parser.add_argument("--cache-hit-rate", type=float, default=0.70)
    args = parser.parse_args()

    report = evaluate_speculative_matrix(
        ssd_gbps=args.ssd_gbps,
        target_tokens=args.target_tokens,
        cache_hit_rate=args.cache_hit_rate,
    )
    print(report)


if __name__ == "__main__":
    main()
