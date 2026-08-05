"""Discrete GPU runtime configuration generator for DwarfStar (ds4).

Calculates exact memory allocations for discrete GPU targets and generates C preprocessor
macros or CLI flags to prevent OOM errors on non-APU hardware.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Insert path for roofline model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "roofline"))

from model import FLASH, Shape
from roofline import Machine, plan_budget


def generate_discrete_config(
    vram_gb: float = 12.0,
    ram_gb: float = 16.0,
    ctx_tokens: int = 100_000,
    kv_bytes_per_token: float = 20_000.0,
    shape: Shape = FLASH,
) -> Dict[str, Any]:
    """Calculate exact discrete memory budgets and constants."""
    gib = 1024.0**3
    machine = Machine(
        vram_bytes=vram_gb * gib,
        ram_bytes=ram_gb * gib,
    )
    budget = plan_budget(shape, machine, ctx_tokens, kv_bytes_per_token)
    per_expert = shape.bytes_per_routed_expert()

    vram_expert_count = max(0, int(budget.vram_expert_cache // per_expert))
    host_expert_count = max(0, int(budget.host_expert_cache // per_expert))

    return {
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "ctx_tokens": ctx_tokens,
        "nonrouted_vram_bytes": budget.nonrouted_vram,
        "kv_vram_bytes": budget.kv_state_vram,
        "graph_scratch_vram_bytes": budget.graph_scratch_vram,
        "vram_expert_cache_bytes": max(0.0, budget.vram_expert_cache),
        "vram_expert_count": vram_expert_count,
        "host_expert_cache_bytes": max(0.0, budget.host_expert_cache),
        "host_expert_count": host_expert_count,
        "total_cache_experts": vram_expert_count + host_expert_count,
        "total_model_experts": shape.total_routed_experts,
        "capacity_fraction": (vram_expert_count + host_expert_count) / shape.total_routed_experts,
    }


def export_c_header(config: Dict[str, Any], output_path: Path | str) -> None:
    """Export C header file with discrete memory macro overrides."""
    lines = [
        "/* Auto-generated discrete GPU configuration header for Dreamflash */",
        "#ifndef DS4_CONFIG_AUTO_H",
        "#define DS4_CONFIG_AUTO_H",
        "",
        f"#define DS4_CONFIG_VRAM_BYTES ({int(config['vram_gb'] * 1024**3)}ULL)",
        f"#define DS4_CONFIG_RAM_BYTES ({int(config['ram_gb'] * 1024**3)}ULL)",
        f"#define DS4_CONFIG_CTX_TOKENS ({config['ctx_tokens']})",
        f"#define DS4_CONFIG_VRAM_EXPERT_CAPACITY ({config['vram_expert_count']})",
        f"#define DS4_CONFIG_HOST_EXPERT_CAPACITY ({config['host_expert_count']})",
        f"#define DS4_CONFIG_TOTAL_CACHE_CAPACITY ({config['total_cache_experts']})",
        "",
        "#endif /* DS4_CONFIG_AUTO_H */",
    ]
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vram-gb", type=float, default=12.0)
    parser.add_argument("--ram-gb", type=float, default=16.0)
    parser.add_argument("--ctx", type=int, default=100_000)
    parser.add_argument("--header-out", type=str, default="include/ds4_config_auto.h")
    args = parser.parse_args()

    cfg = generate_discrete_config(
        vram_gb=args.vram_gb,
        ram_gb=args.ram_gb,
        ctx_tokens=args.ctx,
    )

    print(f"Discrete GPU Memory Configuration:")
    print(f"  VRAM Expert Cache : {cfg['vram_expert_cache_bytes'] / 1024**3:.2f} GiB ({cfg['vram_expert_count']} experts)")
    print(f"  Host Expert Cache : {cfg['host_expert_cache_bytes'] / 1024**3:.2f} GiB ({cfg['host_expert_count']} experts)")
    print(f"  Total Capacity    : {cfg['total_cache_experts']} / {cfg['total_model_experts']} experts ({cfg['capacity_fraction']*100:.1f}%)")

    export_c_header(cfg, args.header_out)
    print(f"Exported auto header to {args.header_out}")


if __name__ == "__main__":
    main()
