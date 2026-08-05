"""Dreamflash decode roofline.

Answers one question: given a machine and a target decode rate, what fraction of
routed-expert reads MUST be served from cache for the SSD not to be the wall?

This is deliberately a closed-form model, not a simulation. It establishes an
upper bound on achievable tok/s. A real implementation can only ever be slower
than what this predicts -- so if the roofline says 10 tok/s is unreachable, no
amount of kernel tuning will rescue it, and the honest move is to say so.

Usage:
    python roofline.py                 # target machine, default assumptions
    python roofline.py --help
    python roofline.py --ssd-gbps 12 --vram-gb 24   # explore upgrades
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from model import FLASH, Shape

GB = 1_000_000_000.0     # decimal, matches how drive vendors quote bandwidth
GIB = 1024.0 ** 3


@dataclass
class Machine:
    """Measured (or assumed) hardware capability.

    Defaults describe the Dreamflash target box. Every default is a PLACEHOLDER
    until tools/hardware_probe writes real numbers into results/hardware/ --
    they are labelled as such in the output so nobody mistakes them for
    measurements.
    """

    name: str = "target (assumed)"
    vram_bytes: float = 12 * GIB
    ram_bytes: float = 16 * GIB
    ssd_read_bps: float = 6.0 * GB       # large-block random read, high QD
    pcie_h2d_bps: float = 24.0 * GB      # PCIe 4.0 x16 practical, pinned
    measured: bool = False


@dataclass
class Budget:
    """How VRAM and host RAM get divided up. All values in bytes."""

    nonrouted_vram: float
    kv_state_vram: float
    graph_scratch_vram: float
    vram_expert_cache: float

    os_reserve_ram: float
    pinned_staging_ram: float
    host_expert_cache: float


def plan_budget(
    shape: Shape,
    machine: Machine,
    ctx_tokens: int,
    kv_bytes_per_token: float,
    nonrouted_quant: str = "Q8_0",
    graph_scratch_bytes: float = 1.0 * GIB,
    os_reserve_bytes: float = 3.0 * GIB,
    pinned_staging_bytes: float = 1.5 * GIB,
) -> Budget:
    """Lay out the three tiers, VRAM first, and report what is left for cache.

    Priority order follows the design doc: mandatory runtime state, then
    attention/KV, then non-routed weights, then whatever remains is expert
    cache. Non-routed weights are pinned to VRAM because they are touched every
    token by every layer -- streaming them would be strictly worse than
    streaming experts.
    """
    from model import BPW

    nonrouted = shape.nonrouted_params * BPW[nonrouted_quant] / 8.0
    kv = ctx_tokens * kv_bytes_per_token

    vram_used = nonrouted + kv + graph_scratch_bytes
    vram_cache = machine.vram_bytes - vram_used

    ram_used = os_reserve_bytes + pinned_staging_bytes
    host_cache = machine.ram_bytes - ram_used

    return Budget(
        nonrouted_vram=nonrouted,
        kv_state_vram=kv,
        graph_scratch_vram=graph_scratch_bytes,
        vram_expert_cache=vram_cache,
        os_reserve_ram=os_reserve_bytes,
        pinned_staging_ram=pinned_staging_bytes,
        host_expert_cache=host_cache,
    )


@dataclass
class Roofline:
    target_tok_s: float
    bytes_per_token_cold: float
    required_ssd_bps_cold: float

    required_hit_rate_ssd: float   # combined VRAM+RAM hit rate SSD demands
    required_hit_rate_vram: float  # VRAM-only hit rate PCIe demands

    cache_capacity_experts: int
    total_experts: int
    capacity_fraction: float

    max_tok_s_at_zero_hit: float
    feasible_without_cache: bool


def compute(
    shape: Shape,
    machine: Machine,
    budget: Budget,
    target_tok_s: float = 10.0,
) -> Roofline:
    per_expert = shape.bytes_per_routed_expert()
    cold_bpt = shape.routed_bytes_per_token()

    required_ssd = cold_bpt * target_tok_s

    # SSD constraint: (1-h) * bytes_per_token * tok_s <= ssd_bandwidth
    h_ssd = 1.0 - machine.ssd_read_bps / required_ssd

    # PCIe constraint: every VRAM miss crosses PCIe, whether it came from host
    # RAM or from SSD. So PCIe traffic is governed by the VRAM hit rate alone.
    h_vram = 1.0 - machine.pcie_h2d_bps / required_ssd

    cache_bytes = max(0.0, budget.vram_expert_cache) + max(
        0.0, budget.host_expert_cache
    )
    cap_experts = int(cache_bytes // per_expert)

    return Roofline(
        target_tok_s=target_tok_s,
        bytes_per_token_cold=cold_bpt,
        required_ssd_bps_cold=required_ssd,
        required_hit_rate_ssd=h_ssd,
        required_hit_rate_vram=h_vram,
        cache_capacity_experts=cap_experts,
        total_experts=shape.total_routed_experts,
        capacity_fraction=cap_experts / shape.total_routed_experts,
        max_tok_s_at_zero_hit=machine.ssd_read_bps / cold_bpt,
        feasible_without_cache=machine.ssd_read_bps >= required_ssd,
    )


def attainable_tok_s(
    shape: Shape, machine: Machine, hit_rate: float
) -> float:
    """Decode rate the SSD permits at a given combined cache hit rate.

    Pure bandwidth bound -- ignores compute, latency, and sync, all of which
    only make it worse. This is a ceiling, never a prediction.
    """
    cold = shape.routed_bytes_per_token()
    miss_bpt = cold * (1.0 - hit_rate)
    if miss_bpt <= 0:
        return float("inf")
    return machine.ssd_read_bps / miss_bpt


def _fmt_gib(b: float) -> str:
    return f"{b / GIB:8.2f} GiB"


def report(shape: Shape, machine: Machine, budget: Budget, rl: Roofline,
           ctx_tokens: int) -> str:
    per_expert = shape.bytes_per_routed_expert()
    L = []
    A = L.append

    tag = "MEASURED" if machine.measured else "ASSUMED -- NOT YET MEASURED"
    A("=" * 72)
    A(f"DREAMFLASH DECODE ROOFLINE  --  {shape.name}")
    A(f"machine: {machine.name}   [{tag}]")
    A("=" * 72)

    A("")
    A("-- MODEL GEOMETRY (transcribed from ds4.c) " + "-" * 29)
    A(f"  layers                        {shape.n_layer}")
    A(f"  routed experts per layer      {shape.n_expert}")
    A(f"  experts used per token/layer  {shape.n_expert_used}")
    A(f"  expert FFN dim                {shape.n_ff_exp}")
    A(f"  total routed experts          {shape.total_routed_experts:,}")
    A(f"  bytes per routed expert       {per_expert/1024/1024:.3f} MiB"
      "   (IQ2_XXS gate/up + Q2_K down)")
    A(f"  routed weights total          {_fmt_gib(shape.total_routed_params * 2.25 / 8)}"
      "  (approx, mixed quant)")
    A(f"  non-routed params             {shape.nonrouted_params/1e9:.1f} B")

    A("")
    A("-- PER-TOKEN TRAFFIC " + "-" * 51)
    A(f"  expert loads per token        {shape.expert_loads_per_token()}"
      f"   ({shape.n_layer} layers x {shape.n_expert_used})")
    A(f"  cold-cache bytes per token    {rl.bytes_per_token_cold/1e9:.3f} GB")
    A(f"  at {rl.target_tok_s:.1f} tok/s that demands     "
      f"{rl.required_ssd_bps_cold/1e9:.1f} GB/s of weight traffic")

    A("")
    A("-- MEMORY BUDGET " + "-" * 55)
    A(f"  VRAM total                    {_fmt_gib(machine.vram_bytes)}")
    A(f"    non-routed weights (Q8)     {_fmt_gib(budget.nonrouted_vram)}")
    A(f"    KV / compressed state       {_fmt_gib(budget.kv_state_vram)}"
      f"   @ {ctx_tokens:,} ctx")
    A(f"    graph scratch               {_fmt_gib(budget.graph_scratch_vram)}")
    A(f"    -> VRAM expert cache        {_fmt_gib(budget.vram_expert_cache)}"
      f"   = {max(0, int(budget.vram_expert_cache // per_expert)):,} experts")
    A(f"  host RAM total                {_fmt_gib(machine.ram_bytes)}")
    A(f"    OS reserve                  {_fmt_gib(budget.os_reserve_ram)}")
    A(f"    pinned staging              {_fmt_gib(budget.pinned_staging_ram)}")
    A(f"    -> host expert cache        {_fmt_gib(budget.host_expert_cache)}"
      f"   = {max(0, int(budget.host_expert_cache // per_expert)):,} experts")

    A("")
    A("-- THE BINDING CONSTRAINT " + "-" * 46)
    A(f"  cache holds                   {rl.cache_capacity_experts:,}"
      f" / {rl.total_experts:,} experts"
      f"  = {rl.capacity_fraction*100:.1f}% of the model")
    A(f"  SSD sustained read            {machine.ssd_read_bps/1e9:.1f} GB/s")
    A(f"  PCIe host->device             {machine.pcie_h2d_bps/1e9:.1f} GB/s")
    A("")
    A(f"  required hit rate (SSD)       {rl.required_hit_rate_ssd*100:6.1f}%"
      "   <-- combined VRAM+RAM")
    A(f"  required hit rate (PCIe)      {rl.required_hit_rate_vram*100:6.1f}%"
      "   <-- VRAM only")

    binding = ("SSD" if rl.required_hit_rate_ssd > rl.required_hit_rate_vram
               else "PCIe")
    A(f"  binding constraint            {binding}")

    A("")
    A("-- ATTAINABLE DECODE RATE vs HIT RATE " + "-" * 34)
    A("  (SSD bandwidth bound only; compute/latency/sync make it worse)")
    A("")
    A("    hit rate   ceiling tok/s")
    for h in (0.0, 0.25, 0.50, 0.60, 0.667, 0.75, 0.80, 0.85, 0.90, 0.95):
        t = attainable_tok_s(shape, machine, h)
        mark = "  <-- target met" if t >= rl.target_tok_s else ""
        A(f"    {h*100:5.1f}%     {t:7.2f}{mark}")

    A("")
    A("-- VERDICT " + "-" * 61)
    need = rl.required_hit_rate_ssd
    if need <= 0:
        A(f"  {rl.target_tok_s:.0f} tok/s is reachable with NO cache at all.")
    elif need >= 1.0:
        A(f"  {rl.target_tok_s:.0f} tok/s is IMPOSSIBLE on this machine.")
        A("  Even a 100% hit rate cannot supply the required bandwidth.")
    else:
        A(f"  {rl.target_tok_s:.0f} tok/s requires a {need*100:.1f}% hit rate")
        A(f"  from a cache covering {rl.capacity_fraction*100:.1f}% of experts.")
        ratio = need / rl.capacity_fraction if rl.capacity_fraction else float("inf")
        A(f"  That is a {ratio:.1f}x locality amplification over uniform routing.")
        A("")
        if ratio > 5.0:
            A("  ASSESSMENT: at or beyond the edge of plausible MoE routing")
            A("  locality. Treat 10 tok/s as a stretch goal, not a plan.")
        elif ratio > 3.0:
            A("  ASSESSMENT: demanding but not absurd. Hinges entirely on")
            A("  measured routing skew -- run tools/trace_replay before betting.")
        else:
            A("  ASSESSMENT: plausible with a good replacement policy.")
    A("=" * 72)
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vram-gb", type=float, default=12.0)
    p.add_argument("--ram-gb", type=float, default=16.0)
    p.add_argument("--ssd-gbps", type=float, default=6.0,
                   help="sustained large-block random read, GB/s")
    p.add_argument("--pcie-gbps", type=float, default=24.0)
    p.add_argument("--ctx", type=int, default=100_000)
    p.add_argument("--kv-bytes-per-token", type=float, default=20_000.0,
                   help="PLACEHOLDER until measured; see docs/LONG_CONTEXT.md")
    p.add_argument("--target", type=float, default=10.0)
    p.add_argument("--measured", action="store_true",
                   help="assert these numbers came from hardware_probe")
    a = p.parse_args()

    m = Machine(
        name="target",
        vram_bytes=a.vram_gb * GIB,
        ram_bytes=a.ram_gb * GIB,
        ssd_read_bps=a.ssd_gbps * GB,
        pcie_h2d_bps=a.pcie_gbps * GB,
        measured=a.measured,
    )
    b = plan_budget(FLASH, m, a.ctx, a.kv_bytes_per_token)
    rl = compute(FLASH, m, b, a.target)
    print(report(FLASH, m, b, rl, a.ctx))


if __name__ == "__main__":
    main()
