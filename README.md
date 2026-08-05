# Dreamflash

Running **DeepSeek-V4-Flash** (284 B params, 13 B active, 43 layers, top-6 MoE)
on a consumer box with **12 GB VRAM + 16 GB RAM + NVMe SSD**, via
[`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar 4).

The model is ~81 GiB. The machine has 28 GB of memory in total. Every token
must stream experts from disk. This repo is the analysis, budget math, and
work plan that makes that tractable — and honest about where it stops.

> **North star:** 10 accepted tok/s sustained, batch 1, ≥512 generated tokens,
> at a 100K-token context, with no layers or experts dropped.

---

## Start here

```bash
python tools/roofline/roofline.py      # the feasibility math
python tests/test_roofline.py          # 15 consistency tests
```

Then read **[docs/FINDINGS.md](docs/FINDINGS.md)**. It is the highest-value
document in the repo and it will save you from three wrong turns.

## What is already settled

The hard analytical work is done. These are conclusions, not guesses — each is
backed by `file:line` citations into upstream in `docs/FINDINGS.md`.

1. **The ROCm backend is a 128 GB unified-memory APU design**, not a low-VRAM
   streaming design. `make rocm` is an alias for `make strix-halo`; the expert
   cache is `hipMallocManaged`; the default free-reserve is 16 GiB — larger
   than the entire target VRAM. The gap is architectural.
2. **`ds4_ssd.c` contains no I/O engine.** Only arg parsing and cache-size
   arithmetic. The NVMe read path is new construction.
3. **The SSD is the bottleneck; PCIe is not.** 18.3 GB/s of demand at 10 tok/s
   vs ~24 GB/s of PCIe 4.0 x16. This inverts the priority order in the original
   brief — build the host-RAM cache tier, skip the elaborate PCIe pipelining.
4. **Cold-cache ceiling is 3.29 tok/s.** Milestones M4 (1) and M5 (3) need no
   cache at all. M6+ is *purely* a cache-hit-rate problem.
5. **10 tok/s needs a 67% hit rate from 18.9% capacity** — a 3.5x locality
   amplification. Plausible, unproven. Measure skew before betting on it.
6. **The best hardware upgrade is a faster/second NVMe, not a bigger GPU.**
   Doubling SSD bandwidth drops the required hit rate from 67% to ~34%.

## Numbers you will need constantly

| Quantity | Value | Source |
|---|---:|---|
| Bytes per routed expert | 6.750 MiB | derived, `tools/roofline/model.py` |
| Expert loads per token | 258 | 43 layers x 6 |
| Cold-cache bytes per token | 1.826 GB | derived |
| Total routed experts | 11,008 | 43 x 256 |
| Non-routed params (must stay VRAM-resident) | ~7 B (~6.9 GiB @ Q8) | 284 B − 277 B |
| Upstream commit | `b030961` | 2026-08-05 |

---

## Work plan

Ordered by dependency. Each task states its **done-when** so it can be verified
rather than believed. Tasks marked **[MEASURE]** need the real Linux/AMD target
machine; the rest can be done anywhere.

### Phase 0 — ground truth *(blocks everything above M5)*

- [ ] **[MEASURE]** `tools/hardware_probe/` — SSD sequential + random read at
      4K/64K/1M/8M, QD 1–64, buffered vs `O_DIRECT` vs `io_uring`, cold and warm
      page cache. Record GB/s, IOPS, p50/p95/p99.
      **Done when:** `results/hardware/ssd.json` exists and
      `roofline.py --ssd-gbps <measured> --measured` runs clean.
      *Why first: the entire feasibility verdict currently rests on an assumed
      6 GB/s.*
- [ ] **[MEASURE]** PCIe h2d/d2h, pinned vs pageable, 1–64 MB, 1/2/4 streams.
      **Done when:** confirms (or refutes) that PCIe has slack.
- [ ] **[MEASURE]** `docs/HARDWARE_REPORT.md` — exact GPU, `gfx` string, VRAM,
      ReBAR status, ROCm version. **Critical:** confirm whether the card is
      ROCm-supported at all, and whether `hipMallocManaged` functions on it.
- [ ] **Routing-skew trace.** Instrument expert selection, dump
      `results/traces/*.jsonl` over the calibration corpora (coding, chat, math,
      long-doc, tool-use). **Done when:** you can state the measured hit rate at
      18.9% capacity under LRU. *This single number decides whether 10 tok/s is
      a plan or a fantasy.*

### Phase 1 — the offline simulator *(no GPU needed — highest value per hour)*

- [ ] `tools/trace_replay/` — replay traces against pluggable policies (LRU,
      LFU, LRU-K, recency-frequency hybrid, cost-aware, Belady oracle).
      Report hit rate, SSD bytes, PCIe bytes, evictions, wasted prefetch.
      **Done when:** Belady gives the upper bound and LRU the lower, and the
      gap between them tells you how much policy work is worth doing.
      *Do this before writing a single kernel.*

### Phase 2 — make it run at all

- [ ] Split the build: separate `ROCM backend` / `GPU arch` / `unified-memory`
      / `discrete-memory` axes. Today `make rocm` == `make strix-halo` with
      `gfx1151` hardcoded (`Makefile:57,161`). Build-only patch, no behavior change.
- [ ] Audit every unified-memory assumption listed in `docs/FINDINGS.md` §1 and
      make the memory-sizing constants configurable (the 16 GiB free-reserve and
      8 GiB managed-KV reserve are both larger than the target's whole VRAM).
- [ ] **[MEASURE]** Baseline attempts at 512 / 2048 / 4096 / 8192 context.
      **Record complete failures as valid evidence** — a clean OOM with a log is
      a result, not a setback.

### Phase 3 — the streaming path

- [ ] `ExpertIOBackend` interface + `io_uring` backend + `pread` fallback.
      This is new code; nothing upstream to extend.
- [ ] Three-tier memory manager (VRAM / pinned host RAM / SSD) with the byte
      budget from `plan_budget()`. Non-routed weights are VRAM-pinned and
      non-negotiable — they are touched every layer, every token.
- [ ] Persistent VRAM expert cache keyed by `(layer, expert, tensor_group,
      quant_type)`, with generation counters. **First read the existing
      `cuda_stream_resident_*` cache in `rocm/ds4_rocm_runtime.cuh` — it already
      does LRU and already refuses to evict in-use experts. Extend, don't
      replace.**
- [ ] Host-RAM warm cache. Justified by finding #3: converts a 6 GB/s SSD read
      into a 24 GB/s PCIe transfer.
- [ ] Per-expert telemetry: VRAM hit / RAM hit / SSD miss / prefetch hit /
      prefetch waste / bytes / latencies / GPU wait.

### Phase 4 — only if Phase 1 says it pays

Prefetching, cache-aware DSpark, CPU-vs-PCIe expert scheduling, HIP graphs,
sidecar repacking with co-activation ordering, TurboQuant-style compressed
state. **Every one of these is a controlled experiment with a keep/revert
decision recorded in `docs/BOTTLENECK_HISTORY.md`.**

---

## Rules that are not negotiable

Inherited from the brief, and they are what make the result mean anything:

- **Never** report proposed draft tokens, prefill tokens, cached responses, or
  short bursts as decode throughput. Accepted user-visible tokens per wall-clock
  second, or nothing.
- **Never** drop layers, drop experts, shorten context, or reduce top-6 routing
  in the default path. Approximations live behind `--experimental-approx-*` and
  **cannot count toward the 10 tok/s target**.
- A valid speed claim needs ≥512 accepted tokens, ≥3 different prompts,
  5 measured runs, median reported, no OOM, no swap, no thermal throttling.
- Cache hit rate is not the metric. Net decode speed is. A hit that adds
  synchronization can still lose.
- Correctness milestones are never skipped to reach a speed milestone.
- Server binds to `127.0.0.1` only.

## Repo layout

```
docs/FINDINGS.md          upstream analysis — READ FIRST
tools/roofline/model.py   model geometry, transcribed from ds4.c
tools/roofline/roofline.py  feasibility calculator
tests/test_roofline.py    15 consistency tests
```

Everything else in the brief's deliverable list is still to be created; the
work plan above is the order to create it in.
