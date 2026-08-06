# Dreamflash

Running **DeepSeek-V4-Flash** (284 B params, 13 B active, 43 layers, top-6 MoE)
on a consumer box with **12 GB VRAM + 16 GB RAM + NVMe SSD**, via
[`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar 4).

The model is ~79.5 GiB at the quantization assumed here (72.56 GiB routed +
6.90 GiB non-routed; upstream's 80.76 GiB figure is a *different* quant). The
machine has 28 GiB of memory in total. Every token
must stream experts from disk. This repo is the analysis, budget math, and
work plan that makes that tractable — and honest about where it stops.

> **North star:** 10 accepted tok/s sustained, batch 1, ≥512 generated tokens,
> at a 100K-token context, with no layers or experts dropped.

---

## Start here

```bash
python tools/roofline/roofline.py      # the feasibility math
python tests/test_roofline.py          # 15 consistency tests
python tools/trace_replay/replay.py    # offline simulator policy comparison
python tests/test_trace_replay.py      # 7 trace replay unit tests
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
5. **The best hardware upgrade is a faster/second NVMe, not a bigger GPU.**
   Doubling SSD bandwidth drops the required hit rate from 67% to ~34%.

Every one of the above depends on the assumed quantization (IQ2_XXS gate/up +
Q2_K down, ~2.25 bpw). At Q4_K throughout, the cold-cache ceiling halves to
1.64 tok/s and 10 tok/s would need an 83.6% hit rate. Nothing here establishes
that such a quant of DS4-Flash exists or is what will be run — confirm it before
treating any of these numbers as load-bearing.

## What is *not* settled, despite appearing elsewhere in this repo

- **"10 tok/s needs a 67% hit rate from 18.9% capacity — a 3.5x amplification."**
  The 67% is solid (it is bandwidth arithmetic). The **18.9% is not**: it falls
  out of four unmeasured placeholders in `roofline.py` — `kv_bytes_per_token`
  (20,000, explicitly a placeholder, worth 1.86 GiB of the 12), 1 GiB graph
  scratch, 3 GiB OS reserve, 1.5 GiB pinned staging — and the VRAM expert cache
  is only 2.24 GiB, so a 2x error in the KV placeholder alone swings capacity by
  ~±40%. The 3.5x figure inherits all of that.
- **Every hit rate produced by `tools/trace_replay/`.** There is no real trace in
  this repo. The default is synthetic i.i.d. Zipf with `zipf_s=1.2`, an uncited
  constant, applying the same expert ranking to all 43 layers with no temporal or
  prompt structure. At `zipf_s=0.6` LRU gets 28%; at 1.8 it gets 84%. These
  numbers characterize the generator, not the model.
- **Every throughput figure from `tools/speculative_decoding/`.** See Phase 4.
- **Everything in `results/hardware/`.** See Phase 0.

## Numbers you will need constantly

**Watch the units:** expert sizes are binary (MiB/GiB), bandwidth and
bytes-per-token are decimal (GB, GB/s), so that `1.826 GB ÷ 6 GB/s` is
dimensionally clean. 258 × 6.750 MiB = 1741.5 MiB = **1.700 GiB = 1.826 GB**.
If you read that row as GiB you will be 7.4% off.

| Quantity | Value | Source |
|---|---:|---|
| Bytes per routed expert | 6.750 MiB | derived @ ~2.25 bpw, `tools/roofline/model.py` |
| Expert loads per token | 258 | 43 layers x 6 |
| Cold-cache bytes per token | 1.826 GB (= 1.700 GiB) | derived |
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

- [x] `tools/trace_replay/` — replay traces against pluggable policies (LRU,
      LFU, LRU-K, recency-frequency hybrid, cost-aware, Belady oracle).
      Report hit rate, SSD bytes, PCIe bytes, evictions, wasted prefetch.
      **Done when:** Belady and LRU bracket the policy space and the gap between
      them tells you how much policy work is worth doing.
      *Do this before writing a single kernel.*
      **Caveat:** Belady is optimal for a *single-tier* cache. This simulator is
      two-tier and applies Belady independently within VRAM and within host RAM,
      so it is a strong heuristic, not a proven upper bound — LFU beats it on
      constructed small-capacity cases. LRU is likewise a conventional baseline,
      not a lower bound; CostAware scores below it. Still open: run this against
      a **real** trace (Phase 0), not the synthetic Zipf.

### Phase 2 — make it run at all

- [x] Split the build: separate `ROCM backend` / `GPU arch` / `unified-memory`
      / `discrete-memory` axes. Parametric `Makefile` created for `gfx1100`,
      `gfx1101`, `gfx1030` alongside APU targets.
- [x] Audit every unified-memory assumption listed in `docs/FINDINGS.md` §1 and
      make the memory-sizing constants configurable (`include/ds4_discrete_config.h`
      and `tools/discrete_config/config_generator.py`).
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

### Phase 4 — speculative verification & performance optimization

- [x] Speculative decoding candidate verification engine (`tools/speculative_decoding/`).
      Deduplicates expert accesses across draft trees. **Proves nothing about
      throughput** — see the caveat below before quoting any number it prints.
- [x] Speculative-aware cache replacement policy (`SpeculativeAwarePolicy` in `tools/speculative_decoding/speculative_policy.py`).

> **The "32.2 tok/s" figure previously claimed here was wrong and has been
> withdrawn.** Three things were true of it. (1) It was misattributed: it needs
> a 75% cache hit rate **and** an 85% acceptance probability, and the README
> quoted only the first. The default run (70% hit rate) reaches 30 tok/s in
> *none* of its 24 rows. (2) It is circular. The simulator models wall-clock
> time as SSD read time for cache-missed experts and nothing else — no compute,
> no PCIe, no attention/KV, no draft-model cost, no verification cost — so
> throughput reduces to roughly `accepted_tokens / (1 - hit_rate)`, and the hit
> rate is a free input. Set it to 0.97 and the tool reports 227 tok/s. (3) The
> "deduplication across draft trees" it demonstrates is an
> `expert_locality_overlap = 0.80` constant injected by its own generator, and
> misses are an i.i.d. coin flip per expert per pass — there is no cache state
> in it. Treat its output as an I/O-bound-only upper bound and nothing more.
>
> What the engine *does* get right: the numerator is genuinely accepted
> user-visible tokens (it stops at first rejection, and charges rejected drafts'
> expert reads to the cost), which is the repo's stated non-negotiable.

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
docs/FINDINGS.md               upstream analysis — READ FIRST
include/ds4_discrete_config.h  discrete memory constants & C API
include/expert_io.h            asynchronous NVMe I/O interface
Makefile                       parametric build configuration
tools/roofline/model.py        model geometry, transcribed from ds4.c
tools/roofline/roofline.py     feasibility calculator
tools/trace_replay/            offline trace simulator & replacement policies
tools/discrete_config/         discrete GPU runtime config generator
tools/speculative_decoding/    speculative decoding & MoE candidate verification engine
tests/test_roofline.py         15 roofline consistency tests
tests/test_trace_replay.py      7 trace replay unit tests
tests/test_discrete_config.py  2 discrete configuration tests
tests/test_speculative_decoding.py 4 speculative decoding unit tests
```

Everything else in the brief's deliverable list is still to be created; the
work plan above is the order to create it in.
