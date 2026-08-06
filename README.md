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
make                                   # every test suite, Python and C
python tools/roofline/roofline.py      # the feasibility math
python tools/trace_replay/replay.py    # offline simulator policy comparison
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
3. **The SSD is the bottleneck; PCIe is not — *at 10 tok/s*.** 18.3 GB/s of
   demand at 10 tok/s vs ~24 GB/s of PCIe 4.0 x16. This inverts the priority
   order in the original brief — build the host-RAM cache tier, skip the
   elaborate PCIe pipelining. **This holds only near the 10 tok/s north star.**
   PCIe carries every host-tier hit *and* every SSD miss, so raising the hit rate
   moves traffic onto the bus rather than off it. Past roughly 29 tok/s on this
   box, PCIe binds and the SSD stops mattering — see "Reaching 30 tok/s" below.
4. **Cold-cache ceiling is 3.29 tok/s.** Milestones M4 (1) and M5 (3) need no
   cache at all. M6+ is *purely* a cache-hit-rate problem.
5. **The best hardware upgrade is more host RAM — then a second NVMe, and only
   then a bigger GPU.** Doubling SSD bandwidth drops the required hit rate from
   67% to ~34%. But host RAM is cheaper per tok/s than either: it removes SSD
   traffic outright rather than making it faster, and once RAM is at 64 GB a
   second NVMe adds nothing at all. See "Beyond 13" below.

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
  prompt structure. These numbers characterize the generator, not the model — run
  `python tools/trace_replay/replay.py --sweep-zipf` to see the range: LRU spans
  **29.1% at `s=0.6` to 87.0% at `s=1.8`**, a 58-point swing, and the verdict on
  the north star flips from "no" to "yes" between `s=1.0` and `s=1.2`. Until
  Phase 0 produces a measured trace, **the range is the finding**; any single hit
  rate quoted from this repo is a choice of exponent, not a property of
  DeepSeek-V4-Flash.

  Worth noticing: the default `s=1.2` yields 67.29% against the 67.1% the target
  requires. Of every exponent in the plausible range, the default is the one that
  makes 10 tok/s look *just* achievable. That may be coincidence, but a default
  landing on the threshold to two decimal places is a number to re-derive rather
  than rely on.
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
- [ ] **Adjacent-token routing overlap.** From the same trace, measure what
      fraction of a token's top-6 experts are also selected by the next token, per
      layer. This sets how many *distinct* experts a speculative verification pass
      must fetch, and it is the most powerful single parameter in the whole model:
      0.80 → 0.90 is worth **+55%**, and it also decides the optimal draft width
      (at 0.70, K barely matters; at 0.90, wider is much better). It is currently
      a guess (`--overlap`, default 0.80). **Done when:** you can state the
      measured overlap and re-run `run_speculative.py --overlap <measured>`.

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

- [~] `ExpertIOBackend` interface + `pread` backend (`src/expert_io.c`), with
      O_DIRECT, alignment enforcement, short-read reporting, and a Win32
      overlapped backend. **`io_uring` is still missing** — requesting it
      downgrades to `pread`, and `expert_io_backend_in_use()` reports the
      downgrade rather than letting a caller assume async.
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
- [ ] **Double-buffer the SSD→host and host→VRAM stages.** Highest-value item in
      this phase: worth ~29%, more than every replacement-policy improvement
      combined. Read expert N+1 off disk while expert N is crossing the bus.
      Model it with `run_speculative.py --pipeline`. **Done when:** measured
      transfer time per pass approaches `max(t_ssd, t_pcie)` rather than their sum.
- [ ] Per-expert telemetry: VRAM hit / RAM hit / SSD miss / prefetch hit /
      prefetch waste / bytes / latencies / GPU wait.

### Phase 4 — speculative verification & performance optimization

- [x] Speculative decoding candidate verification engine (`tools/speculative_decoding/`).
      Deduplicates expert accesses across draft trees. **Proves nothing about
      throughput** — see the caveat below before quoting any number it prints.
- [x] Speculative-aware cache replacement policy (`SpeculativeAwarePolicy` in `tools/speculative_decoding/speculative_policy.py`).

> **The "32.2 tok/s" figure previously claimed here was withdrawn, and the
> engine behind it has been rebuilt.** The original number was misattributed (it
> needed an unstated 85% acceptance probability on top of the 75% hit rate) and
> circular: wall-clock time was modelled as SSD reads for cache-missed experts
> and nothing else, so throughput reduced to `accepted_tokens / (1 - hit_rate)`
> with the hit rate as a free input. At 0.97 it reported 227 tok/s. There was
> also no cache in it — misses were an i.i.d. coin flip per expert per pass.
>
> What it does now:
> - **Hit rate is an output, not an input.** Expert reads are served against the
>   same two-tier LRU cache `trace_replay` uses, so the hit rate emerges from
>   capacity and access pattern. `--cache-hit-rate` is gone; capacities come from
>   `plan_budget()`.
> - **Wall clock is charged properly** (`cost_model.py`): VRAM streaming of the
>   ~6.9 GiB non-routed weights, PCIe for host-tier hits, SSD for misses,
>   per-layer kernel launch, and the draft model's own K sequential passes. Every
>   bandwidth is still a labeled placeholder.
> - **It reports a range, not a point.** Overlap efficiency is unmeasured, so it
>   prints a serialized (nothing overlaps) and an ideal (transfer fully hidden)
>   figure. Quote the serialized end.
> - **Speedup is measured against the same cost model**, not against SSD-only
>   cold decode, so it isolates speculation instead of crediting it with the cache.
>
> Current output at the default 18.9% capacity: **6–9 tok/s serialized**, at an
> emergent 37–54% hit rate. Two results worth noting: nothing in the matrix
> reaches 30 tok/s, and hit rate *falls* as K grows (54% at K=1 to 37% at K=6)
> because wider draft trees touch more distinct experts. The old fixed-input
> model could not express either.
>
> What still limits it: the expert access stream remains synthetic — the
> cross-candidate overlap is a hardcoded 0.80 — and acceptance probability is
> still an input, since it is a property of a draft model that does not exist yet.
>
> What the engine always got right: the numerator is genuinely accepted
> user-visible tokens (it stops at first rejection, and charges rejected drafts'
> expert reads to the cost), which is the repo's stated non-negotiable.

---

## Reaching 30 tok/s

Measured with `tools/speculative_decoding/` at K=5, 85% acceptance, on the
placeholder bandwidths. Where one pass costs 498 ms, it splits:

| Term | Per pass | Share |
|---|---:|---:|
| SSD (misses) | 337.9 ms | 67.9% |
| PCIe (host hits + misses) | 127.8 ms | 25.7% |
| VRAM (weights + experts) | 21.3 ms | 4.3% |
| Draft model | 10.7 ms | 2.2% |
| Kernel launch | 0.2 ms | 0.0% |

Compute is 6% of the problem. This is an I/O problem end to end, which is
consistent with findings #3 and #6.

**The stock box cannot reach 30.** Holding the hardware fixed (12 GiB VRAM,
16 GiB RAM, one 6 GB/s NVMe, PCIe 4.0 x16):

| Configuration | tok/s |
|---|---:|
| As configured | 9.19 |
| Perfect replacement policy (unbounded host tier) | 12.46 |
| **+ SSD cost eliminated entirely (perfect prefetch)** | **28.84** |

That last row is a hard ceiling: every expert already resident in host RAM and
the SSD free. It is still under 30, because PCIe and the ~6.9 GiB of non-routed
weights remain. **No amount of policy, prefetch, or cache work reaches 30 tok/s
on this machine.** Anything claiming otherwise has changed an assumption.

What does reach it:

| Change | tok/s |
|---|---:|
| 2x NVMe RAID0 (12 GB/s) | 13.94 |
| 4x NVMe RAID0 (24 GB/s) | 18.79 |
| 4x NVMe + PCIe 5.0 x16 (48 GB/s) | 25.52 |
| 4x NVMe + PCIe 5.0 + 24 GB card (VRAM tier 2,000 experts) | **32.46** |

Note the ordering: SSD bandwidth is the largest single lever until roughly
19 tok/s, after which PCIe binds and further SSD spend buys nothing. Raising the
*VRAM* hit rate is the only lever that removes PCIe traffic rather than
relocating it, and that needs a bigger card — at the stock 339-expert VRAM tier
a single K=5 pass touches ~456 distinct experts, so the VRAM tier thrashes by
construction and contributes only ~23 hits per pass.

**Honest summary: 10 tok/s is a plausible target on this box; 30 is a hardware
purchase.** All of the above rests on placeholder bandwidths and a synthetic
access stream — the ranking of levers is more trustworthy than the absolute
numbers, and Phase 0 is what turns either into evidence.

### Best achievable in software, same hardware, same quantization

Cumulative, measured at K=5 / 85% acceptance:

| Change | tok/s (serial) | tok/s (ideal) |
|---|---:|---:|
| Baseline (LRU, serialized transfer) | 9.19 | 9.78 |
| + LFU replacement | 9.93 | 10.63 |
| + TinyLFU admission control | 9.95 | 10.65 |
| **+ pipelined SSD→host→VRAM (double-buffered)** | **12.83** | **14.02** |
| + K tuned to 8 | 13.07 | 14.22 |

**~13 tok/s serialized, ~14 with good overlap — the 10 tok/s north star clears
with room to spare.** The first two rows are already the defaults; the third is
`--policy tinylfu`; the fourth is `--pipeline` and is *not implemented yet*.

Two things are worth taking from this:

1. **Pipelining the two transfer stages is the single biggest software lever
   (+29%), and it dwarfs all replacement-policy work combined (+8%).** SSD and
   PCIe are 68% and 26% of the pass; double-buffering removes the smaller of them
   almost entirely. Phase 3 should be built around this, not around policy.
2. **Replacement-policy work is close to exhausted.** LFU, LRU-2,
   recency/frequency hybrid, TinyLFU admission, and a statically pinned hot set
   all land within 39–41% hit rate and 9.9–10.1 tok/s. The spread across
   *policies* is under 2%, while the spread across *access distributions* (the
   `zipf_s` sweep) is 58 points. Measuring real routing skew is worth far more
   than any further policy tuning.

Not pursued, and why: a more aggressive quant cuts SSD and PCIe proportionally
and is by far the largest remaining lever, but it trades against quality and the
rules below bar it from counting in the default path. Reclaiming VRAM from the
`kv_bytes_per_token` placeholder could enlarge the VRAM tier, but that number is
a guess — measure it before spending it. (It is also worth little: dropping
context to 8K only moves the VRAM tier 339 → 599 experts and buys 0.3 tok/s.)

### Beyond 13: host RAM is the cheap lever, not the GPU

**Correction to the table above.** That "what reaches 30" list was computed
without pipelining and without considering a RAM upgrade, and it is misleading as
a result. It is far cheaper than it suggests. Measured at steady state
(512-token warm-up), K=8, pipelined:

| Configuration | serial | ideal | hit% | cost |
|---|---:|---:|---:|---|
| Stock box | 14.47 | 15.82 | 40.9% | — |
| **32 GB RAM** | **20.15** | 22.87 | 59.1% | ~$60 |
| **64 GB RAM** | **38.20** | 49.29 | 90.1% | ~$150 |
| 2nd NVMe RAID0 | 26.67 | 31.63 | 40.9% | ~$120 |
| 2nd NVMe + 64 GB RAM | 38.20 | 49.29 | 90.1% | ~$270 |

Host RAM is the cheapest tok/s in the system, and by a wide margin — it removes
SSD traffic outright rather than making it faster. Note the last row: once RAM is
at 64 GB the second NVMe adds *nothing*, because at a 90% hit rate the SSD is
barely on the critical path. **Buy RAM before NVMe, and NVMe before GPU.**

**How much of this survives the assumptions** — the part that matters:

| `zipf_s` | 16 GB | 32 GB | 64 GB |
|---:|---:|---:|---:|
| 0.8 | 12.14 | 16.97 | 33.69 |
| 1.0 | 12.60 | 17.61 | 33.52 |
| 1.2 (default) | 14.47 | **20.15** | 38.20 |
| 1.4 | 15.01 | 21.01 | 38.54 |

**"32 GB reaches 20" is not robust** — it holds only at the default skew and
above; at `s=0.8`–`1.0` it lands at ~17. **"64 GB reaches 30+" is robust**, since
a 64 GB host tier holds enough of the model that skew stops mattering.

And both require pipelining, which is **not implemented**. Without it, 32 GB RAM
gives 12–15 tok/s across the whole skew range. The order of work is therefore:
build the double-buffered streaming path first, then buy RAM. Buying RAM without
the streaming path buys almost nothing.

### On MTP as the drafter

DeepSeek ships a Multi-Token Prediction head intended exactly for this. Adopt it —
it is free and it is the right design — but do not expect much here: **it is worth
about +2.6%** (14.47 → 14.84 tok/s). MTP's saving is that it shares the target's
trunk instead of running separate draft weights, i.e. it saves *compute*. Compute
is 6% of a pass on this box. In a normal compute-bound deployment MTP is a large
win; against expert streaming it is rounding error.

The parameter that *does* matter is easy to confuse with it. `--overlap` is not a
property of the drafter at all — it is how much the **target's own routing** agrees
between adjacent token positions, which sets how many distinct experts a pass must
fetch:

| overlap | tok/s | distinct experts/pass | SSD GB/pass |
|---:|---:|---:|---:|
| 0.80 (assumed) | 14.47 | 529 | 2.21 |
| 0.85 | 17.05 | 442 | 1.68 |
| 0.90 | 22.08 | 381 | 1.28 |
| 0.95 | 29.97 | 319 | 0.87 |

It also decides the optimal draft width — at 0.70 the choice of K is worth almost
nothing, at 0.90 wider windows keep paying (K=16 → 24.03). **So K cannot be tuned
before overlap is measured**, and it is now a Phase 0 item.

Note what this table is: a sensitivity analysis of a guessed constant, not a
result. Nobody has measured routing overlap on DeepSeek-V4. It is the single
highest-value measurement left in the project, and it is one instrumented forward
pass away.

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
src/expert_io.c                pread/Win32 expert read engine (no io_uring yet)
src/memory_manager.c           VRAM/host residency tracker (slot counts, not bytes)
tests/test_roofline.py         15 roofline consistency tests
tests/test_trace_replay.py      7 trace replay unit tests
tests/test_discrete_config.py  2 discrete configuration tests
tests/test_speculative_decoding.py 4 speculative decoding unit tests
tests/test_expert_io.c         40 checks over the C engine (`make test-c`)
```

## Tooling caveats worth knowing before you trust output

- `tools/hardware_probe/pcie_probe.py` requires a real HIP/CUDA device and exits
  non-zero when there isn't one. It has no simulation mode by design — the
  previous version reported a host-to-host `bytearray` slice as PCIe bandwidth.
- `tools/hardware_probe/ssd_probe.py` defaults to a 4 GB probe file and drops the
  page cache between runs. It reports cold *and* warm, and sets
  `trustworthy: false` in its JSON (plus a non-zero exit) if the two are within
  10% on buffered I/O, which means eviction did not work.
- `tools/download_chunked.py` refuses to append to a partial file unless the
  server returns 206 with a matching `Content-Range`. A server that ignores
  `Range` would otherwise append a second full copy and silently corrupt the file.
- Neither downloader verifies checksums. Size is not integrity.

Everything else in the brief's deliverable list is still to be created; the
work plan above is the order to create it in.
