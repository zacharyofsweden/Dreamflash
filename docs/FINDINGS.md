# Upstream findings — what DwarfStar actually is, and where the real gap is

**Status:** verified by direct source inspection.
**Upstream:** `antirez/ds4` @ `b0309611041655f4e45671cfd9c9886aff161406` (2026-08-05)
**Reference:** `NeelM0906/turbo-fieldfare` @ `f8abc4422e33a8808d5a5c1032a0e97ed5aa5118`

Every claim below cites `file:line`. Claims without a citation are marked
**[INFERENCE]** and must be treated as unverified.

---

## 1. The single most important finding

**The DS4 ROCm backend is not a low-VRAM streaming design. It is a
128 GB unified-memory APU design.** The Dreamflash target — a *discrete* GPU
with 12 GB VRAM and 16 GB host RAM — is a fundamentally different machine, and
the gap is architectural, not a matter of tuning.

Evidence:

| Evidence | Location |
|---|---|
| `make rocm` is a literal alias for `make strix-halo` | `Makefile:161` |
| GPU arch hardcoded to `gfx1151` (Strix Halo APU) | `Makefile:57` |
| Expert cache uses `hipMallocManaged` (unified/demand-paged) | `rocm/ds4_rocm_runtime.cuh:5946`, `ds4_rocm.h:35` |
| Runtime comment: "Headroom kept free on the **(unified-memory) device**" | `rocm/ds4_rocm_runtime.cuh:1519` |
| Cache comment: "expert cache lives in the same physical RAM as the OS on a **unified-memory APU**" | `ds4_ssd.c:99` |
| Setup doc targets **128 GB RAM**, an **80.76 GiB** model, and a 124 GB GTT aperture | `STRIXHALO.md` §3 |
| Default free-reserve headroom is **16 GiB** — larger than the entire target VRAM | `rocm/ds4_rocm_runtime.cuh:1524` |
| Managed-KV reserve floor is **8 GiB** | `rocm/ds4_rocm_runtime.cuh:1525` |
| README's streaming guidance recommends **32–59 GB** expert caches | `README.md:303,348` |

On Strix Halo, "SSD streaming" works because the GPU addresses system RAM
directly. An expert "miss" is an OS page fault serviced from page cache or
disk. **There is no PCIe hop and no explicit copy anywhere in that path.** On a
discrete GPU that entire mechanism is unavailable: `hipMallocManaged` on
consumer RDNA does not provide performant GPU-side demand paging (no XNACK on
most consumer parts) — **[INFERENCE]**, must be confirmed on the actual card as
milestone M0.

## 2. `ds4_ssd.c` is not an I/O engine

The prompt brief lists "SSD expert-streaming infrastructure" as existing.
Read the file: all 7 KB of it is

- CLI argument parsing (`ds4_parse_gib_arg`, `ds4_parse_streaming_cache_experts_arg`)
- cache-size *planning arithmetic* (`ds4_ssd_auto_cache_plan`)
- an `mlock` helper used only by `--simulate-used-memory`

There is **no** read path, no `io_uring`, no `pread` loop, no queue, no
staging buffer, no completion handling. Expert bytes reach the GPU via the
mmap'd GGUF and managed memory. So §12 of the brief ("implement the NVMe I/O
engine") is genuinely **new construction**, not an extension.

## 3. What DOES exist and is worth keeping

Do not rewrite these:

- A complete, current DeepSeek-V4-Flash implementation (`ds4.c`, 2.9 MB).
- A real ROCm kernel suite — 24 files in `rocm/`, ~1.2 MB, including
  `ds4_rocm_moe.cuh` (195 KB), `ds4_rocm_attention.cuh` (59 KB),
  `ds4_rocm_q8.cuh` (71 KB), `ds4_rocm_indexer.cuh` (47 KB).
  These are *shared source* with CUDA via `__HIP_PLATFORM_AMD__` (`ds4_rocm.cu:1`).
- A resident expert-cache subsystem with LRU eviction already:
  `cuda_stream_resident_evict_at`, `cuda_stream_selected_is_current`,
  `g_stream_resident_experts` in `rocm/ds4_rocm_runtime.cuh:~1500`.
  **It evicts by `last_used` and refuses to evict currently-selected experts** —
  so the brief's §10 "cache exists but re-reads every token" failure mode is
  *not* present upstream. Verify before assuming otherwise.
- A 205 KB static hot-expert list (`ds4_streaming_hotlist.inc`) — direct
  evidence that routing skew is real and already exploited upstream.
- Correct backend gating: on a ROCm build, SSD streaming **is** permitted
  (`ds4.c:399-410`, `ds4.c:416-421`), and the model-generic streaming helpers
  (`ds4_streaming_cacheable_expert_count`, `ds4.c:4569`) are keyed on
  `DS4_N_EXPERT`/`DS4_N_LAYER`, not on GLM.

### Correction to a plausible misreading

`README.md:281` says streaming is available "on Metal and for GLM 5.2 on ROCm",
which reads as though DS4-Flash + ROCm + streaming is *blocked*. The code says
otherwise — the gate at `ds4.c:399` allows it. The README is describing what has
been **validated**, not what is **permitted**. Treat DS4F-on-ROCm streaming as
*untested*, not *unsupported*. That distinction sets the risk level for M2.

## 4. The roofline — why this is hard

Derived in `tools/roofline/`, all geometry transcribed from
`ds4.c:541` (`DS4_SHAPE_FLASH`). Run `python tools/roofline/roofline.py`.

```
bytes per routed expert       6.750 MiB     (IQ2_XXS gate/up + Q2_K down)
expert loads per token        258           (43 layers x 6 experts)
cold-cache bytes per token    1.826 GB
demand at 10 tok/s            18.3 GB/s
```

Cross-check: 11,008 experts x 25.17 M params = **277 B** routed, leaving 7 B
non-routed against the published 284 B total. Consistent.

With 12 GB VRAM / 16 GB RAM at 100K context, the cache holds ~2,083 of 11,008
experts (**18.9%**), and:

| Constraint | Required hit rate at 10 tok/s |
|---|---|
| SSD @ 6 GB/s | **67.1%** |
| PCIe 4.0 x16 @ 24 GB/s | *negative — has slack* |

### Two consequences that should drive the whole design

1. **PCIe is not the bottleneck. The SSD is.** At 24 GB/s the link carries the
   full 18.3 GB/s cold demand with room to spare. Optimizing PCIe transfer is
   therefore low-value, and — importantly — **a host-RAM cache tier is worth
   building even though every RAM hit still crosses PCIe**, because it converts
   a 6 GB/s SSD read into a 24 GB/s PCIe transfer. §11 of the brief is well
   motivated; §23.19 is nearly free; §13's elaborate PCIe pipelining is
   premature.

2. **The honest ceiling.** With a cold cache the SSD alone permits **3.29 tok/s**.
   So M4 (1 tok/s) and M5 (3 tok/s) are reachable with *no cache at all*, and
   everything from M6 up is purely a cache-hit-rate problem. Reaching 10 tok/s
   needs 67% hits from 18.9% capacity — a **3.5x locality amplification**.

**Assessment:** demanding but not absurd. MoE routing is genuinely skewed (the
205 KB upstream hotlist is proof), but 3.5x is a real bet, not a rounding
error. The deciding measurement is the routing-skew trace — which is why
`tools/trace_replay/` must be built **before** any kernel work. If measured
skew yields <67%, the correct output of this project is an honest
"highest sustained X tok/s + the exact bottleneck", exactly as the brief's
§25 demands.

## 5. Recommended next hardware upgrade (brief §31 item 25)

The roofline answers this directly, and the answer is **not** a bigger GPU:

- **A faster / second NVMe is the highest-leverage upgrade.** Required hit rate
  falls with SSD bandwidth; `--ssd-gbps 12` (Gen5, or two Gen4 striped) drops
  the requirement from 67% to ~34%, which is comfortably inside plausible
  routing skew. Brief §23.1 (dual-NVMe striping) is thus far more valuable than
  it is presented as being.
- **More host RAM beats more VRAM per dollar**, because the host cache is
  ~5x the size of the VRAM cache in this budget and PCIe has slack.

Verify both with `roofline.py --ssd-gbps N --ram-gb N`.

---

## Open questions — MUST be resolved by measurement, not assumption

1. Does `hipMallocManaged` work at all on the target discrete RDNA card, and at
   what page-fault cost? Determines whether the existing path degrades or dies.
2. What is the real KV/compressed-state size per token at 100K? The roofline
   currently uses a **placeholder** 20 KB/token. `docs/LONG_CONTEXT.md` owes a
   measured number; it directly trades against expert-cache VRAM.
3. What is the actual measured routing skew? Everything above M5 depends on it.
4. Does the upstream `g_stream_resident_experts` cache survive across tokens on
   a discrete device, or does it thrash? Read
   `rocm/ds4_rocm_runtime.cuh` `cuda_stream_resident_*` in full.
5. Real sustained large-block random read on the target SSD at QD32 —
   6 GB/s is an assumption.
