"""Wall-clock cost model for one MoE verification pass at batch 1.

Modelling decode time as "SSD bytes for cache-missed experts, and nothing else"
makes throughput reduce to accepted_tokens / (1 - hit_rate), which is unbounded in
the hit rate and tells you nothing you did not already assume. This module charges
the other terms too.

Batch-1 decode is memory-bandwidth-bound, not FLOP-bound, so every term here is a
bytes/bandwidth quotient rather than a FLOP count:

  VRAM     non-routed weights (~6.9 GiB, touched every layer every token) plus the
           routed experts the kernels read, all streamed from VRAM. This is the
           term the current speculative literature actually exploits: K candidate
           tokens verified in one pass share a SINGLE non-routed weight read, so
           this cost amortizes across the pass while the SSD term does not.
  PCIe     experts served from the host RAM tier, plus anything fetched from SSD,
           must cross the bus into VRAM.
  SSD      experts resident in neither tier.
  Launch   fixed per-layer kernel overhead, which at 43 layers is not negligible
           once the other terms get small.
  Draft    the draft model runs K times SEQUENTIALLY before the pass, and it is not
           free. Omitting it is how speculative-decoding estimates get optimistic.

Overlap: transfer and compute can overlap in a well-pipelined implementation, but
how well is unmeasured. Rather than pick a number, `pass_time_seconds` returns a
RANGE -- serialized (nothing overlaps) and ideal (transfer fully hidden behind
compute, bounded by whichever is larger). Real hardware lands between them. Any
single-number throughput claim from this repo should quote the serialized end.

EVERY DEFAULT BELOW IS A PLACEHOLDER pending tools/hardware_probe/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

GIB = 1024 ** 3


@dataclass
class CostModel:
    """Bandwidths and fixed overheads for one target machine."""

    # -- measured-by-Phase-0 quantities, all currently ASSUMED ------------------
    ssd_read_bps: float = 6.0e9
    """SSD sequential read. The single most load-bearing assumption in the repo."""

    pcie_bps: float = 24.0e9
    """PCIe 4.0 x16 effective host-to-device. See docs/FINDINGS.md finding #3."""

    vram_bps: float = 500.0e9
    """GPU memory bandwidth. 500 GB/s is a deliberately conservative stand-in for a
    12 GB discrete RDNA3 card; gfx1100-class parts are roughly 800 GB/s and gfx1030
    roughly 512 GB/s. Confirm against the real card before quoting anything."""

    kernel_launch_s_per_layer: float = 5.0e-6
    """Fixed per-layer dispatch overhead. 43 layers x 5 us = 215 us per pass, which
    matters once the transfer terms shrink."""

    nonrouted_bytes: float = 6.9 * GIB
    """Non-routed params that stay VRAM-resident and are read every token: attention,
    shared expert, embeddings, output head. ~7 B params at Q8."""

    draft_model_bytes: float = 1.0 * GIB
    """Draft model weights, read once per drafted token. A ~1 B model at Q8."""

    pipeline_transfer: bool = False
    """Whether the SSD->host and host->VRAM stages are double-buffered.

    False charges them serially: read an expert off SSD, then send it over PCIe,
    then start the next. True charges max(ssd, pcie), which is what a real streaming
    path achieves by DMAing expert N over the bus while expert N+1 is still being
    read off disk. This is the Phase 3 "three-tier streaming path" and it is the
    single largest gain available in software: the two stages are 68% and 26% of the
    pass, so overlapping them removes the smaller one almost entirely.

    Left False by default because nothing here has been built or measured yet.
    Turning it on is a claim about an implementation that does not exist.
    """

    def pass_time_seconds(
        self,
        draft_k: int,
        vram_experts: int,
        host_experts: int,
        ssd_experts: int,
        bytes_per_expert: float,
        n_layers: int,
    ) -> Tuple[float, float]:
        """Return (serialized_seconds, ideal_overlap_seconds) for one pass.

        The counts are UNIQUE experts per tier for the pass -- deduplicated across
        the K draft candidates, which is the saving speculative verification buys.
        """
        ssd_bytes = ssd_experts * bytes_per_expert
        # Anything not already in VRAM has to cross the bus.
        pcie_bytes = (host_experts + ssd_experts) * bytes_per_expert
        # The kernels read every expert used this pass out of VRAM, plus the
        # non-routed weights once for the whole pass.
        vram_bytes = (
            (vram_experts + host_experts + ssd_experts) * bytes_per_expert
            + self.nonrouted_bytes
        )

        t_ssd = ssd_bytes / self.ssd_read_bps if self.ssd_read_bps > 0 else 0.0
        t_pcie = pcie_bytes / self.pcie_bps if self.pcie_bps > 0 else 0.0
        t_vram = vram_bytes / self.vram_bps if self.vram_bps > 0 else 0.0
        t_launch = n_layers * self.kernel_launch_s_per_layer

        # The draft model is sequential: K forward passes before verification, each
        # reading its own weights. This is the cost speculation must earn back.
        t_draft = draft_k * (self.draft_model_bytes / self.vram_bps) if self.vram_bps > 0 else 0.0

        # Double-buffered, the two transfer stages run concurrently and the slower
        # one sets the pace; serially, they add. The pipeline still has to fill and
        # drain, but at ~450 experts per pass that edge effect is negligible.
        transfer = max(t_ssd, t_pcie) if self.pipeline_transfer else (t_ssd + t_pcie)
        compute = t_vram + t_launch + t_draft

        serialized = transfer + compute
        ideal = max(transfer, compute)
        return serialized, ideal

    def cold_baseline_tok_s(self, routed_bytes_per_token: float, n_layers: int) -> float:
        """Non-speculative, empty-cache decode speed under the same cost model.

        The correct baseline for a speedup claim. Comparing against SSD-bytes-only
        cold decode (the old 3.29 tok/s) inflates the speedup by attributing the
        cache's benefit to speculation.
        """
        t_ssd = routed_bytes_per_token / self.ssd_read_bps if self.ssd_read_bps > 0 else 0.0
        t_pcie = routed_bytes_per_token / self.pcie_bps if self.pcie_bps > 0 else 0.0
        t_vram = (
            (routed_bytes_per_token + self.nonrouted_bytes) / self.vram_bps
            if self.vram_bps > 0
            else 0.0
        )
        t_launch = n_layers * self.kernel_launch_s_per_layer
        total = t_ssd + t_pcie + t_vram + t_launch
        return 1.0 / total if total > 0 else 0.0
