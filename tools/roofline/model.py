"""Model geometry and quantization sizing for the DeepSeek-V4 family on DwarfStar.

Every architectural constant here is transcribed from upstream antirez/ds4 at
commit b0309611041655f4e45671cfd9c9886aff161406, file ds4.c, structs
DS4_SHAPE_FLASH (ds4.c:541) and DS4_SHAPE_PRO (ds4.c:578). Nothing is guessed.

Bits-per-weight is *derived* from the GGUF block structs in ds4_rocm.cu:60-84
rather than hardcoded, so these numbers stay honest if a block layout changes.
"""

from dataclasses import dataclass

QK_K = 256  # ds4_rocm.cu:57  -- #define CUDA_QK_K 256


def _bpw(block_bytes: int, weights_per_block: int) -> float:
    """Bits per weight for a GGUF block type."""
    return 8.0 * block_bytes / weights_per_block


# Block sizes transcribed from the structs in ds4_rocm.cu:60-84.
#   cuda_block_q2_K:    scales[QK_K/16] + qs[QK_K/4] + d(u16) + dmin(u16)
#   cuda_block_iq2_xxs: d(u16) + qs[QK_K/8](u16)
#   cuda_block_q4_K:    d(u16) + dmin(u16) + scales[12] + qs[QK_K/2]
Q2_K_BLOCK_BYTES = (QK_K // 16) + (QK_K // 4) + 2 + 2   # 16 + 64 + 2 + 2 = 84
IQ2_XXS_BLOCK_BYTES = 2 + 2 * (QK_K // 8)               # 2 + 64         = 66
Q4_K_BLOCK_BYTES = 2 + 2 + 12 + (QK_K // 2)             # 2+2+12+128     = 144

BPW = {
    "IQ2_XXS": _bpw(IQ2_XXS_BLOCK_BYTES, QK_K),   # 2.0625
    "Q2_K": _bpw(Q2_K_BLOCK_BYTES, QK_K),         # 2.625
    "Q4_K": _bpw(Q4_K_BLOCK_BYTES, QK_K),         # 4.5
    "Q8_0": _bpw(2 + 32, 32),                     # 8.5
    "F16": 16.0,
}


@dataclass(frozen=True)
class Shape:
    """Transcribed from the ds4_shape struct in ds4.c."""

    name: str
    n_layer: int
    n_embd: int
    n_vocab: int
    n_expert: int
    n_expert_used: int
    n_expert_shared: int
    n_ff_exp: int
    total_params: int   # MODEL_CARD.md, used to cross-check our derivation
    active_params: int

    @property
    def params_per_routed_expert(self) -> int:
        """gate + up + down, each (n_embd x n_ff_exp)."""
        return 3 * self.n_embd * self.n_ff_exp

    @property
    def total_routed_experts(self) -> int:
        return self.n_layer * self.n_expert

    @property
    def total_routed_params(self) -> int:
        return self.total_routed_experts * self.params_per_routed_expert

    @property
    def nonrouted_params(self) -> int:
        """Everything that is not a routed expert: attention, shared experts,
        embeddings, output, norms. Derived by subtraction from the published
        total, which is more trustworthy than enumerating every tensor.
        """
        return self.total_params - self.total_routed_params

    def bytes_per_routed_expert(
        self, gate_up: str = "IQ2_XXS", down: str = "Q2_K"
    ) -> float:
        """Bytes on disk for one complete routed expert (gate + up + down).

        THE governing number of the whole streaming design: it is the atomic
        unit of SSD read, PCIe transfer, and cache occupancy.
        """
        elems = self.n_embd * self.n_ff_exp
        return elems * (2 * BPW[gate_up] + BPW[down]) / 8.0

    def routed_bytes_per_token(
        self, gate_up: str = "IQ2_XXS", down: str = "Q2_K"
    ) -> float:
        """Routed-expert bytes touched to decode one token with a cold cache:
        n_layer * n_expert_used * bytes_per_expert.
        """
        return (
            self.n_layer
            * self.n_expert_used
            * self.bytes_per_routed_expert(gate_up, down)
        )

    def expert_loads_per_token(self) -> int:
        return self.n_layer * self.n_expert_used


FLASH = Shape(
    name="DeepSeek V4 Flash",
    n_layer=43,
    n_embd=4096,
    n_vocab=129280,
    n_expert=256,
    n_expert_used=6,
    n_expert_shared=1,
    n_ff_exp=2048,
    total_params=284_000_000_000,
    active_params=13_000_000_000,
)

PRO = Shape(
    name="DeepSeek V4 Pro",
    n_layer=61,
    n_embd=7168,
    n_vocab=129280,
    n_expert=384,
    n_expert_used=6,
    n_expert_shared=1,
    n_ff_exp=3072,
    total_params=1_600_000_000_000,
    active_params=49_000_000_000,
)
