"""Trace representation and synthetic trace generation for MoE expert routing.

A trace is a sequence of tokens, where each token requires expert accesses across
layers (e.g., 43 layers x 6 experts/layer = 258 accesses per token for DeepSeek-V4 Flash).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple
import random


@dataclass(frozen=True)
class ExpertAccess:
    """A single expert access within a model layer for a specific token."""
    token_idx: int
    layer_idx: int
    expert_idx: int

    @property
    def key(self) -> Tuple[int, int]:
        """Unique identifier for the expert: (layer_idx, expert_idx)."""
        return (self.layer_idx, self.expert_idx)


@dataclass
class TokenAccess:
    """All expert accesses for a single decoded token."""
    token_idx: int
    # List of (layer_idx, expert_idx)
    accesses: List[Tuple[int, int]] = field(default_factory=list)


class Trace:
    """Sequence of expert accesses representing an inference run."""

    def __init__(self, n_layers: int = 43, n_experts: int = 256, n_expert_used: int = 6):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_expert_used = n_expert_used
        self.tokens: List[TokenAccess] = []

    def add_token(self, token_idx: int, accesses: List[Tuple[int, int]]) -> None:
        self.tokens.append(TokenAccess(token_idx=token_idx, accesses=accesses))

    def total_tokens(self) -> int:
        return len(self.tokens)

    def total_accesses(self) -> int:
        return sum(len(t.accesses) for t in self.tokens)

    def iter_accesses(self) -> Iterator[ExpertAccess]:
        for token in self.tokens:
            for layer_idx, expert_idx in token.accesses:
                yield ExpertAccess(
                    token_idx=token.token_idx,
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                )

    def save_jsonl(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for token in self.tokens:
                f.write(json.dumps({
                    "token_idx": token.token_idx,
                    "accesses": token.accesses
                }) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path | str, n_layers: int = 43, n_experts: int = 256, n_expert_used: int = 6) -> Trace:
        trace = cls(n_layers=n_layers, n_experts=n_experts, n_expert_used=n_expert_used)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                trace.add_token(
                    token_idx=data["token_idx"],
                    accesses=[(a[0], a[1]) for a in data["accesses"]]
                )
        return trace


def generate_synthetic_trace(
    n_tokens: int = 100,
    n_layers: int = 43,
    n_experts: int = 256,
    n_expert_used: int = 6,
    distribution: str = "zipf",
    zipf_s: float = 1.2,
    seed: int = 42,
) -> Trace:
    """Generate a synthetic routing trace.

    Distributions:
    - 'zipf': Power-law routing skew per layer (realistic for MoE where certain experts are hot).
    - 'uniform': Equal probability for all experts.
    - 'repeating': High locality phase repeating patterns.
    """
    rng = random.Random(seed)
    trace = Trace(n_layers=n_layers, n_experts=n_experts, n_expert_used=n_expert_used)

    if distribution == "zipf":
        # Precompute CDF for Zipf distribution per expert
        weights = [1.0 / (i ** zipf_s) for i in range(1, n_experts + 1)]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
    else:
        probs = [1.0 / n_experts] * n_experts

    expert_pool = list(range(n_experts))

    for tok_i in range(n_tokens):
        accesses: List[Tuple[int, int]] = []
        for l_i in range(n_layers):
            if distribution == "repeating" and tok_i % 10 < 7:
                # 70% of time sample from hot top-10 per layer
                selected = rng.sample(range(min(12, n_experts)), n_expert_used)
            elif distribution == "zipf":
                # Sample n_expert_used unique experts according to weights
                selected = []
                while len(selected) < n_expert_used:
                    choice = rng.choices(expert_pool, weights=probs, k=1)[0]
                    if choice not in selected:
                        selected.append(choice)
            else:
                # Uniform
                selected = rng.sample(expert_pool, n_expert_used)

            for exp in selected:
                accesses.append((l_i, exp))

        trace.add_token(tok_i, accesses)

    return trace
