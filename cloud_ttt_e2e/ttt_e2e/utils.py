import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def select_named_parameters(model: torch.nn.Module, include: Optional[str] = None) -> Dict[str, torch.nn.Parameter]:
    params = dict(model.named_parameters())
    if include is None:
        return params
    return {k: v for k, v in params.items() if include in k}


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # logits: [B, T, V], targets: [B, T]
    vocab = logits.size(-1)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    )
    if mask is None:
        return loss.mean()
    mask = mask.reshape(-1).float()
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def make_window_mask(seq_len: int, window_size: Optional[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Returns additive mask with 0 for allowed and -inf for masked.
    i = torch.arange(seq_len, device=device)
    j = torch.arange(seq_len, device=device)
    ii = i[:, None]
    jj = j[None, :]
    causal = jj > ii
    if window_size is None:
        mask = causal
    else:
        too_old = (ii - jj) >= window_size
        mask = causal | too_old
    out = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    out = out.masked_fill(mask, float("-inf"))
    return out


@dataclass
class TrainConfig:
    vocab_size: int = 4096
    max_seq_len: int = 8192
    context_len: int = 4096
    query_len: int = 1024
    window_size: Optional[int] = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.0
    # number of suffix layers to update in inner loop
    suffix_layers: int = -1
    # use dual MLP (fast + slow) in suffix blocks
    dual_mlp: bool = True
    # chunk size for inner-loop scan
    chunk_size: int = 1024
    # fraction of query keys forced to be from the early context region
    far_frac: float = 0.0
    # number of unique keys/values per sequence
    num_keys: int = 256
    batch_size: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.0
    steps: int = 1000
    inner_steps: int = 1
    inner_lr: float = 1e-2
    # comma-separated substrings to select inner-loop parameters
    ttt_param_filter: Optional[str] = None
    device: str = "cuda"
    seed: int = 1337
