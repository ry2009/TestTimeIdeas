from dataclasses import dataclass
from typing import Optional

import torch

from .attention import WindowedSelfAttention


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    max_seq_len: int = 8192
    window_size: Optional[int] = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.0
    use_gradgrad_attention: bool = True
    # Number of suffix layers to update in inner loop; prefix uses remaining layers
    suffix_layers: int = -1
    # Use dual MLP (fast + slow) in suffix blocks
    dual_mlp: bool = True


class PrefixBlock(torch.nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(cfg.d_model)
        self.attn = WindowedSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            window_size=cfg.window_size,
            use_gradgrad=cfg.use_gradgrad_attention,
        )
        self.ln2 = torch.nn.LayerNorm(cfg.d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(cfg.d_model, cfg.d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class SuffixBlock(torch.nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(cfg.d_model)
        self.attn = WindowedSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            window_size=cfg.window_size,
            use_gradgrad=cfg.use_gradgrad_attention,
        )
        self.ln2 = torch.nn.LayerNorm(cfg.d_model)
        self.mlp_fast = torch.nn.Sequential(
            torch.nn.Linear(cfg.d_model, cfg.d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.mlp_slow = None
        if cfg.dual_mlp:
            self.mlp_slow = torch.nn.Sequential(
                torch.nn.Linear(cfg.d_model, cfg.d_ff),
                torch.nn.GELU(),
                torch.nn.Linear(cfg.d_ff, cfg.d_model),
            )
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        h = self.ln2(x)
        out = self.mlp_fast(h)
        if self.mlp_slow is not None:
            out = out + self.mlp_slow(h)
        x = x + self.dropout(out)
        return x


class TTTModel(torch.nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = torch.nn.Embedding(cfg.max_seq_len, cfg.d_model)

        suffix_layers = cfg.suffix_layers
        if suffix_layers < 0 or suffix_layers > cfg.n_layers:
            suffix_layers = cfg.n_layers
        self.suffix_layers = suffix_layers
        self.prefix_layers = cfg.n_layers - suffix_layers

        prefix_blocks = [PrefixBlock(cfg) for _ in range(self.prefix_layers)]
        suffix_blocks = [SuffixBlock(cfg) for _ in range(self.suffix_layers)]
        self.prefix_blocks = torch.nn.ModuleList(prefix_blocks)
        self.suffix_blocks = torch.nn.ModuleList(suffix_blocks)

        self.ln_f = torch.nn.LayerNorm(cfg.d_model)
        self.head = torch.nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device)
        return self.embed(input_ids) + self.pos(pos)[None, :, :]

    def prefix_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self._embed(input_ids)
        for block in self.prefix_blocks:
            x = block(x)
        return x

    def suffix_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden
        for block in self.suffix_blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        *,
        mode: str = 'full',
        prefix_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mode == 'prefix':
            if input_ids is None:
                raise ValueError('input_ids required for prefix mode')
            return self.prefix_forward(input_ids)
        if mode == 'suffix':
            if prefix_hidden is None:
                raise ValueError('prefix_hidden required for suffix mode')
            return self.suffix_forward(prefix_hidden)
        # full
        if input_ids is None:
            raise ValueError('input_ids required for full mode')
        hidden = self.prefix_forward(input_ids)
        return self.suffix_forward(hidden)
