import math
import torch
from torch import nn
import torch.nn.functional as F

from .utils import make_window_mask


def _attention(q, k, v, attn_mask):
    # q,k,v: [B,H,T,HD] or [B,H,Q,HD], attn_mask broadcastable to [B,H,Q,K]
    d = q.size(-1)
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


class GradGradAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask):
        ctx.save_for_backward(q, k, v, attn_mask)
        with torch.no_grad():
            out = _attention(q, k, v, attn_mask)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, attn_mask = ctx.saved_tensors
        need_higher = grad_out.requires_grad
        if need_higher:
            q_ = q
            k_ = k
            v_ = v
            q_.requires_grad_(True)
            k_.requires_grad_(True)
            v_.requires_grad_(True)
        else:
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _attention(q_, k_, v_, attn_mask)
            grads = torch.autograd.grad(
                out,
                (q_, k_, v_),
                grad_out,
                create_graph=need_higher,
                retain_graph=need_higher,
            )
        return grads[0], grads[1], grads[2], None


def _local_window_mask(q_start, q_end, k_start, k_end, window_size, device, dtype):
    q_idx = torch.arange(q_start, q_end, device=device)
    k_idx = torch.arange(k_start, k_end, device=device)
    qi = q_idx[:, None]
    ki = k_idx[None, :]
    causal = ki > qi
    if window_size is None:
        mask = causal
    else:
        too_old = (qi - ki) >= window_size
        mask = causal | too_old
    out = torch.zeros((q_end - q_start, k_end - k_start), device=device, dtype=dtype)
    out = out.masked_fill(mask, float("-inf"))
    return out


class WindowedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int | None, use_gradgrad: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.use_gradgrad = use_gradgrad

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        b, t, d = x.shape
        x = x.view(b, t, self.n_heads, self.head_dim)
        return x.transpose(1, 2)  # [B,H,T,HD]

    def _merge_heads(self, x):
        b, h, t, d = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(b, t, h * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.window_size is None or t <= self.window_size:
            if self.use_gradgrad:
                attn_mask = make_window_mask(t, self.window_size, x.device, x.dtype).view(1, 1, t, t)
                out = GradGradAttentionFn.apply(q, k, v, attn_mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = self._merge_heads(out)
            return self.out_proj(out)

        outputs = []
        chunk = self.window_size
        for start in range(0, t, chunk):
            end = min(t, start + chunk)
            kv_start = max(0, end - self.window_size)
            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, kv_start:end, :]
            v_chunk = v[:, :, kv_start:end, :]
            if self.use_gradgrad:
                mask = _local_window_mask(start, end, kv_start, end, self.window_size, x.device, x.dtype)
                mask = mask.view(1, 1, end - start, end - kv_start)
                out_chunk = GradGradAttentionFn.apply(q_chunk, k_chunk, v_chunk, mask)
            else:
                mask = _local_window_mask(start, end, kv_start, end, self.window_size, x.device, x.dtype)
                mask = mask.view(1, 1, end - start, end - kv_start)
                out_chunk = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, attn_mask=mask)
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=2)
        out = self._merge_heads(out)
        return self.out_proj(out)
