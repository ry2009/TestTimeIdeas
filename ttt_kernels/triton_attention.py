import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from . import backward_stubs


def _attention_math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        t = scores.size(-1)
        mask = torch.triu(torch.ones((t, t), device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qm, stride_qk,
    stride_kb, stride_kn, stride_kk,
    stride_vb, stride_vn, stride_vk,
    stride_ob, stride_om, stride_ok,
    n_ctx, d_head,
    scale,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)

    m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, n_ctx, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + pid_bh * stride_kb + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=(offs_n[None, :] < n_ctx) & (offs_d[:, None] < d_head), other=0.0)

        # Manual matmul to avoid dot layout issues
        k_t = tl.trans(k)  # (BLOCK_N, BLOCK_D)
        qk = tl.sum(q[:, None, :] * k_t[None, :, :], axis=2) * scale

        if causal:
            mask = offs_m[:, None] < offs_n[None, :]
            qk = tl.where(mask, -float('inf'), qk)

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_qk = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * tl.exp(m_i - m_i_new) + tl.sum(exp_qk, axis=1)

        v_ptrs = v_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
        v = v.to(tl.float32)

        acc = acc * tl.exp(m_i - m_i_new)[:, None] + tl.sum(exp_qk[:, :, None] * v[None, :, :], axis=1)
        m_i = m_i_new

    out = acc / l_i[:, None]
    o_ptrs = o_ptr + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, out, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head))


class TritonAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, scale: Optional[float] = None, bwd_mode: str = 'recompute'):
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))
        ctx.causal = causal
        ctx.scale = scale
        ctx.bwd_mode = bwd_mode
        ctx.save_for_backward(q, k, v)

        if not _TRITON_AVAILABLE:
            return _attention_math(q, k, v, causal, scale)

        b, h, t, d = q.shape
        if d > 128:
            return _attention_math(q, k, v, causal, scale)

        qh = q.reshape(b * h, t, d)
        kh = k.reshape(b * h, t, d)
        vh = v.reshape(b * h, t, d)
        out = torch.empty_like(qh)

        stride_qb, stride_qm, stride_qk = qh.stride()
        stride_kb, stride_kn, stride_kk = kh.stride()
        stride_vb, stride_vn, stride_vk = vh.stride()
        stride_ob, stride_om, stride_ok = out.stride()

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = 64 if d <= 64 else 128
        grid = (triton.cdiv(t, BLOCK_M), b * h)

        _attn_fwd_kernel[grid](
            qh, kh, vh, out,
            stride_qb, stride_qm, stride_qk,
            stride_kb, stride_kn, stride_kk,
            stride_vb, stride_vn, stride_vk,
            stride_ob, stride_om, stride_ok,
            t, d,
            scale,
            causal=causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )
        return out.reshape(b, h, t, d)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v = ctx.saved_tensors
        causal = ctx.causal
        scale = ctx.scale

        if ctx.bwd_mode == 'custom':
            dq = backward_stubs.triton_attn_bwd_dq(q, k, v, grad_out, causal=causal, scale=scale)
            dk = backward_stubs.triton_attn_bwd_dk(q, k, v, grad_out, causal=causal, scale=scale)
            dv = backward_stubs.triton_attn_bwd_dv(q, k, v, grad_out, causal=causal, scale=scale)
            return dq, dk, dv, None, None, None

        if ctx.bwd_mode == 'dv_only':
            dv = backward_stubs.triton_attn_bwd_dv(q, k, v, grad_out, causal=causal, scale=scale)
            with torch.enable_grad():
                out = _attention_math(q, k, v, causal, scale)
                dq, dk, _ = torch.autograd.grad(out, (q, k, v), grad_out, create_graph=torch.is_grad_enabled())
            return dq, dk, dv, None, None, None

        # Default: differentiable recompute for grad-grad safety
        with torch.enable_grad():
            out = _attention_math(q, k, v, causal, scale)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_out, create_graph=torch.is_grad_enabled())
        return dq, dk, dv, None, None, None


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False, scale: Optional[float] = None, bwd_mode: str = 'recompute') -> torch.Tensor:
    return TritonAttentionFn.apply(q, k, v, causal, scale, bwd_mode)
