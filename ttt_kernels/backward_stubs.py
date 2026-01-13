"""Backward / double-backward stubs for grad-grad attention.

We start with a correctness-first Triton dV kernel (slow but real),
and keep dq/dk as TODO. This is the first step toward full custom
backward + double-backward.
"""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


@triton.jit
def _attn_bwd_dv_kernel(
    q_ptr, k_ptr, do_ptr, dv_ptr,
    stride_qb, stride_qm, stride_qk,
    stride_kb, stride_kn, stride_kk,
    stride_dob, stride_dom, stride_dok,
    stride_dvb, stride_dvn, stride_dvk,
    n_ctx, d_head,
    scale,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for start_m in range(0, n_ctx, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)

        # Load Q and dO for this M block
        q_ptrs = q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = do_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
        do = tl.load(do_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
        do = do.to(tl.float32)

        # Pass 1: compute softmax stats (m_i, l_i) across all N
        m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        for start_n_all in range(0, n_ctx, BLOCK_N):
            offs_n_all = start_n_all + tl.arange(0, BLOCK_N)
            k_ptrs = k_ptr + pid_bh * stride_kb + offs_n_all[None, :] * stride_kn + offs_d[:, None] * stride_kk
            k = tl.load(k_ptrs, mask=(offs_n_all[None, :] < n_ctx) & (offs_d[:, None] < d_head), other=0.0)
            k_t = tl.trans(k)
            qk = tl.sum(q[:, None, :] * k_t[None, :, :], axis=2) * scale
            if causal:
                mask = offs_m[:, None] < offs_n_all[None, :]
                qk = tl.where(mask, -float('inf'), qk)
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_i_new[:, None])
            l_i = l_i * tl.exp(m_i - m_i_new) + tl.sum(p, axis=1)
            m_i = m_i_new

        # Pass 2: compute p for current N tile only, accumulate dV
        k_ptrs_tile = k_ptr + pid_bh * stride_kb + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k_tile = tl.load(k_ptrs_tile, mask=(offs_n[None, :] < n_ctx) & (offs_d[:, None] < d_head), other=0.0)
        k_tile_t = tl.trans(k_tile)
        qk_tile = tl.sum(q[:, None, :] * k_tile_t[None, :, :], axis=2) * scale
        if causal:
            mask = offs_m[:, None] < offs_n[None, :]
            qk_tile = tl.where(mask, -float('inf'), qk_tile)

        p_tile = tl.exp(qk_tile - m_i[:, None]) / l_i[:, None]
        dv += tl.sum(tl.trans(p_tile)[:, :, None] * do[None, :, :], axis=1)

    dv_ptrs = dv_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
    tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head))


def triton_attn_bwd_dv(q, k, v, do, *, causal: bool, scale: float):
    if not _TRITON_AVAILABLE:
        raise RuntimeError('Triton is not available')
    b, h, t, d = q.shape
    if d > 128:
        raise RuntimeError('d_head > 128 not supported in dv kernel')

    qh = q.reshape(b * h, t, d)
    kh = k.reshape(b * h, t, d)
    doh = do.reshape(b * h, t, d)
    dvh = torch.empty_like(doh)

    stride_qb, stride_qm, stride_qk = qh.stride()
    stride_kb, stride_kn, stride_kk = kh.stride()
    stride_dob, stride_dom, stride_dok = doh.stride()
    stride_dvb, stride_dvn, stride_dvk = dvh.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 64 if d <= 64 else 128
    grid = (triton.cdiv(t, BLOCK_N), b * h)

    _attn_bwd_dv_kernel[grid](
        qh, kh, doh, dvh,
        stride_qb, stride_qm, stride_qk,
        stride_kb, stride_kn, stride_kk,
        stride_dob, stride_dom, stride_dok,
        stride_dvb, stride_dvn, stride_dvk,
        t, d,
        scale,
        causal=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return dvh.reshape(b, h, t, d)


# TODO: dq/dk kernels

def triton_attn_bwd_dq(q, k, v, do, *, causal: bool, scale: float):
    raise NotImplementedError('dq Triton kernel not implemented yet')


def triton_attn_bwd_dk(q, k, v, do, *, causal: bool, scale: float):
    raise NotImplementedError('dk Triton kernel not implemented yet')


# Double-backward stubs

def triton_attn_bwd2_dq(*args, **kwargs):
    raise NotImplementedError('double-backward dq not implemented yet')


def triton_attn_bwd2_dk(*args, **kwargs):
    raise NotImplementedError('double-backward dk not implemented yet')


def triton_attn_bwd2_dv(*args, **kwargs):
    raise NotImplementedError('double-backward dv not implemented yet')
