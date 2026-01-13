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

if _TRITON_AVAILABLE:
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
                qk = tl.dot(q, k) * scale
                qk = qk.to(tl.float32)
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
            qk_tile = tl.dot(q, k_tile) * scale
            qk_tile = qk_tile.to(tl.float32)
            if causal:
                mask = offs_m[:, None] < offs_n[None, :]
                qk_tile = tl.where(mask, -float('inf'), qk_tile)

            p_tile = tl.exp(qk_tile - m_i[:, None]) / l_i[:, None]
            dv += tl.dot(tl.trans(p_tile).to(tl.float16), do)

        dv_ptrs = dv_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
        tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head))


    def triton_attn_bwd_dv(q, k, v, do, *, causal: bool, scale: float):
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

    @triton.jit
    def _attn_bwd_dv_from_p_kernel(
        p_ptr, do_ptr, dv_ptr,
        stride_pb, stride_pm, stride_pn,
        stride_dob, stride_dom, stride_dok,
        stride_dvb, stride_dvn, stride_dvk,
        n_ctx, d_head,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_bh = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

        for start_m in range(0, n_ctx, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            p_ptrs = p_ptr + pid_bh * stride_pb + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
            do_ptrs = do_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
            p = tl.load(p_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_n[None, :] < n_ctx), other=0.0)
            do = tl.load(do_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
            acc += tl.dot(tl.trans(p).to(tl.float16), do)

        dv_ptrs = dv_ptr + pid_bh * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
        tl.store(dv_ptrs, acc, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head))


    def triton_attn_bwd_dv_from_p(p, do):
        b, h, t, d = do.shape
        ph = p.reshape(b * h, t, t)
        doh = do.reshape(b * h, t, d)
        dvh = torch.empty_like(doh)

        stride_pb, stride_pm, stride_pn = ph.stride()
        stride_dob, stride_dom, stride_dok = doh.stride()
        stride_dvb, stride_dvn, stride_dvk = dvh.stride()

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = 64 if d <= 64 else 128
        grid = (triton.cdiv(t, BLOCK_N), triton.cdiv(d, BLOCK_D), b * h)

        _attn_bwd_dv_from_p_kernel[grid](
            ph, doh, dvh,
            stride_pb, stride_pm, stride_pn,
            stride_dob, stride_dom, stride_dok,
            stride_dvb, stride_dvn, stride_dvk,
            t, d,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )
        return dvh.reshape(b, h, t, d)

    @triton.jit
    def _attn_rowsum_kernel(
        p_ptr, do_ptr, v_ptr, rowsum_ptr,
        stride_pb, stride_pm, stride_pn,
        stride_dob, stride_dom, stride_dok,
        stride_vb, stride_vn, stride_vk,
        stride_rsb, stride_rsm,
        n_ctx, d_head,
        causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        do_ptrs = do_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(do_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)

        rowsum = tl.zeros((BLOCK_M,), tl.float32)
        for start_n in range(0, n_ctx, BLOCK_N):
            block_active = 1.0
            if causal:
                max_m = pid_m * BLOCK_M + BLOCK_M - 1
                block_active = tl.where(max_m >= start_n, 1.0, 0.0)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            p_ptrs = p_ptr + pid_bh * stride_pb + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
            p = tl.load(p_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_n[None, :] < n_ctx), other=0.0)
            v_ptrs = v_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)

            dp = tl.dot(do, tl.trans(v))  # (M, N)
            dp = dp.to(tl.float32)
            rowsum += block_active * tl.sum(dp * p.to(tl.float32), axis=1)

        rs_ptrs = rowsum_ptr + pid_bh * stride_rsb + offs_m * stride_rsm
        tl.store(rs_ptrs, rowsum, mask=(offs_m < n_ctx))


    def triton_attn_rowsum_from_p(p, do, v, *, causal: bool):
        b, h, t, d = do.shape
        ph = p.reshape(b * h, t, t)
        doh = do.reshape(b * h, t, d)
        vh = v.reshape(b * h, t, d)
        rowsum = torch.empty((b * h, t), device=do.device, dtype=torch.float32)

        stride_pb, stride_pm, stride_pn = ph.stride()
        stride_dob, stride_dom, stride_dok = doh.stride()
        stride_vb, stride_vn, stride_vk = vh.stride()
        stride_rsb, stride_rsm = rowsum.stride()

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = 64 if d <= 64 else 128
        grid = (triton.cdiv(t, BLOCK_M), b * h)

        _attn_rowsum_kernel[grid](
            ph, doh, vh, rowsum,
            stride_pb, stride_pm, stride_pn,
            stride_dob, stride_dom, stride_dok,
            stride_vb, stride_vn, stride_vk,
            stride_rsb, stride_rsm,
            t, d,
            causal=causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )
        return rowsum.reshape(b, h, t)


    @triton.jit
    def _attn_bwd_dq_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, p_ptr, rowsum_ptr, dq_ptr,
        stride_qb, stride_qm, stride_qk,
        stride_kb, stride_kn, stride_kk,
        stride_vb, stride_vn, stride_vk,
        stride_dob, stride_dom, stride_dok,
        stride_pb, stride_pm, stride_pn,
        stride_rsb, stride_rsm,
        stride_dqb, stride_dqm, stride_dqk,
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
        do_ptrs = do_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
        do = tl.load(do_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)

        rs_ptrs = rowsum_ptr + pid_bh * stride_rsb + offs_m * stride_rsm
        row_sum = tl.load(rs_ptrs, mask=(offs_m < n_ctx), other=0.0)

        dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for start_n in range(0, n_ctx, BLOCK_N):
            block_active = 1.0
            if causal:
                max_m = pid_m * BLOCK_M + BLOCK_M - 1
                block_active = tl.where(max_m >= start_n, 1.0, 0.0)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = k_ptr + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
            v_ptrs = v_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
            p_ptrs = p_ptr + pid_bh * stride_pb + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
            p = tl.load(p_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_n[None, :] < n_ctx), other=0.0)

            dp = tl.dot(do, tl.trans(v))  # (M, N)
            ds = (dp.to(tl.float32) - row_sum[:, None]) * p.to(tl.float32)
            dq += block_active * tl.dot(ds.to(tl.float16), k)

        dq = dq * scale
        dq_ptrs = dq_ptr + pid_bh * stride_dqb + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
        tl.store(dq_ptrs, dq, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head))


    @triton.jit
    def _attn_bwd_dk_kernel(
        q_ptr, v_ptr, do_ptr, p_ptr, rowsum_ptr, dk_ptr,
        stride_qb, stride_qm, stride_qk,
        stride_vb, stride_vn, stride_vk,
        stride_dob, stride_dom, stride_dok,
        stride_pb, stride_pm, stride_pn,
        stride_rsb, stride_rsm,
        stride_dkb, stride_dkn, stride_dkk,
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

        dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

        for start_m in range(0, n_ctx, BLOCK_M):
            block_active = 1.0
            if causal:
                block_active = tl.where((start_m + BLOCK_M - 1) >= (pid_n * BLOCK_N), 1.0, 0.0)
            offs_m = start_m + tl.arange(0, BLOCK_M)
            q_ptrs = q_ptr + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
            do_ptrs = do_ptr + pid_bh * stride_dob + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok
            do = tl.load(do_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
            rs_ptrs = rowsum_ptr + pid_bh * stride_rsb + offs_m * stride_rsm
            row_sum = tl.load(rs_ptrs, mask=(offs_m < n_ctx), other=0.0)

            v_ptrs = v_ptr + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head), other=0.0)
            p_ptrs = p_ptr + pid_bh * stride_pb + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
            p = tl.load(p_ptrs, mask=(offs_m[:, None] < n_ctx) & (offs_n[None, :] < n_ctx), other=0.0)

            dp = tl.dot(do, tl.trans(v))  # (M, N)
            ds = (dp.to(tl.float32) - row_sum[:, None]) * p.to(tl.float32)
            dk += block_active * tl.dot(tl.trans(ds).to(tl.float16), q)

        dk = dk * scale
        dk_ptrs = dk_ptr + pid_bh * stride_dkb + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
        tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < n_ctx) & (offs_d[None, :] < d_head))


    def triton_attn_bwd_dqdk_from_p(q, k, v, do, p, rowsum, *, causal: bool, scale: float):
        b, h, t, d = q.shape
        qh = q.reshape(b * h, t, d)
        kh = k.reshape(b * h, t, d)
        vh = v.reshape(b * h, t, d)
        doh = do.reshape(b * h, t, d)
        ph = p.reshape(b * h, t, t)
        rsh = rowsum.reshape(b * h, t)
        dqh = torch.empty_like(qh)
        dkh = torch.empty_like(kh)

        stride_qb, stride_qm, stride_qk = qh.stride()
        stride_kb, stride_kn, stride_kk = kh.stride()
        stride_vb, stride_vn, stride_vk = vh.stride()
        stride_dob, stride_dom, stride_dok = doh.stride()
        stride_pb, stride_pm, stride_pn = ph.stride()
        stride_rsb, stride_rsm = rsh.stride()
        stride_dqb, stride_dqm, stride_dqk = dqh.stride()
        stride_dkb, stride_dkn, stride_dkk = dkh.stride()

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = 64 if d <= 64 else 128

        grid_q = (triton.cdiv(t, BLOCK_M), b * h)
        _attn_bwd_dq_kernel[grid_q](
            qh, kh, vh, doh, ph, rsh, dqh,
            stride_qb, stride_qm, stride_qk,
            stride_kb, stride_kn, stride_kk,
            stride_vb, stride_vn, stride_vk,
            stride_dob, stride_dom, stride_dok,
            stride_pb, stride_pm, stride_pn,
            stride_rsb, stride_rsm,
            stride_dqb, stride_dqm, stride_dqk,
            t, d,
            scale,
            causal=causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )

        grid_k = (triton.cdiv(t, BLOCK_N), b * h)
        _attn_bwd_dk_kernel[grid_k](
            qh, vh, doh, ph, rsh, dkh,
            stride_qb, stride_qm, stride_qk,
            stride_vb, stride_vn, stride_vk,
            stride_dob, stride_dom, stride_dok,
            stride_pb, stride_pm, stride_pn,
            stride_rsb, stride_rsm,
            stride_dkb, stride_dkn, stride_dkk,
            t, d,
            scale,
            causal=causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )

        return dqh.reshape(b, h, t, d), dkh.reshape(b, h, t, d)
else:
    def triton_attn_bwd_dv(*args, **kwargs):
        raise RuntimeError('Triton is not available')
    def triton_attn_bwd_dv_from_p(*args, **kwargs):
        raise RuntimeError('Triton is not available')
    def triton_attn_rowsum_from_p(*args, **kwargs):
        raise RuntimeError('Triton is not available')
    def triton_attn_bwd_dqdk_from_p(*args, **kwargs):
        raise RuntimeError('Triton is not available')


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
