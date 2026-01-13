# Grad-Grad Attention: Kernel Design Notes

Goal: fast attention that **preserves grad-grad** so `torch.autograd.grad(create_graph=True)` works for meta-learning (TTT-E2E).

## Why FlashAttention fails for grad-grad
Standard FA backward uses fused, non-differentiable optimizations. The backward graph does **not** preserve intermediates in a way autograd can differentiate **again**. So meta-gradients break.

## Strategy
1) **Forward kernel** is custom Triton/CUDA (fast).
2) **Backward** is differentiable, either:
   - (A) recompute with PyTorch ops (slow but grad-grad safe), or
   - (B) custom backward kernel + custom **double-backward** kernels (fast + grad-grad safe).

We implement (A) now, and a first **Triton dV kernel** as proof-of-work.

## Dataflow

Forward:

Q,K,V  ->  QK^T (scaled)  ->  softmax  ->  P @ V  ->  O

Backward (classic):

P = softmax(QK^T)
DO = dL/dO

DV = P^T @ DO
DP = DO @ V^T
DS = d softmax(S) where S = QK^T
DQ = DS @ K
DK = DS^T @ Q

Double-backward requires differentiating each of the above w.r.t. Q/K/V/DO.

## Kernel layout (Triton forward)

We use a blocked attention kernel with an online softmax:

- Process Q in BLOCK_M rows
- Stream K,V in BLOCK_N tiles
- Maintain running (m_i, l_i) for softmax normalization

Pseudocode (per block):

m_i = -inf
l_i = 0
acc = 0
for K_tile in [0..T) step BLOCK_N:
    qk = (Q @ K_tile^T) * scale
    if causal: mask future
    m_i_new = max(m_i, max(qk))
    p = exp(qk - m_i_new)
    l_i = l_i * exp(m_i - m_i_new) + sum(p)
    acc = acc * exp(m_i - m_i_new) + p @ V_tile
    m_i = m_i_new
O = acc / l_i

## Backward options

### (A) Recompute backward (current)

Backward recomputes forward in PyTorch and calls autograd.grad:

with enable_grad():
    O = attention_math(Q,K,V)
    dQ,dK,dV = grad(O, (Q,K,V), DO, create_graph=True)

Pros:
- Correct grad-grad
- Easy to maintain

Cons:
- Slower (~2x-3x vs fused backward)

### (B) Custom backward + double-backward (future)

We need:
- Triton backward kernels for dQ, dK, dV
- Triton double-backward kernels for d^2Q, d^2K, d^2V

**Current progress**:
- Implemented a correctness-first Triton dV kernel (`bwd_mode='dv_only'`).
- dQ/dK still recompute (grad-grad safe).

## Implementation scaffolding

- `ttt_kernels/triton_attention.py` contains:
  - Triton forward kernel
  - Backward recompute (grad-grad safe)
  - Optional Triton dV path (`bwd_mode='dv_only'`)

- `ttt_kernels/backward_stubs.py`:
  - Triton dV kernel
  - dq/dk + double-backward stubs

## Next steps checklist

[ ] Validate dV vs PyTorch (parity tests)
[ ] Triton dQ/dK kernels
[ ] Double-backward for dV
[ ] Double-backward for dQ/dK
[ ] Autotune configs (block sizes)
[ ] Benchmark vs FlashAttention
