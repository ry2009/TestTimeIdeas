import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttt_kernels.triton_attention import triton_attention


def _math_attention(q, k, v, causal, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        t = scores.size(-1)
        mask = torch.triu(torch.ones((t, t), device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    b, h, t, d = 1, 1, 64, 64
    dtype = torch.float16 if device == 'cuda' else torch.float32
    scale = 1.0 / (d ** 0.5)

    q = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)

    # Forward
    out_ref = _math_attention(q, k, v, causal=True, scale=scale)
    out_tri = triton_attention(q, k, v, causal=True, bwd_mode='save_p_triton_full')
    f_err = (out_ref - out_tri).abs().max().item()
    f_mean = (out_ref - out_tri).abs().mean().item()

    # First order grads
    loss_ref = out_ref.float().mean()
    loss_tri = out_tri.float().mean()
    grads_ref = torch.autograd.grad(loss_ref, (q, k, v), create_graph=True)
    grads_tri = torch.autograd.grad(loss_tri, (q, k, v), create_graph=True)
    g_err = [ (grads_ref[i] - grads_tri[i]).abs().max().item() for i in range(3) ]

    # Second order grads
    s_ref = sum([g.pow(2).mean() for g in grads_ref])
    s_tri = sum([g.pow(2).mean() for g in grads_tri])
    gg_ref = torch.autograd.grad(s_ref, (q, k, v))
    gg_tri = torch.autograd.grad(s_tri, (q, k, v))
    gg_err = [ (gg_ref[i] - gg_tri[i]).abs().max().item() for i in range(3) ]

    print("== gradgrad compare save_p_triton_full vs math ==")
    print(f"forward max abs err: {f_err:.4e}, mean abs err: {f_mean:.4e}")
    print(f"grad max abs err (q,k,v): {g_err}")
    print(f"gradgrad max abs err (q,k,v): {gg_err}")


if __name__ == "__main__":
    main()
