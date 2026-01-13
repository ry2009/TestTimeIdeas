import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttt_kernels.triton_attention import triton_attention


def test_gradgrad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    b, h, t, d = 2, 2, 64, 64
    q = torch.randn(b, h, t, d, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(b, h, t, d, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(b, h, t, d, device=device, dtype=torch.float32, requires_grad=True)

    out = triton_attention(q, k, v, causal=True)
    loss = out.float().mean()
    grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)
    # second-order
    s = sum([g.pow(2).mean() for g in grads])
    torch.autograd.grad(s, (q, k, v))
