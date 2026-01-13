import argparse
import time

import torch
import torch.nn.functional as F

from ttt_e2e.attention import GradGradAttentionFn
from ttt_e2e.utils import make_window_mask


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()

    b, h, t, d = args.batch, args.heads, args.seq, args.dim
    device = 'cuda'
    q = torch.randn(b, h, t, d, device=device, requires_grad=True)
    k = torch.randn(b, h, t, d, device=device, requires_grad=True)
    v = torch.randn(b, h, t, d, device=device, requires_grad=True)
    attn_mask = make_window_mask(t, t, device=device, dtype=q.dtype).view(1, 1, t, t)

    def gradgrad_step():
        out = GradGradAttentionFn.apply(q, k, v, attn_mask)
        loss = out.sum()
        loss.backward()
        q.grad = None
        k.grad = None
        v.grad = None

    def ref_step():
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        loss = out.sum()
        loss.backward()
        q.grad = None
        k.grad = None
        v.grad = None

    gg_ms = bench(gradgrad_step, args.iters, args.warmup)
    ref_ms = bench(ref_step, args.iters, args.warmup)

    print(f'gradgrad attention: {gg_ms:.2f} ms/iter')
    print(f'ref attention:      {ref_ms:.2f} ms/iter')


if __name__ == '__main__':
    main()
