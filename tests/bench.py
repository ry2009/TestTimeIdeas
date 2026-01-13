import argparse
import time

import torch

from ttt_kernels.triton_attention import triton_attention


def _math_attention(q, k, v, causal, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        t = scores.size(-1)
        mask = torch.triu(torch.ones((t, t), device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def bench(fn, q, k, v, iters=50):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(q, k, v)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, default=2)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--t', type=int, default=2048)
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--dtype', type=str, default='fp16')
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    scale = 1.0 / (args.d ** 0.5)

    q = torch.randn(args.b, args.h, args.t, args.d, device='cuda', dtype=dtype)
    k = torch.randn(args.b, args.h, args.t, args.d, device='cuda', dtype=dtype)
    v = torch.randn(args.b, args.h, args.t, args.d, device='cuda', dtype=dtype)

    t_math = bench(lambda q, k, v: _math_attention(q, k, v, True, scale), q, k, v, iters=args.iters)
    t_triton = bench(lambda q, k, v: triton_attention(q, k, v, causal=True), q, k, v, iters=args.iters)

    print(f'math forward:   {t_math:.3f} ms')
    print(f'triton forward: {t_triton:.3f} ms')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required for bench')
    main()
