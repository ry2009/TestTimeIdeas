import argparse
import os
import sys
import time

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


def _sync(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def bench(fn, q, k, v, iters=50, warmup=10):
    for _ in range(warmup):
        fn(q, k, v)
    _sync(q.device)
    t0 = time.time()
    for _ in range(iters):
        fn(q, k, v)
    _sync(q.device)
    return (time.time() - t0) * 1000 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, default=2)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--t', type=int, default=2048)
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--mode', type=str, default='both', choices=['both', 'math', 'triton'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--causal', action='store_true')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    scale = 1.0 / (args.d ** 0.5)
    q = torch.randn(args.b, args.h, args.t, args.d, device=device, dtype=dtype)
    k = torch.randn(args.b, args.h, args.t, args.d, device=device, dtype=dtype)
    v = torch.randn(args.b, args.h, args.t, args.d, device=device, dtype=dtype)

    math_fn = lambda q, k, v: _math_attention(q, k, v, args.causal, scale)
    if args.compile:
        math_fn = torch.compile(math_fn)

    if args.mode in ('both', 'math'):
        t_math = bench(math_fn, q, k, v, iters=args.iters, warmup=args.warmup)
        print(f'math forward:   {t_math:.3f} ms')

    if args.mode in ('both', 'triton'):
        if device.type != 'cuda':
            print('triton forward: skipped (CUDA required)')
        else:
            t_triton = bench(lambda q, k, v: triton_attention(q, k, v, causal=args.causal), q, k, v,
                             iters=args.iters, warmup=args.warmup)
            print(f'triton forward: {t_triton:.3f} ms')


if __name__ == '__main__':
    main()
