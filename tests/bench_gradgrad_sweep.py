import argparse
import csv
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


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.time()
    for _ in range(iters):
        fn()
    _sync()
    return (time.time() - t0) * 1000 / iters


def _run_once(b, h, t, d, dtype, causal, bwd_mode):
    device = torch.device('cuda')
    scale = 1.0 / (d ** 0.5)
    q = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)

    def _math():
        out = _math_attention(q, k, v, causal, scale)
        loss = out.float().mean()
        grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)
        s = sum([g.pow(2).mean() for g in grads])
        torch.autograd.grad(s, (q, k, v))

    def _triton():
        out = triton_attention(q, k, v, causal=causal, bwd_mode=bwd_mode)
        loss = out.float().mean()
        grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)
        s = sum([g.pow(2).mean() for g in grads])
        torch.autograd.grad(s, (q, k, v))

    return _math, _triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, default=1)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--bwd_mode', type=str, default='save_p_triton_full')
    parser.add_argument('--ts', type=str, default='512,1024,2048,4096,8192,16384')
    parser.add_argument('--out', type=str, default='')
    args = parser.parse_args()

    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    ts = [int(x.strip()) for x in args.ts.split(',') if x.strip()]

    rows = []
    for causal in (False, True):
        for t in ts:
            times_math = []
            times_triton = []
            for _ in range(args.repeats):
                _math, _triton = _run_once(args.b, args.h, t, args.d, dtype, causal, args.bwd_mode)
                times_math.append(_timeit(_math, iters=args.iters, warmup=args.warmup))
                times_triton.append(_timeit(_triton, iters=args.iters, warmup=args.warmup))
            rows.append({
                't': t,
                'causal': int(causal),
                'math_ms_mean': sum(times_math) / len(times_math),
                'math_ms_std': (torch.tensor(times_math).std(unbiased=False).item()),
                'triton_ms_mean': sum(times_triton) / len(times_triton),
                'triton_ms_std': (torch.tensor(times_triton).std(unbiased=False).item()),
            })
            print(f"[t={t} causal={causal}] math {rows[-1]['math_ms_mean']:.3f} ms, triton {rows[-1]['triton_ms_mean']:.3f} ms")

    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
