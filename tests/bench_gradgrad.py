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


def _timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    _sync(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    t0 = time.time()
    for _ in range(iters):
        fn()
    _sync(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    return (time.time() - t0) * 1000 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b', type=int, default=2)
    parser.add_argument('--h', type=int, default=2)
    parser.add_argument('--t', type=int, default=128)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--bwd_mode', type=str, default='recompute',
                        choices=['recompute', 'recompute_manual', 'recompute_sdp', 'recompute_sdp_auto', 'save_p', 'recompute_compiled', 'dv_only', 'custom'])
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
    q = torch.randn(args.b, args.h, args.t, args.d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.b, args.h, args.t, args.d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(args.b, args.h, args.t, args.d, device=device, dtype=dtype, requires_grad=True)

    def _run_math():
        out = _math_attention(q, k, v, args.causal, scale)
        loss = out.float().mean()
        grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)
        s = sum([g.pow(2).mean() for g in grads])
        torch.autograd.grad(s, (q, k, v))

    def _run_triton():
        out = triton_attention(q, k, v, causal=args.causal, bwd_mode=args.bwd_mode)
        loss = out.float().mean()
        grads = torch.autograd.grad(loss, (q, k, v), create_graph=True)
        s = sum([g.pow(2).mean() for g in grads])
        torch.autograd.grad(s, (q, k, v))

    if args.compile:
        _run_math = torch.compile(_run_math)

    t_math = _timeit(_run_math)
    print(f'gradgrad math:   {t_math:.3f} ms')

    if device.type == 'cuda':
        t_triton = _timeit(_run_triton)
        print(f'gradgrad triton: {t_triton:.3f} ms')
    else:
        print('gradgrad triton: skipped (CUDA required)')


if __name__ == '__main__':
    main()
