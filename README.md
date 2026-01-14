# TestTimeIdeas

Kernel prototypes for TTT‑E2E meta‑learning (grad‑grad safe attention).

## What’s here

- `ttt_kernels/triton_attention.py`
  - Custom Triton forward kernel
  - Grad‑grad safe backward with multiple modes (`recompute`, `save_p`, `save_p_triton_bwd`, `save_p_triton_full`)
  - Triton **dV + rowsum + dQ/dK** kernels with fused softmax‑backward in `save_p_triton_full`
  - Block‑sparse causal masking inside the Triton backward kernels
  - Backward/double‑backward logic in `ttt_kernels/backward_stubs.py`

- `docs/gradgrad_attention_design.md`
  - Kernel design notes + pseudocode

- `docs/kernel_diagram.txt`
  - ASCII diagram of forward/backward/double‑backward stack

This repo is the **kernel‑nerd demo**: real Triton kernel + explicit grad‑grad plan.

## Run on H100

```bash
pip install triton
python -m tests.bench --b 2 --h 4 --t 2048 --d 128 --dtype fp16 --iters 50
```

Expected output:
```
math forward:   X.XXX ms
triton forward: Y.YYY ms
```

Latest Modal H100 run (saved in `artifacts/modal_bench_h100_latest.txt`):
- **forward**: triton 0.133 ms vs math 0.213 ms (win)
- **grad‑grad (recompute)**: still slower than math (expected).
- **save_p** (cached softmax, PyTorch mats) is fastest at small/medium T.
- **save_p_triton_full** (Triton dV + rowsum + dQ/dK, grad‑grad safe) wins at long context:
  - T=8192 non‑causal: **3.632 ms vs 4.063 ms** (≈ **1.12×** faster).
  - T=8192 causal: **4.066 ms vs 5.179 ms** (≈ **1.27×** faster).
  - Small T has launch overhead; expect parity only at larger T.

Correctness spot‑check (`tests/test_gradgrad_compare.py`, b=1 h=1 t=64 d=64):
- forward max abs err: **0.0**
- grad max abs err (q,k,v): **~6e‑7**
- grad‑grad max abs err (q,k,v): **0.0**

Grad‑grad sweep (H100, `save_p_triton_full`, b=1 h=1 d=64) saved in:
- `artifacts/gradgrad_sweep.csv`
- `artifacts/gradgrad_sweep.png`
- `artifacts/gradgrad_sweep_speedup.png`

Key points from sweep:
- **non‑causal T=16k:** 11.466 ms vs 15.023 ms (≈ **1.31×**)
- **causal T=16k:** 13.649 ms vs 19.350 ms (≈ **1.42×**)
- crossover happens around **8k**, launch overhead dominates smaller T.

Repro sweep:
```bash
python tests/bench_gradgrad_sweep.py --b 1 --h 1 --d 64 --dtype fp16 --iters 8 --warmup 2 --repeats 4 \\
  --bwd_mode save_p_triton_full --ts 512,1024,2048,4096,8192,16384 --out artifacts/gradgrad_sweep.csv
```

One‑pager PDF (for sharing):
- `artifacts/ttt_gradgrad_onepager.pdf`

## Grad‑grad check

```bash
python -m tests.test_gradgrad
```

## CPU prep (no GPU credits)

Use this to validate correctness + grad‑grad and prep benchmarks.

```bash
# forward bench (CPU-safe, skips triton)
python -m tests.bench --device cpu --mode both --compile

# grad‑grad bench (CPU-safe, skips triton)
python -m tests.bench_gradgrad --device cpu --compile
```

On GPU, drop `--device cpu` and it will run both math + triton.

Note: the Triton path is enabled for fp16/bf16 inputs; fp32 falls back to math.

## Modal (GPU without SSH)

If you want to run on Modal instead of SSH:

```bash
pip install modal
python -m modal setup
modal run modal_app.py
```

The Modal app uses an H100 and runs:

- `tests/bench.py` (forward timing)
- `tests/bench_gradgrad.py` (grad‑grad timing)

## Notes

- Full speed + grad‑grad requires custom backward **and** double‑backward kernels.
- Default `bwd_mode='recompute'` is correct for meta but slow.
- `bwd_mode='save_p_triton_full'` is **grad‑grad safe** and shows wins at long context.

## Next targets

1. Triton backward for Q/K/V grads (reduce recompute)
2. Autotune block sizes + fuse dp/rowsum tighter
3. Double‑backward Triton kernels for full meta‑speed
