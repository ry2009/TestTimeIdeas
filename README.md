# TestTimeIdeas

Kernel prototypes for TTT-E2E meta-learning.

## What’s here

- `ttt_kernels/triton_attention.py`: Triton forward kernel + gradgrad-safe backward (PyTorch recompute). Optional demo Triton backward for `dv`.

This is a **kernel-nerd demo**: custom Triton kernel for attention forward, still supports gradients-of-gradients via recompute, so it is usable inside meta-learning (TTT-E2E).

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

## Notes

- Full speed + gradgrad requires custom backward **and** double-backward kernels. This repo is the stepping stone.
- The `use_triton_backward=True` path is a demo for `dv` only; `dq/dk` still use recompute to preserve gradgrad safety.
