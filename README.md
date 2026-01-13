# TestTimeIdeas

Kernel prototypes for TTT‑E2E meta‑learning (grad‑grad safe attention).

## What’s here

- `ttt_kernels/triton_attention.py`
  - **Custom Triton forward kernel** for attention
  - **Grad‑grad safe backward** (PyTorch recompute)
  - Optional demo Triton backward for `dv` only (still grad‑grad safe because `dq/dk` recompute)

This is a **kernel‑nerd demo**: a real Triton forward kernel with meta‑learning compatibility.
It’s the starting point for full forward/backward/double‑backward kernels.

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

## Grad‑grad check

```bash
python -m tests.test_gradgrad
```

## Notes

- Full speed + grad‑grad requires **custom backward** and **double‑backward** kernels.
- This repo proves the forward‑kernel work and keeps meta‑gradients correct.
- Designed to integrate into TTT‑E2E training loops (swap attention backend).

## Next targets

1. Triton backward for Q/K/V grads (reduce recompute)
2. Double‑backward kernels (true grad‑grad speed)
3. Autotune block sizes and add FlashAttention parity tests
