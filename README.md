# TestTimeIdeas

Kernel prototypes for TTT‑E2E meta‑learning (grad‑grad safe attention).

## What’s here

- `ttt_kernels/triton_attention.py`
  - Custom Triton forward kernel
  - Grad‑grad safe backward (PyTorch recompute)
  - Optional Triton **dV** path via `bwd_mode='dv_only'`
  - Backward/double‑backward stubs in `ttt_kernels/backward_stubs.py`

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

## Grad‑grad check

```bash
python -m tests.test_gradgrad
```

## Notes

- Full speed + grad‑grad requires custom backward **and** double‑backward kernels.
- Default `bwd_mode='recompute'` is correct for meta.
- `bwd_mode='dv_only'` uses Triton for dV and recompute for dQ/dK.

## Next targets

1. Triton backward for Q/K/V grads (reduce recompute)
2. Double‑backward kernels (true grad‑grad speed)
3. Autotune block sizes and add FlashAttention parity tests
