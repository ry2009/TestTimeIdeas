# TTT-E2E PyTorch Prototype (H100)

This repo is a minimal, research-focused prototype of **TTT‑E2E** with:

- meta‑learning with gradients‑of‑gradients
- a grad‑grad‑safe attention path
- prefix/suffix split + chunked inner‑loop scan (aligned to the JAX ref)
- long‑context streaming inference demo

It is designed to be a clean, hackable base for a blog‑quality reproduction and system‑level demo.

## Latest run snapshot (H100)

Artifacts captured from the last meta‑learning demo run (stopped at step ~4100/5000):

- Logs: `meta_8k_v2.log`
- Plots:
  - `artifacts/meta_loss_curves.png` (+ CSV)
  - `artifacts/meta_eval_acc.png` (+ CSV)
- Kernel bench note: `artifacts/kernel_bench_h100.txt`
- Notes: `blog/findings.md` (short summary + takeaways)

## Quickstart

```bash
source /root/ttt-venv/bin/activate
cd /root/ttt-e2e
export PYTHONPATH=/root/ttt-e2e

# grad‑grad safety checks
python scripts/gradgrad_check.py

# meta‑learning run (8K demo config)
python scripts/train_meta.py --config configs/meta_8k.yaml

# pretraining baseline
python scripts/pretrain.py --config configs/pretrain_8k.yaml

# long‑context demo (accuracy + streaming throughput)
python scripts/demo_infer.py --context-len 8192 --query-len 2048 --window-size 512 --chunk-size 1024

# attention benchmark
python scripts/bench_attention.py --seq 1024 --heads 8 --dim 64 --batch 2
```

## What’s implemented

- **Grad‑grad safe attention** (`ttt_e2e/attention.py`)
  - Custom autograd that keeps the backward fully differentiable, enabling `create_graph=True`.
- **TTT‑E2E meta‑learning loop** (`ttt_e2e/meta.py`)
  - Prefix pass computed once, inner loop updates suffix blocks on chunked suffix windows.
- **Prefix/suffix split** (`ttt_e2e/model.py`)
  - Configurable `suffix_layers` for inner‑loop updates.
- **Synthetic KV retrieval dataset** (`ttt_e2e/data.py`)
  - Query keys optionally forced to early context (`far_frac`) so only weight updates help.
- **Streaming inference** (`scripts/demo_infer.py`)
  - Windowed streaming with constant per‑token compute.

## Notes

- **Chunked scan alignment**: `chunk_size` must divide `max_seq_len`.
- **Inner‑loop params**: use `ttt_param_filter` (comma‑separated substrings) to control which params update, e.g. `suffix_blocks,ln_f,head`.
- This is an **engineering prototype**: small model, synthetic task, and minimal configs.
- The attention kernel is designed to be safe for higher‑order gradients. It can be swapped with a faster kernel once you implement a custom Triton/CUDA op.
