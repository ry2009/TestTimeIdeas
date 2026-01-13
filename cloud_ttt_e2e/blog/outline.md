# Blog Outline — TTT‑E2E PyTorch Recreation

## 1) Why this project
- Long‑context scaling pain: full attention vs constant‑time approaches.
- Test‑Time Training (TTT‑E2E) as a principled path.

## 2) Baseline alignment (JAX → PyTorch)
- JAX repo as ground truth for meta loop mechanics.
- Prefix/suffix split + chunked scan alignment.
- Inner loop on suffix, outer loss post‑adaptation.

## 3) What we rebuilt
- TTT‑E2E meta‑learning loop
- Grad‑grad‑safe attention
- Streaming inference demo

## 4) Architecture + training loop
- Model diagram
- Prefix pass once, suffix updates on chunks
- `create_graph=True` and why it matters

## 5) Kernel engineering
- Why FlashAttention blocks grad‑grad
- Our grad‑grad attention path and recompute strategy
- Speed/latency trade‑offs

## 6) Results
- Meta vs pretrain on KV retrieval (8K context)
- TTT gains after adaptation
- Throughput vs context length (6K/12K/24K streaming)

## 7) Lessons + next steps
- What worked, what broke
- Roadmap for fused kernel + larger data
