# Findings (TTT-E2E v2, 8K demo)

## Summary
- Meta-training stopped early at step **4100 / 5000** to save credits.
- Eval at step 4000: **acc_no = 0.035**, **acc_ttt = 0.045** (TTT wins).
- Curves and CSVs are in `cloud_ttt_e2e/artifacts/`.

## Artifacts
- `meta_loss_curves.png` / `meta_loss_curves.csv`
- `meta_eval_acc.png` / `meta_eval_acc.csv`
- `kernel_bench_h100.txt` (kernel demo timing)

## Notes
- This run uses configs: `configs/meta_8k_v2.yaml` + `configs/pretrain_8k_v2.yaml`.
- TTT improves accuracy vs no-TTT in this synthetic long-context key-value task.
- Remaining work: pretrain v2 + eval after pretrain to confirm end-to-end gain.
