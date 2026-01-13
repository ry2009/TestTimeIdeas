#!/usr/bin/env bash
set -euo pipefail

source /root/ttt-venv/bin/activate
export PYTHONPATH=/root/ttt-e2e

python /root/ttt-e2e/scripts/gradgrad_check.py
python /root/ttt-e2e/scripts/train_meta.py --config /root/ttt-e2e/configs/test.yaml --log-every 1
python /root/ttt-e2e/scripts/pretrain.py --config /root/ttt-e2e/configs/test_pretrain.yaml --log-every 1
python /root/ttt-e2e/scripts/demo_infer.py --context-len 1024 --query-len 256 --window-size 128 --chunk-size 128
python /root/ttt-e2e/scripts/bench_attention.py --seq 256 --heads 4 --dim 32 --batch 2 --iters 5 --warmup 2
