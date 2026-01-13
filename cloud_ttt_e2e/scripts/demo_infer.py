import argparse
import time

import torch

from ttt_e2e import ModelConfig, TTTModel
from ttt_e2e.data import generate_kv_batch
from ttt_e2e.meta import ttt_apply, ttt_logits


def stream_latency(model, seq_lens, window_size, chunk_size, device):
    model.eval()
    results = []
    for seq_len in seq_lens:
        tokens = torch.randint(0, model.cfg.vocab_size, (1, seq_len), device=device)
        torch.cuda.synchronize()
        t0 = time.time()
        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            ctx_start = max(0, end - window_size)
            chunk = tokens[:, ctx_start:end]
            _ = model(chunk)
        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_sec = seq_len / (t1 - t0)
        results.append((seq_len, tokens_per_sec))
    return results


def ttt_accuracy(model, context_len, query_len, inner_steps, inner_lr, device, param_filter=None, chunk_size=None, far_frac=0.0):
    model.eval()
    batch = generate_kv_batch(
        batch_size=1,
        context_len=context_len,
        query_len=query_len,
        vocab_size=model.cfg.vocab_size,
        device=device,
        pad_to=model.cfg.max_seq_len,
        far_frac=far_frac,
    )

    # no TTT
    with torch.no_grad():
        logits = model(batch.input_ids)
        preds = logits.argmax(dim=-1)
        correct = ((preds == batch.targets) * batch.outer_mask.bool()).sum().item()
        total = batch.outer_mask.sum().item()
        acc_no = correct / max(total, 1)

    params = ttt_apply(
        model,
        batch.input_ids,
        batch.targets,
        batch.inner_mask,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        param_filter=param_filter,
        chunk_size=chunk_size,
    )
    with torch.no_grad():
        logits = ttt_logits(model, params, batch.input_ids, chunk_size=chunk_size)
        preds = logits.argmax(dim=-1)
        correct = ((preds == batch.targets) * batch.outer_mask.bool()).sum().item()
        total = batch.outer_mask.sum().item()
        acc_ttt = correct / max(total, 1)

    return acc_no, acc_ttt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-len', type=int, default=8192)
    parser.add_argument('--query-len', type=int, default=2048)
    parser.add_argument('--window-size', type=int, default=512)
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--inner-steps', type=int, default=2)
    parser.add_argument('--inner-lr', type=float, default=1e-2)
    parser.add_argument('--suffix-layers', type=int, default=-1)
    parser.add_argument('--param-filter', type=str, default='suffix_blocks,ln_f,head')
    parser.add_argument('--far-frac', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--max-seq-len', type=int, default=0)
    args = parser.parse_args()

    max_seq_len = args.max_seq_len
    if max_seq_len <= 0:
        max_seq_len = args.context_len + args.query_len

    cfg = ModelConfig(
        vocab_size=4096,
        max_seq_len=max_seq_len,
        window_size=args.window_size,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        dropout=0.0,
        use_gradgrad_attention=False,
        suffix_layers=args.suffix_layers,
    )

    model = TTTModel(cfg).to(args.device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(state['model'])

    acc_no, acc_ttt = ttt_accuracy(
        model,
        context_len=args.context_len,
        query_len=args.query_len,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        device=args.device,
        param_filter=args.param_filter,
        chunk_size=args.chunk_size,
        far_frac=args.far_frac,
    )
    print(f'accuracy without TTT: {acc_no:.3f}')
    print(f'accuracy with TTT:    {acc_ttt:.3f}')

    seq_lens = [args.context_len, args.context_len * 2, args.context_len * 4]
    seq_lens = [min(s, 32768) for s in seq_lens]
    results = stream_latency(model, seq_lens, args.window_size, args.chunk_size, args.device)
    for seq_len, tps in results:
        print(f'seq {seq_len:5d} | tokens/s {tps:,.1f}')


if __name__ == '__main__':
    main()
