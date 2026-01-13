import argparse
import torch

from ttt_e2e import ModelConfig, TTTModel
from ttt_e2e.data import generate_kv_batch
from ttt_e2e.meta import ttt_apply, ttt_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--context-len', type=int, default=4096)
    parser.add_argument('--query-len', type=int, default=1024)
    parser.add_argument('--window-size', type=int, default=512)
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--inner-steps', type=int, default=2)
    parser.add_argument('--inner-lr', type=float, default=1e-2)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--suffix-layers', type=int, default=-1)
    parser.add_argument('--param-filter', type=str, default='suffix_blocks,ln_f,head')
    parser.add_argument('--max-seq-len', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--far-frac', type=float, default=0.0)
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
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state['model'])
    model.eval()

    total_outer = 0.0
    correct_no = 0.0
    correct_ttt = 0.0

    for _ in range(args.batches):
        batch = generate_kv_batch(
            batch_size=1,
            context_len=args.context_len,
            query_len=args.query_len,
            vocab_size=cfg.vocab_size,
            num_keys=cfg.num_keys,
            device=args.device,
            pad_to=max_seq_len,
            far_frac=args.far_frac,
        )
        with torch.no_grad():
            logits = model(batch.input_ids)
            preds = logits.argmax(dim=-1)
            correct_no += ((preds == batch.targets) * batch.outer_mask.bool()).sum().item()
            total_outer += batch.outer_mask.sum().item()

        params = ttt_apply(
            model,
            batch.input_ids,
            batch.targets,
            batch.inner_mask,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            param_filter=args.param_filter,
            chunk_size=args.chunk_size,
        )
        with torch.no_grad():
            logits = ttt_logits(model, params, batch.input_ids, chunk_size=args.chunk_size)
            preds = logits.argmax(dim=-1)
            correct_ttt += ((preds == batch.targets) * batch.outer_mask.bool()).sum().item()

    acc_no = correct_no / max(total_outer, 1.0)
    acc_ttt = correct_ttt / max(total_outer, 1.0)

    print(f'no‑TTT acc:  {acc_no:.3f}')
    print(f'TTT acc:     {acc_ttt:.3f}')


if __name__ == '__main__':
    main()
