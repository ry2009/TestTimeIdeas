import argparse
import time
import yaml
from pathlib import Path

import torch
from tqdm import trange

from ttt_e2e import ModelConfig, TTTModel
from ttt_e2e.data import generate_kv_batch
from ttt_e2e.utils import TrainConfig, compute_loss, set_seed


def _coerce_config(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if isinstance(v, str):
            try:
                f = float(v)
                if f.is_integer() and ('e' not in v) and ('E' not in v) and ('.' not in v):
                    out[k] = int(f)
                else:
                    out[k] = f
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def load_config(path: str) -> TrainConfig:
    data = {}
    if path:
        data = yaml.safe_load(Path(path).read_text())
    data = _coerce_config(data)
    return TrainConfig(**data)


def save_ckpt(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--ckpt-path', type=str, default='')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    model_cfg = ModelConfig(
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        window_size=cfg.window_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        use_gradgrad_attention=False,
        suffix_layers=cfg.suffix_layers,
        dual_mlp=cfg.dual_mlp,
    )

    device = torch.device(cfg.device)
    model = TTTModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start = time.time()
    for step in trange(cfg.steps, desc='pretrain'):
        batch = generate_kv_batch(
            batch_size=cfg.batch_size,
            context_len=cfg.context_len,
            query_len=cfg.query_len,
            vocab_size=cfg.vocab_size,
            num_keys=cfg.num_keys,
            device=cfg.device,
            pad_to=cfg.max_seq_len,
            far_frac=cfg.far_frac,
        )
        logits = model(batch.input_ids)
        loss = compute_loss(logits, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - start
            print(f'step {step+1:05d} | loss {loss.item():.4f} | {elapsed:.1f}s')

    if args.ckpt_path:
        save_ckpt(args.ckpt_path, model, optimizer, cfg.steps)


if __name__ == '__main__':
    main()
