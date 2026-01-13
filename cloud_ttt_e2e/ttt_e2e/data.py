from dataclasses import dataclass
from typing import Optional

import torch


PAD = 0
BOS = 1
SEP = 2
KEY = 3
VAL = 4
QRY = 5


@dataclass
class KVBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    inner_mask: torch.Tensor
    outer_mask: torch.Tensor


def _build_pairs(keys, values):
    tokens = []
    for k, v in zip(keys, values):
        tokens.extend([KEY, k, VAL, v, SEP])
    return tokens


def _build_queries(keys, values):
    tokens = []
    value_positions = []
    for k, v in zip(keys, values):
        start = len(tokens)
        tokens.extend([QRY, k, VAL, v, SEP])
        value_positions.append(start + 3)
    return tokens, value_positions


def generate_kv_batch(
    batch_size: int,
    context_len: int,
    query_len: int,
    vocab_size: int,
    num_keys: int = 256,
    device: str = "cuda",
    pad_to: Optional[int] = None,
    pad_multiple: Optional[int] = None,
    far_frac: float = 0.0,
) -> KVBatch:
    tokens_per_pair = 5
    n_pairs = max(1, context_len // tokens_per_pair)
    n_queries = max(1, query_len // tokens_per_pair)

    key_start = 6
    key_end = min(key_start + num_keys, vocab_size // 2)
    val_start = key_end
    val_end = min(val_start + num_keys, vocab_size - 1)

    batch_tokens = []
    inner_masks = []
    outer_masks = []

    # total sequence length includes BOS and is one longer than input_ids
    max_len = 1 + n_pairs * tokens_per_pair + n_queries * tokens_per_pair

    if pad_to is not None:
        # pad_to is desired input_ids length
        max_len = max(max_len, pad_to + 1)

    if pad_multiple is not None:
        input_len = max_len - 1
        if input_len % pad_multiple != 0:
            input_len = ((input_len // pad_multiple) + 1) * pad_multiple
            max_len = input_len + 1

    far_frac = max(0.0, min(float(far_frac), 1.0))

    for _ in range(batch_size):
        # unique keys limited by available pool
        pool = torch.randperm(key_end - key_start) + key_start
        pool = pool[: min(n_pairs, pool.numel())]
        keys = pool.tolist()
        n_pairs_actual = len(keys)
        vals = torch.randint(val_start, val_end, (n_pairs_actual,)).tolist()
        mapping = {k: v for k, v in zip(keys, vals)}

        context = _build_pairs(keys, vals)

        if far_frac > 0.0:
            far_count = max(1, int(n_pairs_actual * far_frac))
            q_keys_idx = torch.randint(0, far_count, (n_queries,)).tolist()
            q_keys = [keys[i] for i in q_keys_idx]
        else:
            q_keys_idx = torch.randint(0, n_pairs_actual, (n_queries,)).tolist()
            q_keys = [keys[i] for i in q_keys_idx]

        q_vals = [mapping[k] for k in q_keys]

        queries, value_positions = _build_queries(q_keys, q_vals)

        seq = [BOS] + context + queries
        if len(seq) < max_len:
            seq = seq + [PAD] * (max_len - len(seq))

        # masks (aligned to targets length)
        inner_mask = [0] * (len(seq) - 1)
        outer_mask = [0] * (len(seq) - 1)

        # inner: context region
        context_start = 1
        context_end = 1 + len(context)
        for i in range(context_start, context_end):
            if i - 1 < len(inner_mask):
                inner_mask[i - 1] = 1

        # outer: value tokens in queries
        query_start = 1 + len(context)
        for pos in value_positions:
            idx = query_start + pos
            if idx - 1 < len(outer_mask):
                outer_mask[idx - 1] = 1

        batch_tokens.append(seq[:max_len])
        inner_masks.append(inner_mask)
        outer_masks.append(outer_mask)

    input_ids = torch.tensor(batch_tokens, device=device, dtype=torch.long)
    targets = input_ids[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    inner_mask = torch.tensor(inner_masks, device=device, dtype=torch.float32)
    outer_mask = torch.tensor(outer_masks, device=device, dtype=torch.float32)

    return KVBatch(input_ids=input_ids, targets=targets, inner_mask=inner_mask, outer_mask=outer_mask)
