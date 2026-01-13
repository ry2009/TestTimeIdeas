from typing import Dict, Optional, Tuple, List

import torch
from torch.func import functional_call

from .utils import compute_loss


def _normalize_filters(param_filter: Optional[object]) -> Optional[List[str]]:
    if param_filter is None:
        return None
    if isinstance(param_filter, (list, tuple)):
        filters = [str(x).strip() for x in param_filter if str(x).strip()]
        return filters or None
    if isinstance(param_filter, str):
        filters = [s.strip() for s in param_filter.split(',') if s.strip()]
        return filters or None
    return [str(param_filter)]


def _split_params(model: torch.nn.Module, param_filter: Optional[object]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    params = dict(model.named_parameters())
    filters = _normalize_filters(param_filter)
    if not filters:
        trainable = list(params.keys())
    else:
        trainable = [k for k in params.keys() if any(f in k for f in filters)]
        if not trainable:
            trainable = list(params.keys())
    return params, trainable


def _chunk_tensor(x: torch.Tensor, chunk_size: Optional[int]) -> List[torch.Tensor]:
    if chunk_size is None:
        return [x]
    if x.dim() < 2:
        return [x]
    t = x.shape[1]
    if t % chunk_size != 0:
        raise ValueError(f"seq len {t} not divisible by chunk_size {chunk_size}")
    return list(x.split(chunk_size, dim=1))


def inner_update(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    trainable: List[str],
    targets: torch.Tensor,
    mask: torch.Tensor,
    lr: float,
    create_graph: bool,
    *,
    prefix_hidden: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if prefix_hidden is not None:
        logits = functional_call(model, params, (None,), {"mode": "suffix", "prefix_hidden": prefix_hidden})
    else:
        if input_ids is None:
            raise ValueError("input_ids required when prefix_hidden is None")
        logits = functional_call(model, params, (input_ids,), {"mode": "full"})
    loss = compute_loss(logits, targets, mask)
    grads = torch.autograd.grad(
        loss,
        [params[name] for name in trainable],
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )
    updated = params.copy()
    for name, g in zip(trainable, grads):
        updated[name] = updated[name] - lr * g
    return updated, loss


def _ttt_inner_scan(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    trainable: List[str],
    prefix_hidden: torch.Tensor,
    targets: torch.Tensor,
    inner_mask: torch.Tensor,
    *,
    inner_lr: float,
    inner_steps: int,
    chunk_size: Optional[int],
    create_graph: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    prefix_chunks = _chunk_tensor(prefix_hidden, chunk_size)
    target_chunks = _chunk_tensor(targets, chunk_size)
    mask_chunks = _chunk_tensor(inner_mask, chunk_size)

    inner_losses: List[torch.Tensor] = []
    for pch, tch, mch in zip(prefix_chunks, target_chunks, mask_chunks):
        for _ in range(inner_steps):
            params, loss = inner_update(
                model,
                params,
                trainable,
                tch,
                mch,
                lr=inner_lr,
                create_graph=create_graph,
                prefix_hidden=pch,
            )
        inner_losses.append(loss)

    inner_loss = torch.stack(inner_losses).mean() if inner_losses else torch.tensor(0.0, device=targets.device)
    return params, inner_loss


def _outer_loss_from_prefix(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    prefix_hidden: torch.Tensor,
    targets: torch.Tensor,
    outer_mask: torch.Tensor,
    *,
    chunk_size: Optional[int],
) -> torch.Tensor:
    prefix_chunks = _chunk_tensor(prefix_hidden, chunk_size)
    target_chunks = _chunk_tensor(targets, chunk_size)
    mask_chunks = _chunk_tensor(outer_mask, chunk_size)

    total_loss = torch.tensor(0.0, device=targets.device)
    total_mask = torch.tensor(0.0, device=targets.device)

    for pch, tch, mch in zip(prefix_chunks, target_chunks, mask_chunks):
        mask_sum = mch.sum()
        if mask_sum.item() <= 0:
            continue
        logits = functional_call(model, params, (None,), {"mode": "suffix", "prefix_hidden": pch})
        loss = compute_loss(logits, tch, mch)
        total_loss = total_loss + loss * mask_sum
        total_mask = total_mask + mask_sum

    return total_loss / total_mask.clamp_min(1.0)


def meta_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    inner_mask: torch.Tensor,
    outer_mask: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    param_filter: Optional[object] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    params, trainable = _split_params(model, param_filter)

    # compute prefix outputs once (prefix params are kept fixed in inner loop)
    prefix_hidden = functional_call(model, params, (input_ids,), {"mode": "prefix"})

    params, inner_loss = _ttt_inner_scan(
        model,
        params,
        trainable,
        prefix_hidden,
        targets,
        inner_mask,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        chunk_size=chunk_size,
        create_graph=True,
    )

    outer_loss = _outer_loss_from_prefix(
        model,
        params,
        prefix_hidden,
        targets,
        outer_mask,
        chunk_size=chunk_size,
    )
    return inner_loss, outer_loss


def ttt_apply(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    inner_mask: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    param_filter: Optional[object] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    params, trainable = _split_params(model, param_filter)
    prefix_hidden = functional_call(model, params, (input_ids,), {"mode": "prefix"})
    params, _ = _ttt_inner_scan(
        model,
        params,
        trainable,
        prefix_hidden,
        targets,
        inner_mask,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        chunk_size=chunk_size,
        create_graph=False,
    )
    return params


def ttt_logits(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    *,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    prefix_hidden = functional_call(model, params, (input_ids,), {"mode": "prefix"})
    if chunk_size is None:
        return functional_call(model, params, (None,), {"mode": "suffix", "prefix_hidden": prefix_hidden})

    prefix_chunks = _chunk_tensor(prefix_hidden, chunk_size)
    logits_chunks = []
    for pch in prefix_chunks:
        logits_chunks.append(functional_call(model, params, (None,), {"mode": "suffix", "prefix_hidden": pch}))
    return torch.cat(logits_chunks, dim=1)
