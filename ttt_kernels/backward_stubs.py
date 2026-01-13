"""Backward / double-backward stubs for grad-grad attention.

These are placeholders to make the kernel roadmap explicit.
"""

import torch


def triton_attn_bwd_dq(q, k, v, do, *, causal: bool, scale: float):
    raise NotImplementedError('dq Triton kernel not implemented yet')


def triton_attn_bwd_dk(q, k, v, do, *, causal: bool, scale: float):
    raise NotImplementedError('dk Triton kernel not implemented yet')


def triton_attn_bwd_dv(q, k, v, do, *, causal: bool, scale: float):
    raise NotImplementedError('dv Triton kernel not implemented yet')


# Double-backward stubs

def triton_attn_bwd2_dq(*args, **kwargs):
    raise NotImplementedError('double-backward dq not implemented yet')


def triton_attn_bwd2_dk(*args, **kwargs):
    raise NotImplementedError('double-backward dk not implemented yet')


def triton_attn_bwd2_dv(*args, **kwargs):
    raise NotImplementedError('double-backward dv not implemented yet')
