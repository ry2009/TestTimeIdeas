import torch
from ttt_e2e.attention import GradGradAttentionFn
from ttt_e2e.utils import make_window_mask


def main():
    torch.manual_seed(0)
    dtype = torch.float64
    device = 'cuda'
    b, h, t, d = 1, 2, 8, 4
    q = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, t, d, device=device, dtype=dtype, requires_grad=True)
    attn_mask = make_window_mask(t, 4, device=q.device, dtype=q.dtype).view(1, 1, t, t)

    def fn(q, k, v):
        return GradGradAttentionFn.apply(q, k, v, attn_mask)

    print('gradcheck:', torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4))
    print('gradgradcheck:', torch.autograd.gradgradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4))


if __name__ == '__main__':
    main()
