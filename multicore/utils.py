from typing import Tuple, Callable, Union, Optional

import torch
from torch import Tensor
from torch.fft import fftn, ifftn

import pdb


def conjugate_gradient(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    dim: Union[int, Tuple[int, ...]],
    T: int,
    tol: float = 1e-10,
    stop_criterion: Optional[Callable[[Tensor], bool]] = None
) -> Tuple[Tensor, bool]:
    x = torch.zeros_like(b)
    r = b
    p = r
    rr = torch.sum(r * r, dim=dim, keepdim=True)

    if stop_criterion is None:
        stop_criterion = lambda x: True

    cont = True
    iter_id = 0
    while cont and iter_id < 20:
        print(f'Got here {iter_id + 1}')
        for t in range(T):
            Ap = A(p)
            pAp = torch.sum(p * Ap, dim=dim, keepdim=True)

            alpha = rr / pAp
            x = x + alpha * p

            r = r - alpha * Ap
            if torch.all(torch.abs(r) < tol):
                return x, True

            rr_old = rr
            rr = torch.sum(r * r, dim=dim, keepdim=True)
            beta = rr / rr_old
            p = r + beta * p
        cont = not stop_criterion(x)
        iter_id += 1
    if cont:
        return x, False
    else:
        return x, True


def conj_grad(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    dim: Union[int, Tuple[int, ...]],
    T: int,
    tol: float = 1e-10,
    stop_criterion: Optional[Callable[[Tensor], bool]] = None
) -> Tuple[Tensor, bool]:
    x = torch.zeros_like(b)
    r = b
    p = r
    rr = torch.sum(r * r, dim=dim, keepdim=True)

    if stop_criterion is None:
        stop_criterion = lambda x: True

    cont = True
    iter_id = 0
    while cont and iter_id < 20:
        print(f'Got here {iter_id + 1}')
        for t in range(T):
            Ap = A(p)
            pAp = torch.sum(p * Ap, dim=dim, keepdim=True)

            alpha = rr / pAp
            x = x + alpha * p

            r = r - alpha * Ap

            metric = (torch.norm(r) / torch.norm(b)) ** 2
            print(metric)
            if (metric < tol):
            # if torch.all(torch.abs(r) < tol):
                return x, True

            rr_old = rr
            rr = torch.sum(r * r, dim=dim, keepdim=True)
            beta = rr / rr_old
            p = r + beta * p
        cont = not stop_criterion(x)
        iter_id += 1
    if cont:
        return x, False
    else:
        return x, True


def finite_diff_2d(data: Tensor, keep_dim: bool = True) -> Tuple[Tensor, Tensor]:
    dx = data[:, 1:, :] - data[:, :-1, :]
    dy = data[:, :, 1:] - data[:, :, :-1]
    if keep_dim:
        B, dim_x, dim_y = data.size()
        dx = torch.cat([dx, torch.zeros(B, 1, dim_y)], dim=-2)
        dy = torch.cat([dy, torch.zeros(B, dim_x, 1)], dim=-1)
    return dx, dy


def inv_finite_diff_2d(dx: Tensor, dy: Tensor):
    Dx = torch.cat(
        [-dx[:, :, 0:1], dx[:, :, 1:-1] - dx[:, :, :-2], dx[:, :, -2:-1]],
        dim=-1
    )
    Dy = torch.cat(
        [-dy[:, 0:1, :], dy[:, 1:-1, :] - dy[:, :-2, :], dy[:, -2:-1, :]],
        dim=-2
    )
    return Dx + Dy


def ufft(
    data: Tensor,
    mask: Tensor,
    signal_ndim: int,
    normalized: bool = False
) -> Tensor:
    """Undersampled fast Fourier transform."""
    return mask * fftn(data, dim=2, norm='ortho')
    # data_ = torch.stack([data, torch.zeros_like(data)], dim=3)
    # mask_ = mask.unsqueeze(dim=3)
    # fft_data_ = torch.fft(data_, signal_ndim, normalized)
    # return torch.view_as_complex(mask_ * fft_data_)


def iufft(
    data: Tensor,
    signal_ndim: int,
    normalized: bool = False
) -> Tensor:
    """Inverse undersampled fast Fourier transform.  Note that iufft(ufft(x)) ≠ x."""
    return ifftn(data, dim=2, norm='ortho').real
    # out = torch.ifft(torch.view_as_real(data), signal_ndim, normalized)[:, :, :, 0]
    # return out
