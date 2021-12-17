from typing import Optional, Tuple, List

import torch
from torch import Tensor

from pytorch_wavelets import DWTForward, DWTInverse

from multicore.projections.projection import Projection
from multicore.utils import get_torch_device

from torch.nn.functional import pad

import pdb


class Wavelet2D(Projection):

    def __init__(
        self,
        num_levels: int = 1,
        wave: str = 'db2',
        mode: str = 'periodization',
        device: Optional[str] = None,
        pad_sizes: Optional[Tuple[int, int]] = None
    ) -> None:
        self.num_levels = num_levels
        self.wave = wave
        self.mode = mode
        device = get_torch_device(device)

        self.xfm = DWTForward(J=num_levels, wave=wave, mode=mode).to(device)
        self.ifm = DWTInverse(wave=wave, mode=mode).to(device)

        self.pad_sizes = pad_sizes

    def apply(self, x: Tensor) -> Tensor:
        
        if self.pad_sizes is not None:
            pad1, pad2 = self.pad_sizes
            pad_scheme = (pad2 // 2, pad2 // 2, pad1 // 2, pad1 // 2)
            x = pad(x, pad_scheme, "constant", 0)

        # Split out real and imag channels for complex data
        complex_input = torch.is_complex(x)
        if complex_input:
            x = torch.cat([x.real, x.imag], dim=0)

        x, shape = self.flatten4d(x)
        H, W = shape[-2:]
        assert H == W # Check that height is equal to width
        assert (H & (H-1) == 0) and H != 0 # Check power of 2

        y_low, y_high = self.xfm(x)

        y = self.coeffs_to_tensor(y_low, y_high)
        y = self.unflatten4d(y, shape)

        if complex_input:
            y = torch.complex(*y.chunk(2, dim=0))

        return y

    def T_apply(self, y: Tensor) -> Tensor:
        
        # Split out real and imag channels for complex data
        complex_input = torch.is_complex(y)
        if complex_input:
            y = torch.cat([y.real, y.imag], dim=0)

        y, shape = self.flatten4d(y)
        y_low, y_high = self.tensor_to_coeffs(y, self.num_levels)
        x = self.ifm((y_low, y_high))

        x = self.unflatten4d(x, shape)

        if complex_input:
            x = torch.complex(*x.chunk(2, dim=0))
        
        if self.pad_sizes is not None:
            pad1, pad2 = self.pad_sizes
            border1, border2 = pad1 // 2, pad2 // 2
            x = x[..., border1:-border1, border2:-border2]

        return x

    @staticmethod
    def coeffs_to_tensor(y_low: Tensor, y_high: List[Tensor]) -> Tensor:
        out = y_low
        for yh in y_high[::-1]:
            ylh = yh[..., 0, :, :]
            yhl = yh[..., 1, :, :]
            yhh = yh[..., 2, :, :]
            out = torch.cat(
                [torch.cat([out, ylh], dim=-1), torch.cat([yhl, yhh], dim=-1)],
                dim=-2
            )
        return out

    @staticmethod
    def tensor_to_coeffs(y: Tensor, num_levels: int) -> Tuple[Tensor, List[Tensor]]:
        base_dim = y.size(dim=-1)
        dim = base_dim // (2 ** num_levels)
        y_low = y[..., :dim, :dim]
        y_high = []
        for i in range(num_levels):
            ylh = y[..., :dim, dim:2*dim]
            yhl = y[..., dim:2*dim, :dim]
            yhh = y[..., dim:2*dim, dim:2*dim]
            y_high.append(torch.stack([ylh, yhl, yhh], dim=2))
            dim *= 2
        y_high = y_high[::-1]
        return y_low, y_high

    @staticmethod
    def flatten4d(x: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
        shape = x.size()
        num_dim = len(shape)
        if num_dim > 4:
            x = x.flatten(start_dim=0, end_dim=-4)
        elif num_dim == 3:
            x = x.unsqueeze(dim=0)
        return x, shape

    @staticmethod
    def unflatten4d(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
        if len(shape) > 4:
            x = x.unflatten(dim=0, sizes=shape[:-3])
        elif x.size(dim=0) == 1:
            x = x.squeeze(dim=0)
        return x
