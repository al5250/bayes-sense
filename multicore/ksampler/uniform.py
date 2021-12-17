from typing import Tuple

import numpy as np
from numpy import ndarray
from numpy.fft import ifftshift

from multicore.ksampler import KSampler


class UniformSampler(KSampler):

    def __init__(
        self, 
        n_kspace: int,
        dim_x: int,
        dim_y: int,
        factor_x: int = 1,
        factor_y: int = 1,
        stagger: bool = False,
        calib: Tuple[int, int] = (0, 0)
    ):
        self.n_kspace = n_kspace
        self._masks = []
        for i in range(self.n_kspace):
            mask = np.zeros((dim_x, dim_y), dtype=np.bool8) 
            if stagger:
                mask[i::factor_x, i::factor_y] = True
            else:
                mask[0::factor_x, 0::factor_y] = True
            start_x, end_x = dim_x // 2 - calib[0] // 2, dim_x // 2 + calib[0] // 2
            start_y, end_y = dim_y // 2 - calib[1] // 2, dim_y // 2 + calib[1] // 2
            mask[start_x:end_x, start_y:end_y] = True
            self._masks.append(ifftshift(mask))
        self._masks = np.array(self._masks)
    
    @property
    def masks(self) -> ndarray:
        return self._masks
    
    def __call__(self, kspaces_full: ndarray) -> ndarray:
        if len(kspaces_full) != self.n_kspace:
            raise ValueError(
                f'KSampler expects {self.n_kspace} items, but {len(kspaces_full)} were provided.'
            )
        return np.array([m * k for k, m in zip(kspaces_full, self._masks)])