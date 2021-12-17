from typing import Tuple

import numpy as np
from numpy import ndarray
from numpy.fft import ifftshift

from multicore.ksampler import KSampler


class RandomUniformSampler(KSampler):

    def __init__(
        self, 
        n_kspace: int,
        dim_x: int,
        dim_y: int,
        accel: int,
        calib: int
    ):
        self.n_kspace = n_kspace
        self._masks = []
        for i in range(self.n_kspace):
            mask = np.zeros((dim_x, dim_y), dtype=np.bool8) 
            start, end = dim_x // 2 - calib // 2, dim_x // 2 + calib // 2
            mask[start:end] = True
            total_select = dim_x // accel
            remain_select = total_select - calib
            options = list(range(start)) + list(range(end, dim_x)) 
            selected = np.random.choice(options, size=remain_select, replace=False)
            mask[selected] = True
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