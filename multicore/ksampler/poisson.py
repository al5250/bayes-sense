import numpy as np
from numpy import ndarray
from numpy.fft import ifftshift
from sigpy.mri import poisson

from multicore.ksampler import KSampler


class PoissonSampler(KSampler):

    def __init__(self, n_kspace, seed, **kwargs):
        self.n_kspace = n_kspace

        if 'calib' in kwargs.keys():
            kwargs['calib'] = tuple(kwargs['calib'])

        self._masks = []
        for i in range(self.n_kspace):
            s = seed[i] if hasattr(seed, '__iter__') else seed
            mask = poisson(seed=s, **kwargs).astype('bool')
            if mask.shape[0] == 1:
                mask = np.tile(mask, (mask.shape[1], 1))
            elif mask.shape[1] == 1:
                mask = np.tile(mask, (1, mask.shape[0]))
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