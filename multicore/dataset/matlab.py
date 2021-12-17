from typing import Optional, List, Tuple

import torch
import numpy as np
from scipy.io import loadmat
from numpy import ndarray
from numpy.fft import fft2
from hydra.utils import instantiate
from torch.nn.functional import pad

from multicore.dataset.dataset import MRIDataset
from multicore.ksampler import KSampler


class MatLabDataset(MRIDataset):

    def __init__(
        self,
        img_file: str,
        img_key: str,
        ksampler: KSampler,
        data_file: Optional[str] = None, 
        data_key: Optional[str] = None, 
        coil_file: Optional[str] = None, 
        coil_key: Optional[str] = None,
        names: Optional[List[str]] = None,
        pad_size: Optional[int] = None,
        max_contrasts: Optional[int] = None,
        filter_idx: Optional[List[int]] = None
    ):
        self._imgs = loadmat(img_file)[img_key].transpose(2, 0, 1)

        # Cast to real
        # self._imgs = np.abs(self._imgs)
        # self._imgs = self._imgs.real

        # Scale to [0, 1]
        # self._imgs = (self._imgs) / (self._imgs.max())

        max_contrasts = max_contrasts or len(self._imgs)
        self._imgs = self._imgs[:max_contrasts]
        
        if data_file is not None:
            data = loadmat(data_file)[data_key]
        
        if coil_file is not None:
            self._coils = loadmat(coil_file)[coil_key].transpose(2, 0, 1)
            
            # Normalize coils
            # self._coils = self._coils / np.abs(self._coils).max()

            # Coils set to identity and add noise
            # self._coils = np.ones((8,) + self.img_size)
            # self._coils = self._coils + 1e-1 * np.random.normal(size=self._coils.shape)

            # Multiply coils
            # self._coils = 100 * self._coils 

            if data_file is None:
                data = np.repeat(np.expand_dims(self._imgs, axis=1), len(self._coils), axis=1)
                self._data = self._coils * data
                print('Created Data')
            else:
                self._data = data.transpose(3, 2, 0, 1)
            self._data = self._data[:max_contrasts]
        else:
            self._coils = None

            if data_file is None:
                data = self._imgs
            else:
                self._data = np.expand_dims(data.transpose(2, 0, 1), axis=1)
        
        if filter_idx is not None:
            self._data = np.array([self._data[x] for x in filter_idx])
            self._imgs = np.array([self._imgs[x] for x in filter_idx])

        # Pad objects (e.g. to power of 2) 
        if pad_size is not None:
            pad2 = pad_size - self._data.shape[-2]
            pad1 = pad_size - self._data.shape[-1]
            pad_scheme = (pad1 // 2, pad1 // 2, pad2 // 2, pad2 // 2)
            self._data = pad(torch.tensor(self._data), pad_scheme, "constant", 0).numpy()
            self._imgs = pad(torch.tensor(self._imgs), pad_scheme, "constant", 0).numpy()
            self._coils = pad(torch.tensor(self._coils), pad_scheme, "constant", 0).numpy()

        if names is not None:
            if len(names) != len(self._img):
                raise ValueError(
                    f'Data file has {len(self._img)} contrasts, '
                    f'but {len(names)} names were provided.'
                )
            self._names = names
        else:
            self._names = [f'Img {i}' for i in range(len(self._data))]
        
        self._ksampler = instantiate(ksampler)
        self._kspaces_full = fft2(self._data, norm='ortho', axes=(-2, -1))
        
        # Added noise -- might reduce # of CG iterations needed
        # noise_std = 3e-5
        # noise = noise_std * np.random.normal(size=self._kspaces_full.shape) + noise_std * 1j * np.random.normal(size=self._kspaces_full.shape)
        # self._kspaces_full = self._kspaces_full + noise

        # Find noise std
        # true_imgs = np.expand_dims(self._imgs, axis=1) * np.expand_dims(self._coils, axis=0)
        # true_ffts = fft2(true_imgs, norm='ortho', axes=(-2, -1))
        # err = self._kspaces_full - true_ffts
        # print(err.var())
        
        self._kspaces = self._ksampler(self._kspaces_full)
    
    @property
    def imgs(self) -> List[ndarray]:
        return self._imgs

    @property
    def kspaces(self) -> List[ndarray]:
        return self._kspaces

    @property
    def kmasks(self) -> List[ndarray]:
        return self._ksampler.masks

    @property
    def coil_sens(self) -> List[ndarray]:
        return self._coils

    @property
    def img_size(self) -> Tuple[int]:
        return self._imgs.shape[-2:]

    @property
    def names(self) -> List[str]:
        return self._names

