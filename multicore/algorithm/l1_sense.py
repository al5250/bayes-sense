from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from numpy import ndarray
from numpy.fft import fftshift
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
from torch.fft import ifftn, fftn
import sigpy.mri as mr
import sigpy as sp

import pdb
import time
from hydra.utils import instantiate

from multicore.dataset import MRIDataset
from multicore.logger import Logger
from multicore.algorithm.algorithm import ReconstructionAlgorithm
from multicore.projections import (
    Projection,
    UndersampledFourier2D,
    MultiCoilProjection,
    Sequential
)
from multicore.utils import conj_grad, get_torch_device, get_torch_dtypes
from multicore.metric import RootMeanSquareError


class L1SENSE(ReconstructionAlgorithm):

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
       
    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> ndarray:
        kspaces = np.array(dataset.kspaces)
        sens = np.array(dataset.coil_sens)
        kmasks = np.array(dataset.kmasks)
        
        shift1 = np.tile(np.array([+1, -1]), kspaces.shape[-2] // 2).reshape(-1, 1) 
        shift2 = np.tile(np.array([+1, -1]), kspaces.shape[-1] // 2)

        recons = []
        for idx in range(len(dataset.imgs)):
            kspace = np.fft.fftshift(shift1 * (shift2 * kspaces[idx]), axes=(-2, -1))
            recon = mr.app.L1WaveletRecon(kspace, sens, **self.kwargs).run()
            recons.append(recon)
        recons = np.array(recons)
    
        logger.log_imgs(
            f"{str(self)}/Masks", 
            np.fft.ifftshift(kmasks.astype(np.float32), axes=(-2, -1))
        )

        imgs = np.array(dataset.imgs)

        metric = RootMeanSquareError(percentage=True)
        rmses = metric(recons, imgs)
        logger.log_vals(
            f"{str(self)}/{str(metric)}", dict(zip(dataset.names, rmses)), 0
        )
        return recons