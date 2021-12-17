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
import pickle

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


class MultiSENSE(ReconstructionAlgorithm):

    def __init__(
        self,
        sparse_proj: Projection,
        num_em_iters: int = 40,
        num_probes: int = 30,
        cg_tol: float = 1e-5,
        max_cg_iters: int = 1000,
        alpha0: float = 1e10,
        alpha_init: float = 1.,
        real_input: bool = False,
        log_imgs_interval: Optional[int] = 5,
        log_rmses: bool = False,
        log_final_variances: bool = False,
        device: Optional[str] = None,
        precision: str = 'double',
        keep_observed_kspace: bool = False,
        alpha_size: Optional[Tuple[int]] = None
    ) -> None:
        # Set device and precision
        self.device = get_torch_device(device)
        dtype, self.cdtype = get_torch_dtypes(precision)
        torch.set_default_dtype(dtype)

        # Create sparse projection object
        self.sparse_proj = instantiate(sparse_proj)

        # Create Bernoulli sampler for probe vectors
        self.bernoulli = Bernoulli(probs=torch.tensor(0.5, device=self.device))

        # Set algorithm parameters
        self.num_em_iters = num_em_iters
        self.num_probes = num_probes
        self.cg_tol = cg_tol
        self.max_cg_iters = max_cg_iters
        self.alpha0 = alpha0
        self.alpha_init = alpha_init
        self.real_input = real_input
        self.keep_observed_kspace = keep_observed_kspace
        self.alpha_size = alpha_size

        # Set logging parameters
        self.log_imgs_interval = log_imgs_interval
        self.log_rmses = log_rmses
        self.log_final_variances = log_final_variances

    @torch.no_grad()
    def reconstruct(self, dataset: MRIDataset, logger: Logger) -> ndarray:
        kspaces = torch.tensor(dataset.kspaces, device=self.device, dtype=self.cdtype)
        sens = torch.tensor(dataset.coil_sens, device=self.device, dtype=self.cdtype)
        kmasks = torch.tensor(dataset.kmasks, device=self.device, dtype=torch.bool)
        kmasks = kmasks.unsqueeze(dim=1).expand_as(kspaces)
        # self.coil_mask = (torch.sum(sens.conj() * sens, dim=0) != 0)

        logger.log_imgs(
            f"{str(self)}/Masks", 
            np.fft.fftshift(np.array(dataset.kmasks).astype(np.float32), axes=(-2, -1))
        )

        fourier = UndersampledFourier2D(mask=kmasks)
        coil_proj = MultiCoilProjection(sens=sens, real_input=self.real_input)
        Phi = Sequential(
            projs=[self.sparse_proj, coil_proj, fourier],
            fwd_apply=[False, True, True]
        )

        alpha_size = tuple(self.alpha_size or dataset.img_size)
        if self.real_input:
            alpha_init = self.alpha_init * torch.ones(alpha_size, device=self.device)
        else:
            # alpha_init = self.alpha_init * torch.complex(
            #     torch.ones(dataset.img_size, device=self.device), 
            #     torch.ones(dataset.img_size, device=self.device)
            # )
            alpha_init = self.alpha_init * torch.ones(alpha_size, device=self.device)
        alpha, mu = self._fastem(kspaces, Phi, alpha_init, dataset, logger)
        imgs = self._compute_imgs(mu, Phi, kspaces)

        error_maps = np.abs(imgs - dataset.imgs)
        logger.log_imgs(f"{str(self)}/ErrorMaps", error_maps)

        if self.log_final_variances:
            variances = self._compute_variances(alpha, Phi, kspaces.size(dim=0), dataset.img_size)
            logger.log_imgs(f"{str(self)}/Variances", variances)

            stds = np.sqrt(variances)
            logger.log_imgs(f"{str(self)}/StanDevs", stds)

        return imgs

    def _fastem(
        self,
        kspaces: Tensor,
        Phi: Projection,
        alpha_init: Tensor,
        dataset: MRIDataset,
        logger: Logger
    ) -> Tuple[Tensor, Tensor]:
        alpha = alpha_init
        sparse_zfill = Phi.T(kspaces)
        mu, sigma_diag, cg_converge_iter = self._estep(alpha, Phi, sparse_zfill)
        err = (torch.norm(kspaces - Phi(mu.unsqueeze(dim=1))) / kspaces.numel()).cpu().item()
        self._log_iteration(0, mu, err, cg_converge_iter, dataset, logger, Phi, kspaces)

        for i in range(self.num_em_iters):
            alpha_new = self._mstep(mu, sigma_diag)

            mu_new, sigma_diag_new, cg_converge_iter = self._estep(
                alpha_new, Phi, sparse_zfill
            )

            if cg_converge_iter is None:
                break
            alpha, mu, sigma_diag = alpha_new, mu_new, sigma_diag_new
            err = (torch.norm(kspaces - Phi(mu.unsqueeze(dim=1))) / kspaces.numel()).cpu().item()
            self._log_iteration(i + 1, mu, err, cg_converge_iter, dataset, logger, Phi, kspaces)
        
        if self.log_final_variances:
            pickle.dump(
                {'sigma': sigma_diag.cpu().numpy(), 'alpha': alpha.cpu().numpy()}, 
                open('bayes_extras.p', 'wb')
            )

        return alpha, mu

    def _estep(
        self,
        alpha: Tensor,
        Phi: Projection,
        sparse_zfill: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[int]]:
        print(alpha.shape)
        N, _, H, W = sparse_zfill.size()
        sparse_zfill = sparse_zfill.unsqueeze(dim=0)
        probes = self._samp_probes((self.num_probes, N, 1, H, W))

        b = torch.cat([sparse_zfill, probes], dim=0)
        if self.real_input:
            A = lambda x: Phi.T(Phi(x)) + alpha / self.alpha0 * x
            x, converge_iter = conj_grad(
                A, b, dim=(-2, -1), max_iters=self.max_cg_iters, tol=self.cg_tol
            )
            mu = x[0]
            sigma_diag =  1 / self.alpha0 * (probes * x[1:]).mean(dim=0).clamp(min=0)
        else:

            # A = lambda x: torch.view_as_real(Phi.T(Phi(torch.view_as_complex(x)))) + \
            #     torch.view_as_real(alpha) / self.alpha0 * x 
            A = lambda x: torch.view_as_real(Phi.T(Phi(torch.view_as_complex(x)))) + \
                alpha.unsqueeze(dim=-1) / self.alpha0 * x 
            x, converge_iter = conj_grad(
                A, torch.view_as_real(b), 
                dim=(-3, -2, -1), max_iters=self.max_cg_iters, tol=self.cg_tol
            )
            mu = torch.view_as_complex(x[0])
            sigma_diag = 1 / self.alpha0 * torch.view_as_complex(
                (torch.view_as_real(probes) * x[1:]).mean(dim=0).clamp(min=0)
            )

        mu = mu.squeeze(dim=1)
        sigma_diag = sigma_diag.squeeze(dim=1)
        return mu, sigma_diag, converge_iter

    def _mstep(self, mu: Tensor, sigma_diag: Tensor):
        if self.real_input:
            alpha = 1 / (mu ** 2 + sigma_diag).mean(dim=0)
        else:
            # alpha = torch.view_as_complex(
            #     1 / (torch.view_as_real(mu) ** 2 + torch.view_as_real(sigma_diag)).mean(dim=0)
            # )
            # alpha[~self.coil_mask] = 1e6 + 1e6j
            alpha = 2 / (mu.real ** 2 + mu.imag ** 2 + sigma_diag.real + sigma_diag.imag).mean(dim=0)
        return alpha

    def _log_iteration(
        self,
        _iter: int,
        mu: Tensor,
        err: float,
        cg_converge_iter: int,
        dataset: MRIDataset,
        logger: Logger,
        Phi: Projection,
        kspaces: Tensor
    ) -> None:
        logger.log_vals(
            f"{str(self)}/num_cg_iters", {'num_cg_iters': cg_converge_iter}, _iter
        )
        logger.log_vals(
            f"{str(self)}/recon_err", {'recon_err': err}, _iter
        )
        imgs = self._compute_imgs(mu, Phi, kspaces)
        sparsity = np.mean(np.abs(imgs) < 1e-6, axis=(-2, -1))
        logger.log_vals(
            f"{str(self)}/sparsity_level", dict(zip(dataset.names, sparsity)), _iter
        )
        if self.log_rmses:
            metric = RootMeanSquareError(percentage=True)
            rmses = metric(imgs, dataset.imgs)
            combined_metric = RootMeanSquareError(percentage=True, combine=True)

            mask = np.sum(dataset.coil_sens.conj() * dataset.coil_sens, axis=0).real > 0
            combined_rmse = combined_metric(imgs[..., mask], dataset.imgs[..., mask]).item()
            logger.log_vals(
                f"{str(self)}/{str(metric)}", dict(zip(dataset.names, rmses)), _iter
            )
            logger.log_vals(
                f"{str(self)}/{str(metric)}", {'Combined': combined_rmse}, _iter
            )
            print(f"Iter {_iter:2d} | RMSE {combined_rmse}")
            error_maps = np.abs(imgs - dataset.imgs)
            logger.log_imgs(f"{str(self)}/ErrorMaps", error_maps, _iter)
        if self.log_imgs_interval is not None and _iter % self.log_imgs_interval == 0:
            logger.log_imgs(f"{str(self)}/Reconstruction", imgs, _iter)
            logger.log_imgs(f"{str(self)}/SparseTransform", mu.cpu().numpy(), _iter)

    def _samp_probes(self, size: Tuple[int, ...]):
        if self.real_input:
            return 2 * self.bernoulli.sample(size) - 1
        else:
            return torch.view_as_complex(2 * self.bernoulli.sample(size + (2,)) - 1)

    def _compute_imgs(self, mu: Tensor, Phi: Projection, kspaces: Tensor) -> ndarray:
        if self.keep_observed_kspace:
            sens = Phi.projs[1].sens
            kmasks = Phi.projs[2].mask

            recon_kspaces = fftn(
                sens * self.sparse_proj.T(mu).unsqueeze(dim=1), dim=(-2, -1), norm='ortho'
            )
            final_kspaces = kmasks * kspaces + (~kmasks) * recon_kspaces
            
            final_coil_data = ifftn(final_kspaces, dim=(-2, -1), norm='ortho')
            norm = torch.sum(torch.abs(sens) ** 2, dim=-3)
            norm[norm == 0] = 1e-16
            imgs = torch.sum(sens.conj() * final_coil_data, dim=-3) / norm

        else:
            imgs = self.sparse_proj.T(mu)
        return imgs.cpu().numpy()

    def _compute_variances(
        self, alpha: Tensor, Phi: Projection, N: int, img_size: Tuple[int, int]
    ) -> ndarray:
        # H, W = alpha.size()
        # probes = self._samp_probes((self.num_probes, N, 1, H, W))
        # b = self.sparse_proj(probes)
        
        # A = lambda x: Phi.T(Phi(x)) + alpha / self.alpha0 * x
        # x, _ = conj_grad(
        #     A, b, dim=(-2, -1), max_iters=self.max_cg_iters, tol=self.cg_tol
        # )
        # x = self.sparse_proj.T(x)
        # variances =  1 / self.alpha0 * (probes * x).mean(dim=0).clamp(min=0)
        # variances = variances.squeeze(dim=1).cpu().numpy()

        probes = self._samp_probes((self.num_probes, N, 1, *img_size))
        b = self.sparse_proj(probes)
        
        A = lambda x: torch.view_as_real(Phi.T(Phi(torch.view_as_complex(x)))) + \
            alpha.unsqueeze(dim=-1) / self.alpha0 * x 
        x, _ = conj_grad(
            A, torch.view_as_real(b), 
            dim=(-3, -2, -1), max_iters=self.max_cg_iters, tol=self.cg_tol
        )
        x = torch.view_as_real(self.sparse_proj.T(torch.view_as_complex(x)))
        variances =  1 / self.alpha0 * (torch.view_as_real(probes) * x).mean(dim=0).clamp(min=0)
        # variances = torch.view_as_complex(variances).squeeze(dim=1).cpu().numpy()
        variances = torch.view_as_complex(variances).squeeze(dim=1).cpu().numpy()

        pickle.dump({'vars': variances}, open('variances.p', 'wb'))
        return variances
