from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np
import pickle

from multicore.dataset import MRIDataset
from multicore.algorithm import ReconstructionAlgorithm
from multicore.metric import Metric


@hydra.main(config_path='../configs/experiment', config_name='experiment')
def experiment(config: DictConfig) -> None:
    dataset = instantiate(config.dataset)
    algorithm = instantiate(config.algorithm)
    metric = instantiate(config.metric)
    logger = instantiate(config.logger)

    logger.log_imgs('Original', dataset.imgs)

    alg_imgs = algorithm.reconstruct(dataset, logger)
    
    mask = np.sum(dataset.coil_sens.conj() * dataset.coil_sens, axis=0).real > 0
    val = metric(alg_imgs[..., mask], dataset.imgs[..., mask])

    # val = metric(alg_imgs, dataset.imgs)

    logger.log_vals(f'Final/{str(metric)}', {'Combined': val})
    logger.log_imgs(str(algorithm), alg_imgs)

    error_maps = np.abs(np.array(dataset.imgs) - alg_imgs)
    logger.log_imgs(f"{str(algorithm)}/ErrorMaps", error_maps)
    
    accel = np.array(dataset.kmasks).size / np.sum(dataset.kmasks)
    hparam_dict = {k: v for k, v in config.algorithm.items() if type(v) in [bool, float, int, str]}
    logger.log_hparams(
        hparam_dict={**{'accel': accel}, **hparam_dict},
        metric_dict={str(metric): val},
    )

    if config.dump_file is not None:
        pickle.dump({'original': dataset.imgs, 'algorithm': alg_imgs}, open(config.dump_file, 'wb'))

    logger.close()


if __name__ == "__main__":
    experiment()
