import ignite.distributed as idist
import hydra
from omegaconf import DictConfig, OmegaConf
import os

import torch

from scenedino.training.trainer import training as full_training
from scenedino.training.trainer_overfit import training as overfit_training
from scenedino.training.trainer_downstream import training as downstream_training


@hydra.main(version_base=None, config_path="configs", config_name="exp_kitti_360_DFT")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)

    os.environ.setdefault("NCCL_DEBUG", "INFO")
    # Single-node DDP does not need IB/RoCE. Some machines expose irdma/RoCE
    # devices that can crash NCCL during bootstrap, so default to the safer
    # socket path unless the user explicitly overrides it.
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.autograd.set_detect_anomaly(False)

    ## Set up for training with multi-GPUs
    backend = config.get("backend", None)
    nproc_per_node = config.get("nproc_per_node", None)
    with_amp = config.get("with_amp", False)
    spawn_kwargs = {}

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    training = globals()[
        config["training_type"]
    ]  ## the script will use the "bts_overfit" training function that's been imported from models.bts.trainer_overfit

    with idist.Parallel(
        backend=backend, **spawn_kwargs
    ) as parallel:  ## A distributed training context is created and the training function is run:
        parallel.run(training, config)


if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=1 python train.py -cn train_scenedino_kitti_360 dataset.is_preprocessed=false

# Single-GPU VGGT-Omega adapter training. The config defaults to backend=null,
# so Ignite will not open a distributed rendezvous port.
# CUDA_VISIBLE_DEVICES=1 python train.py -cn train_scenedino_vggt_omega_kitti_360 dataset.is_preprocessed=false
#
# Multi-GPU VGGT-Omega adapter training:
# CUDA_VISIBLE_DEVICES=1,2 python train.py -cn train_scenedino_vggt_omega_kitti_360 backend=nccl nproc_per_node=2 dataset.is_preprocessed=false


# /home/wy/code/scenedino_1/out/scenedino-pretrained/seg-best-dino
# CUDA_VISIBLE_DEVICES=1 python train.py \
#   -cn train_scenedino_vggt_omega_kitti_360 \
#   dataset.is_preprocessed=false \
#   training.from_pretrained=out/scenedino-pretrained/seg-best-dino/checkpoint.pt\
#   training.num_epochs=10
