import ignite.distributed as idist
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from scenedino.evaluation.unified_evaluator import evaluation


@hydra.main(version_base=None, config_path="configs", config_name="evaluation")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)

    os.environ["NCCL_DEBUG"] = "INFO"
    # torch.autograd.set_detect_anomaly(True)

    backend = config.get("backend", None)
    nproc_per_node = config.get("nproc_per_node", None)
    with_amp = config.get("with_amp", False)
    spawn_kwargs = {}

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(evaluation, config)


if __name__ == "__main__":
    main()



# CUDA_VISIBLE_DEVICES=1 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   checkpoint=out/features-paper/scenedino-vggt-omega-kitti-360-sscbench-lora-exp-002/training_checkpoint_400000.pt


# 6_21
# CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   'checkpoint="out/features-paper/scenedino-vggt-omega-kitti-360-sscbench-lora-exp-002-test/dino_cos_sim_best_model_2_dino_cos_sim=0.9992.pt"'

# /mnt/sdc/wy/code/scenedino_1/out/features-paper/scenedino-vggt-omega-kitti-360-sscbench-lora-exp-002-test/dino_cos_sim_best_model_2_dino_cos_sim=0.9992.pt



# 评估vggt-omega 完整训练版本
# /mnt/sdc/wy/code/scenedino_1/out/ssc-paper/semantic-vggt-omega-kitti-360-sscbench-lora-exp-002-test/stego_cluster_weighted_miou_best_model_1_stego_cluster_weighted_miou=0.0217.pt

# CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   'checkpoint="out/ssc-paper/semantic-vggt-omega-kitti-360-sscbench-lora-exp-002-test/stego_cluster_weighted_miou_best_model_1_stego_cluster_weighted_miou=0.0217.pt"'


# 评估ss_bench 完整训练版本
# /mnt/sdc/wy/code/scenedino_1/out/ssc-paper/ssc-kitti-360-sscbench/stego_cluster_weighted_miou_best_model_1_stego_cluster_weighted_miou=0.3550.pt

# CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   'checkpoint="out/ssc-paper/ssc-kitti-360-sscbench/stego_cluster_weighted_miou_best_model_1_stego_cluster_weighted_miou=0.3550.pt"'


# 评估官方checkpoint_best_dino

# CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   'checkpoint="out/scenedino-pretrained/seg-best-dino/checkpoint.pt"'


# 评估10000pt
# /mnt/sdc/wy/code/scenedino_1/out/features-paper/scenedino-vggt-omega-kitti-360-sscbench-lora-exp-002-test-1/training_checkpoint_10000_save.pt

# CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   'checkpoint="out/features-paper/scenedino-vggt-omega-kitti-360-sscbench-lora-exp-002-test-1/training_checkpoint_10000_save.pt"'



#  CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   'checkpoint="out/features-paper/scenedino-vggt-omega-kitti-360-sscbench-lora-exp-002-test-1/training_checkpoint_20000.pt"'




# /mnt/sdc/wy/code/scenedino_1/out/ssc-paper/semantic-vggt-omega-kitti-360-sscbench-lora-exp-002-test_6_22/stego_cluster_weighted_miou_best_model_16_stego_cluster_weighted_miou=0.0907.pt

#  CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   'checkpoint="out/ssc-paper/semantic-vggt-omega-kitti-360-sscbench-lora-exp-002-test_6_22/stego_cluster_weighted_miou_best_model_16_stego_cluster_weighted_miou=0.0907.pt"'








# 6_24评估
# /mnt/sdc/wy/code/scenedino_1/out/ssc-paper/semantic-vggt-omega-kitti-360-sscbench-no-lora-fixed-features/stego_cluster_weighted_miou_best_model_1_stego_cluster_weighted_miou=0.0217.pt

#  CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_kitti_360 \
#   dataset.is_preprocessed=false \
#   'checkpoint="out/ssc-paper/semantic-vggt-omega-kitti-360-sscbench-no-lora-fixed-features/stego_cluster_weighted_miou_best_model_1_stego_cluster_weighted_miou=0.0217.pt"'



# 6_28评估 da depth 相加
# cd /mnt/sdc/wy/code/scenedino_1
# CUDA_VISIBLE_DEVICES=2 /home/wy/anaconda3/envs/scenedino/bin/python eval.py \
#   -cn evaluate_semantic_dino_da_kitti_360


# 6_29评估 da depth prior
# cd /mnt/sdc/wy/code/scenedino_1
# CUDA_VISIBLE_DEVICES=2 python eval.py \
#   -cn evaluate_semantic_dino_da_depth_prior_kitti_360