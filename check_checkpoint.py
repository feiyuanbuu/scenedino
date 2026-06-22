# print_ckpt_keys.py
import torch
from collections.abc import Mapping, Sequence

# ckpt_path = "/mnt/sdc/wy/code/scenedino_1/out/features-paper/scenedino-vggt-omega-kitti-360-sscbench-lora-exp-002-test/training_checkpoint_200000.pt"
ckpt_path = "/mnt/sdc/wy/code/scenedino_1/out/ssc-paper/semantic-vggt-omega-kitti-360-sscbench-lora-exp-002-test_6_22/stego_cluster_weighted_miou_best_model_16_stego_cluster_weighted_miou=0.0907.pt"


def print_keys(obj, prefix=""):
    """
    递归打印 checkpoint 中的所有 key。
    如果遇到 tensor，会顺便打印 shape 和 dtype。
    """

    if isinstance(obj, Mapping):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)

            if isinstance(v, torch.Tensor):
                print(f"{name} | Tensor | shape={tuple(v.shape)} | dtype={v.dtype}")
            else:
                print(name)
                print_keys(v, name)

    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for i, v in enumerate(obj):
            name = f"{prefix}[{i}]"

            if isinstance(v, torch.Tensor):
                print(f"{name} | Tensor | shape={tuple(v.shape)} | dtype={v.dtype}")
            else:
                print(name)
                print_keys(v, name)


def main():
    print(f"Loading checkpoint from:\n{ckpt_path}\n")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print("普通 torch.load 失败，尝试 weights_only=False")
        print("原始错误：", e)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("\n========== Checkpoint Keys ==========\n")
    print_keys(ckpt)


if __name__ == "__main__":
    main()