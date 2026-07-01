from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from scenedino.models.backbones.dino.dinov2_module import DINOv2Module


_DA_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class FrozenDepthAnythingV2(nn.Module):
    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        encoder: str = "vitl",
        input_size: tuple[int, int] = (196, 644),
        normalize_depth: bool = True,
        use_log_depth: bool = True,
    ):
        super().__init__()
        repo_path = Path(repo_path)
        checkpoint_path = Path(checkpoint_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Depth-Anything-V2 repo not found: {repo_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Depth-Anything-V2 checkpoint not found: {checkpoint_path}")
        if encoder not in _DA_MODEL_CONFIGS:
            raise ValueError(f"Unsupported Depth-Anything-V2 encoder: {encoder}")

        repo_path_str = repo_path.as_posix()
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

        from depth_anything_v2.dpt import DepthAnythingV2

        self.model = DepthAnythingV2(**_DA_MODEL_CONFIGS[encoder])
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        if len(input_size) != 2:
            raise ValueError("Depth-Anything-V2 input_size must be [height, width]")
        self.input_size = tuple(int(v) for v in input_size)
        self.normalize_depth = normalize_depth
        self.use_log_depth = use_log_depth
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        # SceneDINO image tensors are in [-1, 1]. Depth-Anything-V2 expects
        # ImageNet-normalized RGB in [0, 1].
        images = (images * 0.5 + 0.5).clamp(0, 1)
        images = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        images = (images - self.mean.to(images.device, images.dtype)) / self.std.to(images.device, images.dtype)

        with torch.no_grad():
            depth = self.model(images.float()).unsqueeze(1)
            depth = F.interpolate(depth, size=target_size, mode="bilinear", align_corners=False)

        depth = depth.clamp_min(1e-6)
        if self.use_log_depth:
            depth = torch.log(depth)
        if self.normalize_depth:
            mean = depth.mean(dim=(-2, -1), keepdim=True)
            std = depth.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            depth = (depth - mean) / std
        return depth


class DINOAndDepthAnythingAdapter(nn.Module):
    """SceneDINO DINO features with a frozen Depth-Anything depth prior.

    - ground_truth=True returns frozen DINO features used as the reconstruction target.
    - ground_truth=False returns the original SceneDINO DINO decoder features.
    - get_depth_prior returns frozen Depth-Anything depth aligned to a target grid.
    """

    def __init__(self, dino: dict, depth_anything: dict):
        super().__init__()
        self.dino = DINOv2Module.from_conf(dino)
        self.depth_anything = FrozenDepthAnythingV2(
            repo_path=depth_anything.get("repo_path", "/mnt/sdc/wy/code/Depth-Anything-V2"),
            checkpoint_path=depth_anything["checkpoint_path"],
            encoder=depth_anything.get("encoder", "vitl"),
            input_size=tuple(depth_anything.get("input_size", (196, 644))),
            normalize_depth=depth_anything.get("normalize_depth", True),
            use_log_depth=depth_anything.get("use_log_depth", True),
        )

        self.extra_outs = 0
        self.scales = [0]
        self.gt_encoder = self.dino.gt_encoder
        self.patch_size = self.dino.gt_encoder.patch_size
        self.image_size = tuple(self.dino.gt_encoder.image_size)
        self.latent_size = self.dino.latent_size
        self.dino_pca_dim = self.dino.dino_pca_dim

        self.depth_proj = nn.Conv2d(1, self.latent_size, kernel_size=1)
        nn.init.normal_(self.depth_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.depth_proj.bias)
        self.depth_alpha = nn.Parameter(torch.tensor(0.0))
        self.visualization = self.dino.visualization

    def train(self, mode: bool = True):
        super().train(mode)
        self.depth_anything.eval()
        return self

    def forward(self, x: torch.Tensor, ground_truth: bool = False):
        if ground_truth:
            return self.dino(x, ground_truth=True)

        return self.dino(x, ground_truth=False)

    def get_depth_prior(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        return self.depth_anything(x, target_size=target_size)

    def downsample(self, x, mode="patch"):
        return self.dino.downsample(x, mode)

    def expand_dim(self, features):
        return self.dino.expand_dim(features)

    def fit_visualization(self, features, refit=True):
        return self.dino.fit_visualization(features, refit)

    def transform_visualization(self, features, norm=False, from_dim=0):
        return self.dino.transform_visualization(features, norm, from_dim)

    def fit_transform_kmeans_visualization(self, features):
        return self.dino.fit_transform_kmeans_visualization(features)

    @classmethod
    def from_conf(cls, conf):
        return cls(dino=conf["dino"], depth_anything=conf["depth_anything"])
