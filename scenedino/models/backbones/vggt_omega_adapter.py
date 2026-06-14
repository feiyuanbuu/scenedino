from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from scenedino.models.backbones.dino.downsampler import BilinearDownsampler, PatchSalienceDownsampler
from scenedino.models.backbones.lora import inject_lora, lora_state_dict, mark_only_lora_trainable
from scenedino.models.backbones.dino.visualization import VisualizationModule
from vggt_omega.models import VGGTOmega


class VGGTOmegaSceneDINOAdapter(nn.Module):
    """Expose VGGT-Omega multi-view features through SceneDINO's encoder API."""

    multiview_input = True

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        adapter_dim: int,
        dino_pca_dim: int,
        image_size: tuple[int, int],
        checkpoint_path: str | None = None,
        load_checkpoint: bool = True,
        freeze_vggt: bool = True,
        use_patch_tokens: bool = True,
        use_register_tokens: bool = True,
        use_depth: bool = True,
        use_depth_conf: bool = True,
        enable_camera_head: bool = True,
        checkpoint_strict: bool = True,
        lora: dict | None = None,
        downsampler_arch: str = "featup",
        normalize_features: bool = True,
    ):
        super().__init__()

        self.vggt = VGGTOmega(
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera_head,
            enable_depth=use_depth or use_depth_conf,
            enable_alignment=False,
        )
        if load_checkpoint and checkpoint_path:
            self._load_checkpoint(Path(checkpoint_path), strict=checkpoint_strict)

        self.lora_enabled = bool(lora and lora.get("enabled", False))
        self.lora_target_modules = []
        if self.lora_enabled:
            self.lora_target_modules = inject_lora(
                self.vggt,
                target_keywords=lora.get("target_modules", ["qkv", "proj", "fc1", "fc2"]),
                rank=lora.get("rank", 8),
                alpha=lora.get("alpha", 16),
                dropout=lora.get("dropout", 0.0),
            )
            mark_only_lora_trainable(self.vggt)
            if lora.get("checkpoint", None):
                self._load_lora_checkpoint(Path(lora["checkpoint"]), strict=lora.get("checkpoint_strict", False))

        self.freeze_vggt = freeze_vggt
        if freeze_vggt and not self.lora_enabled:
            for param in self.vggt.parameters():
                param.requires_grad_(False)

        self.patch_size = patch_size
        self.image_size = tuple(image_size)
        self.latent_size = dino_pca_dim
        self.dino_pca_dim = dino_pca_dim
        self.extra_outs = 0
        self.scales = [0]
        self.normalize_features = normalize_features
        self.use_patch_tokens = use_patch_tokens
        self.use_register_tokens = use_register_tokens
        self.use_depth = use_depth
        self.use_depth_conf = use_depth_conf

        input_dim = 0
        if use_patch_tokens:
            input_dim += 2 * embed_dim
        if use_register_tokens:
            input_dim += 2 * embed_dim
        if use_depth:
            input_dim += 1
        if use_depth_conf:
            input_dim += 1
        if input_dim == 0:
            raise ValueError("At least one VGGT-Omega feature source must be enabled.")

        self.adapter = nn.Sequential(
            nn.Conv2d(input_dim, adapter_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(adapter_dim, adapter_dim, kernel_size=1),
        )
        self.output_projection = nn.Linear(adapter_dim, dino_pca_dim) if adapter_dim != dino_pca_dim else nn.Identity()

        if downsampler_arch == "featup":
            self.downsampler = PatchSalienceDownsampler(dino_pca_dim, patch_size=patch_size, normalize_features=True)
        elif downsampler_arch == "bilinear":
            self.downsampler = BilinearDownsampler(patch_size=patch_size)
        elif downsampler_arch in (None, "none"):
            self.downsampler = None
        else:
            raise NotImplementedError(f"Unsupported VGGT-Omega downsampler: {downsampler_arch}")

        self.visualization = VisualizationModule(dino_pca_dim)

    def _load_checkpoint(self, checkpoint_path: Path, strict: bool):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.vggt.load_state_dict(state_dict, strict=strict)

    def _load_lora_checkpoint(self, checkpoint_path: Path, strict: bool):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"VGGT-Omega LoRA checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("lora_state_dict", checkpoint)
        self.vggt.load_state_dict(state_dict, strict=strict)

    def forward(self, images, ground_truth=False):
        if images.dim() == 4:
            raise ValueError("VGGTOmegaSceneDINOAdapter expects [B, V, C, H, W] images.")

        images = images * 0.5 + 0.5
        if self.freeze_vggt and not self.lora_enabled:
            with torch.no_grad():
                features = self._extract_features(images)
        else:
            features = self._extract_features(images)

        if ground_truth:
            return [features.detach()]
        return [features]

    def _extract_features(self, images):
        batch_size, num_views, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size

        if self.freeze_vggt and not self.lora_enabled:
            self.vggt.eval()

        aggregated_tokens, patch_token_start = self.vggt.aggregator(images)
        final_tokens = aggregated_tokens[-1]
        if final_tokens is None:
            raise ValueError("VGGT-Omega aggregator did not return final tokens.")

        feature_parts = []
        if self.use_patch_tokens:
            patch_tokens = final_tokens[:, :, patch_token_start:]
            patch_tokens = patch_tokens.reshape(batch_size * num_views, patch_h, patch_w, -1).permute(0, 3, 1, 2)
            feature_parts.append(patch_tokens.float())

        if self.use_register_tokens:
            register_tokens = final_tokens[:, :, 1:patch_token_start].mean(dim=2)
            register_tokens = register_tokens.reshape(batch_size * num_views, -1, 1, 1)
            register_tokens = register_tokens.expand(-1, -1, patch_h, patch_w)
            feature_parts.append(register_tokens.float())

        dense_predictions = None
        if self.use_depth or self.use_depth_conf:
            dense_predictions = self._predict_dense(aggregated_tokens, images, patch_token_start)

        if self.use_depth:
            depth, _ = dense_predictions
            depth = depth.reshape(batch_size * num_views, height, width, -1).permute(0, 3, 1, 2)
            depth = F.interpolate(depth.float(), size=(patch_h, patch_w), mode="bilinear", align_corners=False)
            feature_parts.append(torch.log(depth.clamp_min(1e-4)))

        if self.use_depth_conf:
            _, depth_conf = dense_predictions
            depth_conf = depth_conf.reshape(batch_size * num_views, 1, height, width)
            depth_conf = F.interpolate(depth_conf.float(), size=(patch_h, patch_w), mode="bilinear", align_corners=False)
            feature_parts.append(torch.log(depth_conf.clamp_min(1e-4)))

        adapted = self.adapter(torch.cat(feature_parts, dim=1))
        adapted = adapted.permute(0, 2, 3, 1)
        adapted = self.output_projection(adapted)
        if self.normalize_features:
            adapted = F.normalize(adapted, dim=-1)
        adapted = adapted.permute(0, 3, 1, 2).contiguous()
        return adapted

    def _predict_dense(self, aggregated_tokens, images, patch_token_start):
        if self.vggt.dense_head is None:
            raise ValueError("VGGT-Omega dense head is disabled.")
        return self.vggt.dense_head(aggregated_tokens, images, patch_token_start)

    def downsample(self, x, mode="patch"):
        if self.downsampler is None:
            return None
        return self.downsampler(x, mode)

    def expand_dim(self, features):
        if self.normalize_features:
            return F.normalize(features, dim=-1)
        return features

    def fit_visualization(self, features, refit=True):
        return self.visualization.fit_pca(features, refit)

    def transform_visualization(self, features, norm=False, from_dim=0):
        return self.visualization.transform_pca(features, norm, from_dim)

    def fit_transform_kmeans_visualization(self, features):
        return self.visualization.fit_transform_kmeans_batch(features)

    def lora_state_dict(self):
        return lora_state_dict(self.vggt)

    @classmethod
    def from_conf(cls, conf):
        return cls(
            patch_size=conf.get("patch_size", 16),
            embed_dim=conf.get("embed_dim", 1024),
            adapter_dim=conf.get("adapter_dim", conf.get("decoder_out_dim", 256)),
            dino_pca_dim=conf.get("dino_pca_dim", 256),
            image_size=tuple(conf.get("image_size", (192, 640))),
            checkpoint_path=conf.get("checkpoint_path", None),
            load_checkpoint=conf.get("load_checkpoint", True),
            freeze_vggt=conf.get("freeze_vggt", True),
            use_patch_tokens=conf.get("use_patch_tokens", True),
            use_register_tokens=conf.get("use_register_tokens", True),
            use_depth=conf.get("use_depth", True),
            use_depth_conf=conf.get("use_depth_conf", True),
            enable_camera_head=conf.get("enable_camera_head", True),
            checkpoint_strict=conf.get("checkpoint_strict", True),
            lora=conf.get("lora", None),
            downsampler_arch=conf.get("downsampler_arch", "featup"),
            normalize_features=conf.get("normalize_features", True),
        )
