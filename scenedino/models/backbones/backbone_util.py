from .image_encoder import ImageEncoder
from .monodepth2 import Monodepth2
from .spatial_encoder import SpatialEncoder
from scenedino.models.backbones.dino.dinov2_module import DINOv2Module
from scenedino.models.backbones.vggt_omega_adapter import VGGTOmegaSceneDINOAdapter
from scenedino.models.backbones.dino_da_adapter import DINOAndDepthAnythingAdapter


def make_backbone(conf, **kwargs):
    enc_type = conf.get("type", "monodepth2")  # spatial | global
    if enc_type == "monodepth2":
        net = Monodepth2.from_conf(conf, **kwargs)
    elif enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    elif enc_type == "dinov2":
        net = DINOv2Module.from_conf(conf, **kwargs)
    elif enc_type == "vggt_omega":
        net = VGGTOmegaSceneDINOAdapter.from_conf(conf, **kwargs)
    elif enc_type == "dino_da":
        net = DINOAndDepthAnythingAdapter.from_conf(conf)
    else:
        raise NotImplementedError(f"Unsupported encoder type: {enc_type}")
    return net
