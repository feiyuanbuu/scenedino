#!/usr/bin/env python3
"""Offline Depth-Anything-V2 cache for KITTI-360 images.

This script precomputes the frozen DA depth branch used by
`DINOAndDepthAnythingAdapter` so training can later read cached tensors instead
of running Depth-Anything online every iteration.

It saves one `.pt` file per input image with:
    depth:   float16/float32 tensor, shape [H, W]
    feature: float16/float32 tensor, shape [1, H_patch, W_patch]
    source:  original image path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm


DA_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_hw(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected H,W or HxW")
    return int(parts[0]), int(parts[1])


def iter_kitti360_images(
    kitti_root: Path,
    sequences: set[str] | None,
    cameras: set[str],
    folders: set[str] | None,
) -> Iterable[Path]:
    raw_root = kitti_root / "data_2d_raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"KITTI-360 raw image root not found: {raw_root}")

    for seq_dir in sorted(raw_root.glob("*_sync")):
        if sequences is not None and seq_dir.name not in sequences:
            continue
        for camera in sorted(cameras):
            camera_dir = seq_dir / camera
            if not camera_dir.exists():
                continue
            data_dirs = [p for p in camera_dir.iterdir() if p.is_dir()]
            if folders is not None:
                data_dirs = [p for p in data_dirs if p.name in folders]
            else:
                data_dirs = [p for p in data_dirs if p.name.startswith("data")]
            for data_dir in sorted(data_dirs):
                yield from sorted(data_dir.glob("*.png"))


def output_path_for(image_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = image_path.relative_to(input_root)
    return output_root / rel.with_suffix(".pt")


def load_depth_anything(repo_path: Path, checkpoint_path: Path, encoder: str, device: torch.device):
    repo_path = repo_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Depth-Anything-V2 repo not found: {repo_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Depth-Anything checkpoint not found: {checkpoint_path}")
    if encoder not in DA_MODEL_CONFIGS:
        raise ValueError(f"Unsupported encoder '{encoder}', choose from {sorted(DA_MODEL_CONFIGS)}")

    repo_str = repo_path.as_posix()
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**DA_MODEL_CONFIGS[encoder])
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def preprocess_images(images_bgr: list, input_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    tensors = []
    for image_bgr in images_bgr:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(image)
    batch = torch.stack(tensors, dim=0).to(device)
    batch = F.interpolate(batch, size=input_size, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def normalize_depth(depth: torch.Tensor, use_log_depth: bool, normalize: bool) -> torch.Tensor:
    depth = depth.clamp_min(1e-6)
    if use_log_depth:
        depth = torch.log(depth)
    if normalize:
        mean = depth.mean(dim=(-2, -1), keepdim=True)
        std = depth.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        depth = (depth - mean) / std
    return depth


def save_batch(
    model,
    image_paths: list[Path],
    output_paths: list[Path],
    args,
    device: torch.device,
):
    images_bgr = []
    for image_path in image_paths:
        image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        images_bgr.append(image)

    network_input = preprocess_images(images_bgr, args.da_input_size, device)
    with torch.no_grad():
        depth = model(network_input).unsqueeze(1)
        depth = F.interpolate(depth, size=args.image_size, mode="bilinear", align_corners=False)
        depth = normalize_depth(depth, args.use_log_depth, args.normalize_depth)
        feature = F.interpolate(depth, size=args.feature_size, mode="bilinear", align_corners=False)

    if args.dtype == "float16":
        depth = depth.half()
        feature = feature.half()
    else:
        depth = depth.float()
        feature = feature.float()

    depth = depth.cpu()
    feature = feature.cpu()

    for i, output_path in enumerate(output_paths):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "depth": depth[i, 0].contiguous(),
            "feature": feature[i].contiguous(),
            "source": image_paths[i].as_posix(),
            "image_size": tuple(args.image_size),
            "feature_size": tuple(args.feature_size),
            "da_input_size": tuple(args.da_input_size),
            "normalize_depth": args.normalize_depth,
            "use_log_depth": args.use_log_depth,
            "encoder": args.encoder,
        }
        torch.save(payload, output_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute Depth-Anything-V2 cache for KITTI-360.")
    parser.add_argument("--kitti-root", type=Path, default=Path("/mnt/sdc/wy/KITTI-360"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/sdc/wy/KITTI-360/da_depth_cache_vitl_192x640_24x80"))
    parser.add_argument("--da-repo", type=Path, default=Path("/mnt/sdc/wy/code/Depth-Anything-V2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/sdc/wy/code/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"))
    parser.add_argument("--encoder", choices=sorted(DA_MODEL_CONFIGS), default="vitl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=parse_hw, default=(192, 640), help="saved depth H,W")
    parser.add_argument("--feature-size", type=parse_hw, default=(24, 80), help="cached feature H,W")
    parser.add_argument("--da-input-size", type=parse_hw, default=(196, 644), help="Depth-Anything input H,W; must be divisible by 14")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--normalize-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-log-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cameras", default="image_00,image_01,image_02,image_03")
    parser.add_argument("--folders", default="data_rect,data_rgb", help="comma list, or 'all'")
    parser.add_argument("--sequences", default="", help="comma list of sequence names; empty means all")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    args.kitti_root = args.kitti_root.resolve()
    args.output_root = args.output_root.resolve()
    input_root = args.kitti_root / "data_2d_raw"

    sequences = {s.strip() for s in args.sequences.split(",") if s.strip()} or None
    cameras = {c.strip() for c in args.cameras.split(",") if c.strip()}
    folders = None if args.folders.strip().lower() == "all" else {f.strip() for f in args.folders.split(",") if f.strip()}

    image_paths = list(iter_kitti360_images(args.kitti_root, sequences, cameras, folders))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    todo: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        out_path = output_path_for(image_path, input_root, args.output_root)
        if args.overwrite or not out_path.exists():
            todo.append((image_path, out_path))

    print(f"Images found: {len(image_paths)}")
    print(f"Images to process: {len(todo)}")
    print(f"Output root: {args.output_root}")
    if args.dry_run:
        for image_path, out_path in todo[:10]:
            print(f"{image_path} -> {out_path}")
        return

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = load_depth_anything(args.da_repo, args.checkpoint, args.encoder, device)

    batch_images: list[Path] = []
    batch_outputs: list[Path] = []
    progress = tqdm(todo, desc="Precomputing DA depth", unit="img")
    for image_path, output_path in progress:
        batch_images.append(image_path)
        batch_outputs.append(output_path)
        if len(batch_images) == args.batch_size:
            save_batch(model, batch_images, batch_outputs, args, device)
            batch_images.clear()
            batch_outputs.clear()
    if batch_images:
        save_batch(model, batch_images, batch_outputs, args, device)

    print("Done.")


if __name__ == "__main__":
    main()
