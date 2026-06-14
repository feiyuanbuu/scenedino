from pathlib import Path
from typing import Any

import torch
from ignite.handlers import Checkpoint


def load_checkpoint(ckpt_path: Path, to_save: dict[str, Any], strict: bool = False):
    assert ckpt_path.exists(), f"__Checkpoint '{str(ckpt_path)}' is not found"
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    model = to_save["model"]
    if not (isinstance(checkpoint, dict) and "model" in checkpoint):
        model_checkpoint = checkpoint
    else:
        model_checkpoint = checkpoint["model"]

    if not strict:
        model_state = model.state_dict()
        model_checkpoint = {
            key: value
            for key, value in model_checkpoint.items()
            if key in model_state and model_state[key].shape == value.shape
        }

    to_save = {"model": model}
    checkpoint = {"model": model_checkpoint}
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint, strict=strict)
