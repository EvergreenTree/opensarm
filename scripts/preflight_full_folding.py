#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch

from lerobot.common.datasets.rm_lerobot_dataset import FullFoldingSarmDataset
from utils.device_utils import resolve_torch_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the full_folding OpenSARM adapter.")
    parser.add_argument("--root", default="/mnt/sarm-data/hf/lerobot/full_folding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()

    device = resolve_torch_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_name={torch.cuda.get_device_name(0)}")

    dataset = FullFoldingSarmDataset(root=args.root, episode_limit=args.episodes)
    sample = dataset[0]
    image = sample["observation.images.base"]
    state = sample["state"]
    targets = sample["targets"]
    print(f"episodes={len(dataset.episode_table)} frames={len(dataset)}")
    print(f"image_shape={tuple(image.shape)} dtype={image.dtype} min={image.min().item():.4f} max={image.max().item():.4f}")
    print(f"state_shape={tuple(state.shape)} dtype={state.dtype}")
    print(f"targets_shape={tuple(targets.shape)} min={targets.min().item():.4f} max={targets.max().item():.4f}")
    if not torch.isfinite(targets).all():
        raise RuntimeError("Targets contain non-finite values.")
    if targets.min().item() < 0 or targets.max().item() > 1:
        raise RuntimeError("Targets are outside [0, 1].")


if __name__ == "__main__":
    main()
