#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download lerobot/full_folding into a local LeRobot root.")
    parser.add_argument("--repo-id", default="lerobot/full_folding")
    parser.add_argument("--local-dir", default="/mnt/sarm-data/hf/lerobot/full_folding")
    parser.add_argument("--base-camera-only", action="store_true")
    args = parser.parse_args()

    allow_patterns = ["meta/**", "data/**", "sarm_progress.parquet", "README.md"]
    if args.base_camera_only:
        allow_patterns.append("videos/observation.images.base/**")
    else:
        allow_patterns.append("videos/**")

    local_dir = Path(args.local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    print(f"Downloaded {args.repo_id} to {local_dir}")


if __name__ == "__main__":
    main()
