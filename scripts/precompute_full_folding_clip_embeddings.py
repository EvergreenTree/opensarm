import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.common.datasets.rm_lerobot_dataset import FullFoldingSarmDataset
from models.clip_encoder import FrozenCLIPEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute frozen CLIP image embeddings for lerobot/full_folding.")
    parser.add_argument("--root", default="/mnt/sarm-data/hf/lerobot/full_folding")
    parser.add_argument("--repo-id", default="lerobot/full_folding")
    parser.add_argument("--output", default="/mnt/sarm-data/hf/lerobot/full_folding/cache/clip_image_embeddings.npy")
    parser.add_argument("--done-output", default="/mnt/sarm-data/hf/lerobot/full_folding/cache/clip_image_embeddings.done.npy")
    parser.add_argument("--clip-ckpt", default="openai/clip-vit-base-patch32")
    parser.add_argument("--camera", action="append", dest="cameras", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--episode-limit", type=int, default=None)
    return parser.parse_args()


def encode_frames(clip_encoder, frames, batch_size):
    outputs = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size].to(clip_encoder.device)
        outputs.append(clip_encoder.encode_image(batch).detach().cpu())
    return torch.cat(outputs, dim=0)


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    done_output = Path(args.done_output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cameras = args.cameras or ["observation.images.base"]
    dataset = FullFoldingSarmDataset(
        repo_id=args.repo_id,
        root=root,
        image_names=cameras,
        max_rewind_steps=0,
        video_backend="pyav",
        episode_limit=args.episode_limit,
    )
    num_frames = len(dataset.hf_dataset)
    embed_dim = 512
    dtype = np.float16 if args.dtype == "float16" else np.float32

    if output.exists():
        embeddings = np.load(output, mmap_mode="r+")
    else:
        embeddings = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=dtype,
            shape=(num_frames, len(cameras), embed_dim),
        )

    if done_output.exists():
        done = np.load(done_output, mmap_mode="r+")
    else:
        done = np.lib.format.open_memmap(
            done_output,
            mode="w+",
            dtype=np.bool_,
            shape=(num_frames, len(cameras)),
        )
        done[:] = False
        done.flush()

    meta_path = output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "root": str(root),
                "clip_ckpt": args.clip_ckpt,
                "cameras": cameras,
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
            },
            indent=2,
        )
    )

    clip_encoder = FrozenCLIPEncoder(args.clip_ckpt, torch.device(args.device))
    episode_rows = dataset.episode_table.to_dict("records")
    progress = tqdm(episode_rows, desc="Episodes")
    for ep in progress:
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        indices = list(range(ep_start, ep_end))
        rows = dataset._rows(indices)
        timestamps = [float(ts) for ts in rows["timestamp"]]

        for camera_index, camera in enumerate(cameras):
            if bool(done[ep_start:ep_end, camera_index].all()):
                continue

            from_ts = float(ep[f"videos/{camera}/from_timestamp"])
            shifted_timestamps = [from_ts + ts for ts in timestamps]
            video_path = dataset._video_file_path(ep, camera)

            for offset in range(0, len(indices), args.decode_batch_size):
                chunk_indices = indices[offset : offset + args.decode_batch_size]
                if bool(done[chunk_indices, camera_index].all()):
                    continue
                chunk_timestamps = shifted_timestamps[offset : offset + args.decode_batch_size]
                frames = dataset._decode_video(ep, camera, [ts - from_ts for ts in chunk_timestamps])
                encoded = encode_frames(clip_encoder, frames, args.encode_batch_size).numpy().astype(dtype, copy=False)
                embeddings[chunk_indices, camera_index, :] = encoded
                done[chunk_indices, camera_index] = True

            embeddings.flush()
            done.flush()
            progress.set_postfix(video=video_path.name, done=f"{done[:, camera_index].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
