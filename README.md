# OpenSARM Full Folding Training Fork

This fork adapts [xdofai/opensarm](https://github.com/xdofai/opensarm) for full-dataset SARM reward-model training on [`lerobot/full_folding`](https://huggingface.co/datasets/lerobot/full_folding).

The upstream project README, paper links, citation, and general SARM usage notes live in the original repository:

- Original README: https://github.com/xdofai/opensarm/blob/main/README.md
- Original repository: https://github.com/xdofai/opensarm

## What This Fork Adds

- A LeRobot v3 adapter for `lerobot/full_folding`.
- Mapping from `observation.state` to SARM state tensors.
- Base camera support through `observation.images.base`.
- `sarm_progress.parquet` targets using `progress_sparse`.
- CUDA/MPS/CPU device selection.
- Full-run configuration in `config/sarm_full_folding.yaml`.
- A frozen CLIP image-embedding cache path to avoid decoding MP4 video during every training step.
- A precompute script for the CLIP cache at `scripts/precompute_full_folding_clip_embeddings.py`.

## Expected Dataset Layout

Download or place `lerobot/full_folding` at:

```bash
/mnt/sarm-data/hf/lerobot/full_folding
```

The adapter expects the dataset root to contain:

```bash
data/
meta/
videos/
sarm_progress.parquet
```

The default config uses:

```yaml
cfg:
  general:
    dataset_format: lerobot_v3_full_folding
    dataset_root: /mnt/sarm-data/hf/lerobot/full_folding
    state_key: observation.state
    progress_key: progress_sparse
    camera_names: [observation.images.base]
```

## Install

Use `uv` from the repository root:

```bash
uv sync
```

For offline or server training, the current deployment uses:

```bash
export HF_HOME=/mnt/sarm-data/hf-home
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_MODE=offline
export UV_CACHE_DIR=/mnt/sarm-data/uv-cache
export UV_LINK_MODE=copy
```

## Precompute Frozen CLIP Image Embeddings

The full dataset is video-decode bound if training reads random MP4 frames directly. Since the CLIP image encoder is frozen, precompute image embeddings once:

```bash
uv run python scripts/precompute_full_folding_clip_embeddings.py \
  --root /mnt/sarm-data/hf/lerobot/full_folding \
  --output /mnt/sarm-data/hf/lerobot/full_folding/cache/clip_image_embeddings.npy \
  --done-output /mnt/sarm-data/hf/lerobot/full_folding/cache/clip_image_embeddings.done.npy
```

The script writes a resumable `.npy` memmap plus a `.done.npy` progress mask. The default cache is float16 with shape:

```bash
(num_frames, num_cameras, 512)
```

## Train With The Cache

After the cache exists, start full training:

```bash
uv run python train.py --config-name sarm_full_folding \
  cfg.general.image_embedding_cache=/mnt/sarm-data/hf/lerobot/full_folding/cache/clip_image_embeddings.npy
```

To run in tmux:

```bash
tmux new -d -s sarm_full_folding '
  cd /mnt/sarm-data/src/opensarm &&
  export HF_HOME=/mnt/sarm-data/hf-home &&
  export HF_HUB_ENABLE_HF_TRANSFER=1 &&
  export WANDB_MODE=offline &&
  export UV_CACHE_DIR=/mnt/sarm-data/uv-cache &&
  export UV_LINK_MODE=copy &&
  uv run python train.py --config-name sarm_full_folding \
    cfg.general.image_embedding_cache=/mnt/sarm-data/hf/lerobot/full_folding/cache/clip_image_embeddings.npy \
    2>&1 | tee /mnt/sarm-data/logs/sarm_full_folding.log
'
```

## Smoke Checks

Compile the touched Python modules:

```bash
uv run python -m py_compile \
  train.py \
  workspace/sarm_ws.py \
  utils/data_utils.py \
  lerobot/common/datasets/rm_lerobot_dataset.py \
  scripts/precompute_full_folding_clip_embeddings.py
```

Precompute a one-episode cache smoke:

```bash
uv run python scripts/precompute_full_folding_clip_embeddings.py \
  --episode-limit 1 \
  --output /mnt/sarm-data/hf/lerobot/full_folding/cache/test_clip_embeddings.npy \
  --done-output /mnt/sarm-data/hf/lerobot/full_folding/cache/test_clip_embeddings.done.npy
```

## Notes

- The cache is intentionally not committed.
- The first full run was bottlenecked by random MP4 seeks. This fork keeps the direct video path available, but the cached embedding path is the intended training path for full `lerobot/full_folding`.
- Extra GPUs will not speed up the current training loop unless distributed training is added. Extra CPU can help decode/precompute, but the biggest win is avoiding repeated video decode during training.
