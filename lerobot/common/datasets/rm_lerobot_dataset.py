import torch
from typing import Callable
from pathlib import Path
from .lerobot_dataset import LeRobotDataset
from .video_utils import decode_video_frames, get_safe_default_codec
from datasets import Dataset
import json
import numpy as np
import pandas as pd
import time
from typing import Tuple
from faker import Faker

from utils.data_utils import resolve_lerobot_root


class FullFoldingSarmDataset(torch.utils.data.Dataset):
    """LeRobot v3 adapter for lerobot/full_folding with SARM progress targets."""

    def __init__(
        self,
        repo_id: str = "lerobot/full_folding",
        episodes: list[int] | None = None,
        n_obs_steps: int = 8,
        frame_gap: int = 30,
        max_rewind_steps: int = 4,
        root: str | Path | None = None,
        image_names: list[str] | None = None,
        state_key: str = "observation.state",
        progress_key: str = "progress_sparse",
        task_name: str = "fold clothing",
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        episode_limit: int | None = None,
    ):
        self.repo_id = repo_id
        self.root = resolve_lerobot_root(repo_id, str(root) if root is not None else None)
        self.episodes = episodes
        self.n_obs_steps = n_obs_steps
        self.frame_gap = frame_gap
        self.max_rewind_steps = max_rewind_steps
        self.image_names = image_names or ["observation.images.base"]
        self.state_key = state_key
        self.progress_key = progress_key
        self.task_name = task_name
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or get_safe_default_codec()

        with open(self.root / "meta" / "info.json", "r") as f:
            self.info = json.load(f)
        self.fps = int(self.info["fps"])
        self.video_path_template = self.info["video_path"]
        self.data_path_template = self.info["data_path"]
        self.meta = type("FullFoldingMeta", (), {})()
        self.meta.video_keys = [k for k, v in self.info["features"].items() if v["dtype"] == "video"]
        self.meta.camera_keys = self.meta.video_keys
        self.meta.features = self.info["features"]
        for image_name in self.image_names:
            if image_name not in self.meta.video_keys:
                raise KeyError(f"Video key '{image_name}' not found. Available video keys: {self.meta.video_keys}")

        self.episode_table = self._load_episode_table()
        if episode_limit is not None:
            self.episode_table = self.episode_table.iloc[:episode_limit]
        if self.episodes is not None:
            episode_set = set(int(ep) for ep in self.episodes)
            self.episode_table = self.episode_table[self.episode_table["episode_index"].isin(episode_set)]
        if self.episode_table.empty:
            raise ValueError(f"No episodes selected for {self.root}")
        self.episode_table = self.episode_table.sort_values("episode_index").reset_index(drop=True)
        self.episode_by_index = {
            int(row.episode_index): row._asdict() for row in self.episode_table.itertuples(index=False)
        }

        paths = sorted((self.root / "data").glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet data files under {self.root / 'data'}")
        self.hf_dataset = Dataset.from_parquet([str(p) for p in paths])

        self.progress = self._load_progress()
        self.sample_indices = self._build_sample_indices()
        if len(self.sample_indices) == 0:
            raise ValueError("Selected episodes contain no trainable frame indices.")

    def _load_episode_table(self) -> pd.DataFrame:
        paths = sorted((self.root / "meta" / "episodes").glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No v3 episode metadata under {self.root / 'meta' / 'episodes'}")
        return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

    def _load_progress(self) -> np.ndarray:
        path = self.root / "sarm_progress.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing SARM progress file: {path}")
        progress_df = pd.read_parquet(path, columns=["index", self.progress_key])
        max_index = int(progress_df["index"].max())
        values = np.full(max_index + 1, np.nan, dtype=np.float32)
        values[progress_df["index"].to_numpy(dtype=np.int64)] = progress_df[self.progress_key].to_numpy(dtype=np.float32)
        if np.isnan(values).any():
            missing = int(np.isnan(values).sum())
            print(f"[Data] Warning: {missing} frame progress values are missing in {path}.")
        return values

    def _build_sample_indices(self) -> list[int]:
        indices: list[int] = []
        for row in self.episode_table.itertuples(index=False):
            start = int(getattr(row, "dataset_from_index"))
            end = int(getattr(row, "dataset_to_index"))
            indices.extend(range(start, end))
        return indices

    def __len__(self):
        return len(self.sample_indices)

    def get_frame_indices(self, idx: int, ep_start: int, ep_end: int) -> list[int]:
        idx = max(ep_start, min(idx, ep_end))
        gaps = self.n_obs_steps
        if gaps == 0:
            return [idx]

        total_needed = self.frame_gap * gaps
        available = idx - ep_start
        if available >= total_needed:
            return [idx - self.frame_gap * (gaps - k) for k in range(gaps)] + [idx]

        frames = [ep_start + round(available * k / gaps) for k in range(gaps)] + [idx]
        for i in range(1, len(frames)):
            if frames[i] < frames[i - 1]:
                frames[i] = frames[i - 1]
        return frames

    def _video_file_path(self, ep: dict, video_key: str) -> Path:
        chunk_idx = int(ep[f"videos/{video_key}/chunk_index"])
        file_idx = int(ep[f"videos/{video_key}/file_index"])
        return self.root / self.video_path_template.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )

    def _decode_video(self, ep: dict, video_key: str, timestamps: list[float]) -> torch.Tensor:
        from_ts = float(ep[f"videos/{video_key}/from_timestamp"])
        shifted = [from_ts + float(ts) for ts in timestamps]
        return decode_video_frames(
            self._video_file_path(ep, video_key),
            shifted,
            self.tolerance_s,
            self.video_backend,
        )

    def _rows(self, indices: list[int]) -> dict:
        batch = self.hf_dataset[indices]
        return batch

    def __getitem__(self, idx: int) -> dict:
        abs_idx = int(self.sample_indices[idx])
        item = self.hf_dataset[abs_idx]
        ep_idx = int(item["episode_index"])
        ep = self.episode_by_index[ep_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"]) - 1
        obs_indices = self.get_frame_indices(abs_idx, ep_start, ep_end)
        rows = self._rows(obs_indices)

        seq_item = {}
        state = torch.tensor(rows[self.state_key], dtype=torch.float32)
        seq_item["state"] = state
        seq_item["actions"] = torch.tensor(rows["action"], dtype=torch.float32)
        seq_item["timestamp"] = torch.tensor(rows["timestamp"], dtype=torch.float32)[-1]
        seq_item["frame_index"] = torch.tensor(rows["frame_index"], dtype=torch.int64)[-1]
        seq_item["episode_index"] = torch.tensor(ep_idx, dtype=torch.int64)
        seq_item["index"] = torch.tensor(abs_idx, dtype=torch.int64)
        seq_item["task_index"] = torch.tensor(rows["task_index"][-1], dtype=torch.int64)

        obs_ts_range = [float(ts) for ts in rows["timestamp"]]
        rewind_flag = self.max_rewind_steps > 0 and torch.rand(1).item() < 0.8 and abs_idx > ep_start + self.n_obs_steps * self.frame_gap
        rewind_step = 0
        rewind_indices = []
        for key in self.image_names:
            frames = self._decode_video(ep, key, obs_ts_range)
            if rewind_flag:
                max_valid_step = min(self.max_rewind_steps, max(1, (abs_idx - ep_start) // max(1, self.frame_gap)))
                rewind_step = int(torch.randint(1, max_valid_step + 1, (1,)).item())
                rewind_indices = list(range(abs_idx - rewind_step * self.frame_gap, abs_idx, self.frame_gap))
                rewind_indices = [max(ep_start, min(i, ep_end)) for i in rewind_indices]
                rewind_rows = self._rows(rewind_indices)
                rewind_ts = [float(ts) for ts in rewind_rows["timestamp"]]
                rewind_frames = torch.flip(self._decode_video(ep, key, rewind_ts), dims=[0])
                if rewind_frames.ndim == 3:
                    rewind_frames = rewind_frames.unsqueeze(0)
                pad_count = self.max_rewind_steps - rewind_step
                if pad_count > 0:
                    pad = torch.zeros((pad_count, *rewind_frames.shape[1:]), dtype=rewind_frames.dtype)
                    rewind_frames = torch.cat([rewind_frames, pad], dim=0)
                frames = torch.cat([frames, rewind_frames], dim=0)
            else:
                padding_frames = torch.zeros((self.max_rewind_steps, *frames.shape[1:]), dtype=frames.dtype)
                frames = torch.cat([frames, padding_frames], dim=0)
            seq_item[key] = frames

        targets = torch.zeros(1 + self.n_obs_steps + self.max_rewind_steps, dtype=torch.float32)
        progress_values = torch.tensor(self.progress[np.asarray(obs_indices, dtype=np.int64)], dtype=torch.float32)
        progress_values = torch.nan_to_num(progress_values, nan=0.0).clamp(0.0, 1.0)
        targets[: self.n_obs_steps + 1] = progress_values
        if rewind_flag and rewind_indices:
            rewind_progress = torch.tensor(self.progress[np.asarray(rewind_indices, dtype=np.int64)], dtype=torch.float32)
            rewind_progress = torch.nan_to_num(rewind_progress, nan=0.0).clamp(0.0, 1.0)
            targets[1 + self.n_obs_steps : 1 + self.n_obs_steps + rewind_step] = torch.flip(rewind_progress, dims=[0])
        seq_item["targets"] = targets

        state_with_rewind = torch.zeros([1 + self.n_obs_steps + self.max_rewind_steps, state.shape[-1]], dtype=torch.float32)
        state_with_rewind[: self.n_obs_steps + 1, :] = state
        if rewind_flag and rewind_indices:
            rewind_state = torch.tensor(self._rows(rewind_indices)[self.state_key], dtype=torch.float32)
            state_with_rewind[1 + self.n_obs_steps : 1 + self.n_obs_steps + rewind_step, :] = torch.flip(rewind_state, dims=[0])
        seq_item["state"] = state_with_rewind

        frame_relative_indices = torch.zeros(1 + self.n_obs_steps + self.max_rewind_steps, dtype=torch.float32)
        for i, frame_idx in enumerate(obs_indices):
            frame_relative_indices[i] = (frame_idx - ep_start) / (ep_end - ep_start) if ep_end > ep_start else 0.0
        if rewind_flag and rewind_indices:
            for i, frame_idx in enumerate(reversed(rewind_indices[:rewind_step])):
                frame_relative_indices[1 + self.n_obs_steps + i] = (
                    (frame_idx - ep_start) / (ep_end - ep_start) if ep_end > ep_start else 0.0
                )
        seq_item["frame_relative_indices"] = frame_relative_indices
        seq_item["lengths"] = torch.tensor(1 + self.n_obs_steps + rewind_step, dtype=torch.int32)
        seq_item["task"] = self.task_name
        return seq_item



class FrameGapLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        n_obs_steps: int = 1,
        frame_gap: int = 1,
        max_rewind_steps: int = 0,
        root: str | Path | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        image_names: list[str] = ["top_camera-images-rgb"],
        video_eval: bool = False,
        annotation_list: list[str] | None = None,
        task_name: str = "fold the tshirt",
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        self.n_obs_steps = n_obs_steps
        self.frame_gap = frame_gap
        self.max_rewind_steps = max_rewind_steps
        self.timestamp_tensor = torch.tensor(self.hf_dataset["timestamp"]).flatten()
        assert all(img_name in self.meta.video_keys for img_name in image_names), f"Image names {image_names} not found in metadata video keys."
        assert 'reward' in self.meta.features, f"'reward' (progress label) not found in dataset features."
        self.wrapped_video_keys = image_names  # Use only the specified camera for videos
        self.verbs = ['move', 'grasp', 'rotate', 'push', 'pull', 'slide', 'lift', 'place']
        self.fake = Faker()
        self.video_eval = video_eval
        self.annotation_list = annotation_list
        self.task_name = task_name

    def get_frame_indices(self, idx: int,
                      n_obs_steps: int,
                      frame_gap: int,
                      ep_start: int = 0,
                      ep_end: int | None = None) -> list[int]:
        """
        Build a monotonic sequence of length n_obs_steps+1 ending at idx.
        - Prefer fixed frame_gap when enough history exists.
        - Otherwise adapt the effective gap to fit within [ep_start, idx].
        - No padding; no extra inputs.

        Args:
            idx: last frame index (target frame).
            n_obs_steps: number of history steps (total length = n_obs_steps+1).
            frame_gap: desired fixed stride between history frames when possible.
            ep_start: episode start index (inclusive).
            ep_end: episode end index (inclusive); if None, unbounded above.

        Returns:
            List of indices (non-decreasing), length = n_obs_steps + 1.
        """
        # Clamp idx to episode bounds
        if ep_end is not None:
            idx = min(idx, ep_end)
        idx = max(idx, ep_start)

        gaps = n_obs_steps
        if gaps == 0:
            return [idx]

        # Check if fixed stride fits entirely inside the episode
        total_needed = frame_gap * gaps  # distance from earliest to idx
        available = idx - ep_start

        if available >= total_needed:
            # Use fixed frame_gap
            frames = [idx - frame_gap * (gaps - k) for k in range(gaps)] + [idx]
        else:
            # Not enough history: adapt stride by evenly spacing from ep_start to idx
            # Use integer rounding and enforce monotonicity.
            frames = [ep_start + round(available * k / gaps) for k in range(gaps)] + [idx]
            for i in range(1, len(frames)):
                if frames[i] < frames[i - 1]:
                    frames[i] = frames[i - 1]

        return frames


    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        assert ep_idx in self.episodes, f"Episode {ep_idx} not found in the selected episodes."
        global_idx = self.episodes.index(ep_idx)

        ep_start = self.episode_data_index["from"][global_idx].item()
        ep_end = self.episode_data_index["to"][global_idx].item() - 1

        # Adjust idx if there's not enough history
        required_history = self.n_obs_steps * self.frame_gap
        
        # Compute frame indices for observation
        obs_indices = self.get_frame_indices(idx, self.n_obs_steps, self.frame_gap, ep_start, ep_end)
        sequence = self.hf_dataset.select(obs_indices)

        # Extract sequence data
        seq_item = {}
        for key in sequence.features:
            value = sequence[key]
            if key == "actions":
                seq_item[key] = torch.stack(value)
            elif key == "state":
                seq_item[key] = torch.stack(value)
            elif key == "reward":
                progress_list = torch.stack(value).squeeze(-1)
            else:
                seq_item[key] = value[0]
            del value
        del sequence

        # Query video frames
        obs_ts_range = self.timestamp_tensor[obs_indices].tolist()
        query_ts_dict = {key: obs_ts_range for key in self.wrapped_video_keys}
        video_frames = self._query_videos(query_ts_dict, ep_idx)
        
        if not self.video_eval and self.max_rewind_steps > 0:
            rewind_flag = torch.rand(1).item() < 0.8 and idx > ep_start + required_history
        else:
            rewind_flag = False
        rewind_step = None
        for key in self.wrapped_video_keys:
            frames = video_frames[key]
            if frames.shape[0] < self.n_obs_steps:
                pad_count = self.n_obs_steps - frames.shape[0]
                pad_frame = frames[-1:].repeat(pad_count, 1, 1, 1)
                frames = torch.cat([frames, pad_frame], dim=0)

            if rewind_flag:
                rewind_step, rewind_frames = self._get_rewind(
                    idx, key, ep_idx, rewind_step=rewind_step
                )
                frames = torch.cat([frames, rewind_frames], dim=0)
            else:
                rewind_step = 0
                padding_frames = torch.zeros((self.max_rewind_steps, *frames.shape[1:]), dtype=frames.dtype)
                frames = torch.cat([frames, padding_frames], dim=0)

            seq_item[key] = frames

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in seq_item:
                    seq_item[cam] = self.image_transforms(seq_item[cam])

        # Task string
        pertube_task_flag = torch.rand(1).item() < 0.2
        if self.video_eval:
            pertube_task_flag = False
        if pertube_task_flag:
            num_words = torch.randint(1, 6, (1,)).item()
            verb = self.verbs[torch.randint(0, len(self.verbs), (1,)).item()]
            phrase = [verb] + self.fake.words(nb=num_words)
            seq_item["task"] = " ".join(phrase)
        else:
            seq_item["task"] = self.task_name

        # Progress targets
        seq_item["targets"] = torch.zeros(1 + self.n_obs_steps + self.max_rewind_steps, dtype=torch.float32)
        state_with_rewind = torch.zeros([1 + self.n_obs_steps + self.max_rewind_steps, seq_item["state"].shape[-1]], dtype=torch.float32)
        state_with_rewind[:self.n_obs_steps + 1, :] = seq_item["state"]
        frame_relative_indices = torch.zeros(1 + self.n_obs_steps + self.max_rewind_steps, dtype=torch.float32)

        if not pertube_task_flag:
            seq_item["targets"][:self.n_obs_steps + 1] = progress_list
            for i in range(rewind_step):
                seq_item["targets"][1 + self.n_obs_steps + i] = torch.flip(progress_list, dims=[0])[i + 1]
        
        for i, idx in enumerate(obs_indices):
            frame_relative_indices[i] = (idx - ep_start) / (ep_end - ep_start) if ep_end > ep_start else 0.0
        
        for i in range(rewind_step):
            frame_relative_indices[1 + self.n_obs_steps + i] = torch.flip(frame_relative_indices[:self.n_obs_steps + 1], dims=[0])[i + 1]
            state_with_rewind[1 + self.n_obs_steps + i, :] = torch.flip(seq_item["state"], dims=[0])[i + 1]
        
        seq_item["state"] = state_with_rewind
        seq_item["lengths"] = torch.tensor(1 + self.n_obs_steps + rewind_step, dtype=torch.int32)
        seq_item["frame_relative_indices"] = frame_relative_indices

        del item, video_frames, query_ts_dict, obs_ts_range, progress_list, state_with_rewind, frame_relative_indices

        return seq_item


    def _get_rewind(self, idx: int, key: str, ep_idx: int, rewind_step=None) -> Tuple[int, torch.Tensor]:
        assert self.max_rewind_steps < self.n_obs_steps, "Max rewind steps must be less than n_obs_steps."

        max_valid_step = (idx - self.frame_gap) // self.frame_gap
        max_rewind = min(self.max_rewind_steps, max_valid_step)

        if rewind_step is None:
            rewind_step = torch.randint(1, max_rewind + 1, (1,)).item()

        rewind_indices = list(range(idx - rewind_step * self.frame_gap, idx, self.frame_gap))
        if len(rewind_indices) < rewind_step:
            pad_count = rewind_step - len(rewind_indices)
            rewind_indices += [rewind_indices[-1]] * pad_count

        rewind_ts_range = self.timestamp_tensor[rewind_indices].tolist()
        query_ts_dict = {key: rewind_ts_range}
        rewind_frames = self._query_videos(query_ts_dict, ep_idx)[key]

        if rewind_frames.ndim == 3:
            rewind_frames = rewind_frames.unsqueeze(0)

        rewind_frames = torch.flip(rewind_frames, dims=[0])
        padding_needed = self.max_rewind_steps - rewind_step
        if padding_needed > 0:
            pad = torch.zeros((padding_needed, *rewind_frames.shape[1:]), dtype=rewind_frames.dtype)
            rewind_frames = torch.cat([rewind_frames, pad], dim=0)

        return rewind_step, rewind_frames

