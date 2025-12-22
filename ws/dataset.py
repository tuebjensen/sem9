import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SmokeDataset(Dataset):
    def __init__(self, dataset_root, split, downsample_factor=4):
        self.dataset_root = dataset_root
        self.split = split
        self.downsample_factor = downsample_factor

        metadata_path = os.path.join(dataset_root, "dataset_summary.json")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.sim_config = self.metadata["sim_config"]
        self.snapshot_interval = self.sim_config["snapshot_interval"]
        self.total_steps = self.sim_config["total_steps"]
        self.width = self.sim_config["width"]
        self.height = self.sim_config["height"]
        self.window_size = self.snapshot_interval
        self.downsampled_window_size = self.window_size // self.downsample_factor

        self.sensor_stats = self.metadata["sensor_stats"]
        self.smoke_stats = self.metadata["smoke_stats"]

        # view as (1, 1, 2) for broadcasting over (T, 20, 2)
        self.sensor_mean = torch.tensor(
            self.sensor_stats["mean"], dtype=torch.float32
        ).reshape(1, 1, 2)
        self.sensor_std = torch.tensor(
            self.sensor_stats["std"], dtype=torch.float32
        ).reshape(1, 1, 2)
        self.smoke_mean = torch.tensor(self.smoke_stats["mean"], dtype=torch.float32)
        self.smoke_std = torch.tensor(self.smoke_stats["std"], dtype=torch.float32)

        self.run_ids = self.metadata["split_indices"][split]
        self.samples = []

        available_timestamps = list(range(0, self.total_steps, self.window_size))

        for run_id in self.run_ids:
            for t in available_timestamps:
                if t - self.window_size < 0:
                    continue
                if t + self.window_size >= self.total_steps:
                    continue

                self.samples.append((run_id, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        run_id, t = self.samples[idx]
        run_dir = os.path.join(self.dataset_root, self.split, f"run_{run_id:04d}")

        with np.load(os.path.join(run_dir, "metadata.npz")) as meta_data:
            mask = meta_data["mask"].astype(np.float32).T
            sensor_pos_raw = meta_data["sensor_positions"]
            all_readings = meta_data["sensor_readings"]

        # History: [t - 99, t] (downsampled by 4)
        history_indices = np.linspace(
            t - self.window_size + self.downsample_factor,
            t,
            num=self.downsampled_window_size,
            dtype=int,
        )
        sensor_history_np = all_readings[history_indices].transpose(1, 0, 2)

        # Future: [t + 1, t + 100] (downsampled by 4)
        future_indices = np.linspace(
            t + self.downsample_factor,
            t + self.window_size,
            num=self.downsampled_window_size,
            dtype=int,
        )
        sensor_future_np = all_readings[future_indices].transpose(1, 0, 2)

        sensor_history = torch.from_numpy(sensor_history_np).float()
        sensor_future = torch.from_numpy(sensor_future_np).float()

        sensor_history = (sensor_history - self.sensor_mean) / self.sensor_std
        sensor_future = (sensor_future - self.sensor_mean) / self.sensor_std

        curr_img_path = os.path.join(run_dir, f"raw_data/smoke_{t:05d}.npy")
        curr_smoke_np = np.load(curr_img_path).astype(np.float32).T

        target_t = t + self.window_size
        target_img_path = os.path.join(run_dir, f"raw_data/smoke_{target_t:05d}.npy")
        target_smoke_np = np.load(target_img_path).astype(np.float32).T

        curr_smoke = torch.from_numpy(curr_smoke_np).float()
        target_smoke = torch.from_numpy(target_smoke_np).float()

        curr_smoke = (curr_smoke - self.smoke_mean) / self.smoke_std
        target_smoke = (target_smoke - self.smoke_mean) / self.smoke_std

        input_image = torch.stack([curr_smoke, torch.from_numpy(mask).float()], dim=0)
        target_image = target_smoke.unsqueeze(0)

        aspect_ratio = self.width / self.height
        norm_sensor_pos = np.zeros_like(sensor_pos_raw, dtype=np.float32)
        norm_sensor_pos[:, 0] = (
            (sensor_pos_raw[:, 0] / self.width) * 2 - 1
        ) * aspect_ratio
        norm_sensor_pos[:, 1] = (sensor_pos_raw[:, 1] / self.height) * 2 - 1

        return {
            "sensor_pos": torch.from_numpy(norm_sensor_pos).float(),
            "sensor_history_vals": sensor_history,
            "current_image": input_image,
            "sensor_target_vals": sensor_future,
            "target_image": target_image,
        }
