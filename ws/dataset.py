import json
import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class TemporallyConsistentTransform:
    def __init__(self, image_size=336, augment=True):
        self.image_size = image_size
        self.augment = augment

    def __call__(self, frames):
        # Sample our parameters once for the entire subsequence
        if self.augment:
            rotation_angle = random.uniform(-15, 15)
            do_hflip = random.random() < 0.5
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            saturation_factor = random.uniform(0.8, 1.2)

        transformed_frames = []
        for frame in frames:
            frame = TF.resize(frame, [self.image_size, self.image_size])

            if self.augment:
                frame = TF.rotate(frame, rotation_angle)

                if do_hflip:
                    frame = TF.hflip(frame)

                frame = TF.adjust_brightness(frame, brightness_factor)
                frame = TF.adjust_contrast(frame, contrast_factor)
                frame = TF.adjust_saturation(frame, saturation_factor)

            frame = TF.to_tensor(frame)
            # ImageNet normalization
            frame = TF.normalize(
                frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            transformed_frames.append(frame)

        return transformed_frames


class FungiSequenceDataset(Dataset):
    def __init__(
        self, metadata_path, split, subsequence_length=1, stride=None, transform=None
    ):
        self.metadata_path = metadata_path
        self.split = split

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.frames_per_sequence = self.metadata["frames_per_sequence"]

        assert subsequence_length >= 1, (
            f"subsequence_length must be >= 1, got {subsequence_length}"
        )
        assert subsequence_length <= self.frames_per_sequence, (
            f"subsequence_length ({subsequence_length}) cannot exceed frames_per_sequence ({self.frames_per_sequence})"
        )

        self.subsequence_length = subsequence_length
        self.stride = stride if stride is not None else subsequence_length

        assert self.stride >= 1, f"stride must be >= 1, got {self.stride}"
        assert self.stride <= self.subsequence_length, (
            f"stride ({self.stride}) should be <= subsequence_length ({self.subsequence_length}) "
            f"to ensure all frames are seen"
        )

        augment = split == "train"

        self.transform = (
            transform
            if transform is not None
            else TemporallyConsistentTransform(image_size=336, augment=augment)
        )

        self.class_to_idx = {"spore": 0, "hyphae": 1, "mycelium": 2}
        self.idx_to_class = {
            0: "spore",
            1: "hyphae",
            2: "mycelium",
        }

        self.sequences = self.metadata["splits"][self.split]

        self.index = []
        for seq_idx, sequence in enumerate(self.sequences):
            num_frames = len(sequence["frames"])
            for start_frame in range(
                0,
                num_frames - self.subsequence_length + 1,
                self.stride,
            ):
                self.index.append((seq_idx, start_frame))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, start_frame_idx = self.index[idx]
        sequence = self.sequences[seq_idx]

        frames = []
        class_labels = []
        timestamps = []

        for i in range(self.subsequence_length):
            frame_info = sequence["frames"][start_frame_idx + i]
            frame = Image.open(frame_info["frame_path"]).convert("RGB")
            frames.append(frame)
            class_label = self.class_to_idx[frame_info["class_label"]]
            class_labels.append(class_label)
            timestamps.append(frame_info["transition_ratio"])

        frames = self.transform(frames)
        return {
            "frames": torch.stack(frames),
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
            "sequence_id": sequence["sequence_id"],
        }
