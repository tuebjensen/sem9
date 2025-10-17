import json
import os
import random
import shutil

from generator import generate_multiple_sequences


def split_and_reorganize_dataset(
    metadata_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    output_dir = os.path.dirname(metadata_path)
    num_sequences = len(metadata["sequences"])

    sequence_ids = list(range(num_sequences))
    random.seed(seed)
    random.shuffle(sequence_ids)

    train_end = int(num_sequences * train_ratio)
    val_end = train_end + int(num_sequences * val_ratio)

    train_ids = set(sequence_ids[:train_end])
    val_ids = set(sequence_ids[train_end:val_end])
    test_ids = set(sequence_ids[val_end:])

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    split_metadata = {
        "metadata_path": metadata_path,
        "num_sequences": num_sequences,
        "frames_per_sequence": metadata["frames_per_sequence"],
        "total_frames": metadata["total_frames"],
        "split_info": {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        },
        "splits": {"train": [], "val": [], "test": []},
    }

    for sequence in metadata["sequences"]:
        sequence_id = sequence["sequence_id"]
        if sequence_id in train_ids:
            split = "train"
        elif sequence_id in val_ids:
            split = "val"
        else:
            split = "test"

        # Move sequence directory to the appropriate split folder
        old_sequence_path = os.path.join(output_dir, sequence["sequence_dir"])
        new_sequence_path = os.path.join(output_dir, split, sequence["sequence_dir"])
        if os.path.exists(old_sequence_path):
            if os.path.exists(new_sequence_path):
                shutil.rmtree(new_sequence_path)
            shutil.move(old_sequence_path, new_sequence_path)

        # Update sequence metadata
        updated_frames = []
        for frame in sequence["frames"]:
            frame_filename = frame["frame_filename"]
            new_frame_path = os.path.join(new_sequence_path, frame_filename)
            frame["frame_path"] = new_frame_path
            updated_frames.append(frame)

        split_metadata["splits"][split].append(
            {
                "sequence_id": sequence_id,
                "sequence_dir": sequence["sequence_dir"],
                "sequence_path": new_sequence_path,
                "frames": updated_frames,
            }
        )

    with open(metadata_path, "w") as f:
        json.dump(split_metadata, f, indent=4)

    print(f"Dataset split completed. Metadata updated at {metadata_path}")

    return split_metadata


def create_dataset(
    output_dir="data/test",
    num_sequences=20,
    frames_per_sequence=20,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    overwrite=False,
    use_parallel=True,
    num_workers=None,
):
    print(f"Generating dataset at {output_dir}...")
    print(f"  Number of sequences: {num_sequences}")
    print(f"  Frames per sequence: {frames_per_sequence}")
    print(f"  Train/Val/Test split: {train_ratio} / {val_ratio} / {test_ratio}")
    print(f"  Seed: {seed}")
    print(f"  Overwrite: {overwrite}")
    print(f"  Use parallel: {use_parallel}")
    print(f"  Num workers: {num_workers}")

    print("\n[STEP 1/2] Generating sequences...")
    metadata_path, dataset = generate_multiple_sequences(
        output_dir=output_dir,
        num_sequences=num_sequences,
        frames_per_sequence=frames_per_sequence,
        seed=seed,
        overwrite=overwrite,
        use_parallel=use_parallel,
        num_workers=num_workers,
    )

    print("\n[STEP 2/2] Splitting and reorganizing dataset...")

    metadata = split_and_reorganize_dataset(
        metadata_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    return metadata


if __name__ == "__main__":
    output_dir = "data/test_dataset"
    num_sequences = 20
    frames_per_sequence = 20
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    seed = 42
    overwrite = False
    use_parallel = True
    num_workers = None
    create_dataset(
        output_dir=output_dir,
        num_sequences=num_sequences,
        frames_per_sequence=frames_per_sequence,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        overwrite=overwrite,
        use_parallel=use_parallel,
        num_workers=num_workers,
    )
