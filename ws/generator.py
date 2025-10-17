import json
import os
from multiprocessing import Pool, cpu_count

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for faster rendering
import shutil

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Color interpolation function with clamping
def interpolate_color(start_color, end_color, position):
    r = start_color[0] + (end_color[0] - start_color[0]) * position
    g = start_color[1] + (end_color[1] - start_color[1]) * position
    b = start_color[2] + (end_color[2] - start_color[2]) * position
    return (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))


# Recursive branching function for generating mycelium and hyphae
def generate_randomized_branch(
    ax,
    x,
    y,
    branch_length,
    angle,
    depth,
    branch_width,
    color,
    num_initial_branches,
    temperature_factor,
    growth_factor,
):
    if depth == 0 or branch_length < 0.005:
        return

    # Scale branch length based on growth factor
    branch_length *= 0.7 * np.random.normal(1, 0.2)
    end_x = x + branch_length * np.cos(angle)
    end_y = y + branch_length * np.sin(angle)
    ax.plot([x, end_x], [y, end_y], color=color, alpha=0.8, linewidth=branch_width)

    # Dynamically control sub-branching and branch depth
    num_sub_branches = np.random.poisson(3)
    new_branch_length = (
        branch_length * (0.4 + np.random.uniform(0, 0.2)) * temperature_factor
    )
    new_branch_width = branch_width * (0.6 + np.random.uniform(0, 0.4))

    for _ in range(num_sub_branches):
        new_angle = np.random.uniform(0, 2 * np.pi)
        generate_randomized_branch(
            ax,
            end_x,
            end_y,
            new_branch_length,
            new_angle,
            depth - 1,
            new_branch_width,
            color=color,
            num_initial_branches=0,
            temperature_factor=temperature_factor,
            growth_factor=growth_factor,
        )

    # Initial branches, scaled by growth factor
    if num_initial_branches > 0:
        for _ in range(int(num_initial_branches * growth_factor)):
            new_angle = np.random.uniform(0, 2 * np.pi)
            generate_randomized_branch(
                ax,
                x,
                y,
                branch_length,
                new_angle,
                depth - 1,
                branch_width,
                color=color,
                num_initial_branches=0,
                temperature_factor=temperature_factor,
                growth_factor=growth_factor,
            )


# Function to generate spore images
def generate_spore_image(ax, spores, size_factor=1.0):
    for x, y in spores:
        color = interpolate_color((0.8, 0.7, 0), (1, 0.5, 0.2), y)
        spore_circle = plt.Circle((x, y), 0.02 * size_factor, color=color)
        ax.add_patch(spore_circle)


# Function to generate hyphae with branching
def generate_hyphae_image(
    ax,
    hyphae_positions,
    temperature_factor=1.0,
    growth_factor=0.5,
    size_factor=1.0,
):
    for x, y in hyphae_positions:
        color = interpolate_color((1, 0.6, 0.4), (1, 0.5, 0.15), y)
        hypha_circle = plt.Circle((x, y), 0.006 * size_factor, color=color)
        ax.add_patch(hypha_circle)
        generate_randomized_branch(
            ax,
            x,
            y,
            branch_length=0.06 * size_factor,  # Scale branch length
            angle=np.random.uniform(0, 2 * np.pi),
            depth=3,
            branch_width=2 * size_factor,  # Scale branch width
            color=color,
            num_initial_branches=4,
            temperature_factor=temperature_factor,
            growth_factor=growth_factor,
        )


# Function to generate mycelium with gradual growth
def generate_mycelium_image(
    ax, mycelium_positions, growth_factor=0.5, temperature_factor=1.0
):
    for x, y in mycelium_positions:
        color = interpolate_color((1, 0.6, 0.2), (0.6, 0.2, 0), growth_factor)
        mycelium_circle = plt.Circle((x, y), 0.004 * (0.5 + growth_factor), color=color)
        ax.add_patch(mycelium_circle)
        branch_length = 0.02 + growth_factor * 0.18
        depth = 4
        generate_randomized_branch(
            ax,
            x,
            y,
            branch_length=branch_length,
            angle=np.random.uniform(0, 2 * np.pi),
            depth=depth,
            branch_width=2 * (0.5 + growth_factor),
            color=color,
            num_initial_branches=int(3 + growth_factor * 10),
            temperature_factor=temperature_factor,
            growth_factor=growth_factor,
        )


# Get class label based on transition ratio
def get_class_label(transition_ratio):
    if transition_ratio <= 0.1:
        return "spore"
    elif transition_ratio <= 0.5:
        return "hyphae"
    else:
        return "mycelium"


# Function to generate a single fungi transition frame
# Pass all arguments as a tuple to facilitate multiprocessing with tqdm (annoying limitation)
def generate_fungi_transition_frame(args):
    (
        spore_positions,
        hyphae_positions,
        mycelium_positions,
        transition_ratio,
        i,
        output_dir,
        sequence_id,
    ) = args
    fig, ax = plt.subplots(figsize=(15, 15), dpi=72)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    spore_size_factor = max(0, 0.7 - transition_ratio * 1)
    hyphae_size_factor = 1 - transition_ratio * 0.5
    growth_factor = min(1, max(0, transition_ratio - 0.5) * 2)
    # Generate spores, hyphae, and mycelium
    generate_spore_image(ax, spore_positions, size_factor=spore_size_factor)
    generate_hyphae_image(
        ax,
        hyphae_positions,
        temperature_factor=np.random.normal(1.0, 0.1),
        growth_factor=transition_ratio,
        size_factor=hyphae_size_factor,
    )
    generate_mycelium_image(
        ax,
        mycelium_positions,
        growth_factor=growth_factor,
        temperature_factor=np.random.normal(1.0, 0.1),
    )

    # Save the frame
    frame_filename = f"seq{sequence_id:03d}_frame{i:04d}.png"
    frame_path = os.path.join(output_dir, frame_filename)
    fig.savefig(frame_path, bbox_inches="tight", pad_inches=0, dpi=72)
    plt.close(fig)
    return {
        "sequence_id": sequence_id,
        "frame_id": i,
        "frame_path": frame_path,
        "frame_filename": frame_filename,
        "class_label": get_class_label(transition_ratio),
        "transition_ratio": transition_ratio,
    }


# Main function to generate transition frames
def generate_sequence_frames(
    output_dir, num_frames=100, use_parallel=True, num_workers=None, sequence_id=0
):
    output_dir = os.path.join(output_dir, f"sequence_{sequence_id:03d}")
    os.makedirs(output_dir, exist_ok=True)

    # Initial spore positions
    spores = [
        (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)) for _ in range(20)
    ]
    hyphae_positions = []
    mycelium_positions = []

    all_frame_data = []

    for i in range(num_frames):
        transition_ratio = i / (num_frames - 1)

        # Convert spores to hyphae gradually
        if transition_ratio > 0.1 and len(spores) > 0:
            num_to_convert = max(
                1, int(len(spores) * transition_ratio * 0.1)
            )  # Few spores transition each step
            hyphae_positions.extend(spores[:num_to_convert])  # Add to hyphae
            spores = spores[num_to_convert:]  # Remove from spores

        # Convert hyphae to mycelium gradually
        if transition_ratio > 0.5 and len(hyphae_positions) > 0:
            num_to_convert = max(
                1, int(len(hyphae_positions) * (transition_ratio - 0.5) * 0.1)
            )  # Gradual transition
            mycelium_positions.extend(
                hyphae_positions[:num_to_convert]
            )  # Add to mycelium
            hyphae_positions = hyphae_positions[num_to_convert:]  # Remove from hyphae

        all_frame_data.append(
            {
                "spores": spores.copy(),
                "hyphae": hyphae_positions.copy(),
                "mycelium": mycelium_positions.copy(),
            }
        )
        spores = [
            (x + np.random.uniform(-0.01, 0.01), y + np.random.uniform(-0.01, 0.01))
            for x, y in spores
        ]

    # Generate frames in parallel or sequentially
    all_frames = []
    if use_parallel:
        # Default to number of CPU cores minus one
        num_workers = (
            num_workers if num_workers is not None else max(1, cpu_count() - 1)
        )
        args = [
            (
                frame_data["spores"],
                frame_data["hyphae"],
                frame_data["mycelium"],
                i / (num_frames - 1),
                i,
                output_dir,
                sequence_id,
            )
            for i, frame_data in enumerate(all_frame_data)
        ]
        with Pool(processes=num_workers) as pool:
            # Have to use imap for tqdm to work
            all_frames = list(
                tqdm(
                    pool.imap(generate_fungi_transition_frame, args),
                    total=num_frames,
                    desc="Generating Frames",
                )
            )
    else:
        for i, frame_data in tqdm(
            enumerate(all_frame_data),
            total=num_frames,
            desc="Generating Frames",
        ):
            frame_path = generate_fungi_transition_frame(
                (
                    frame_data["spores"],
                    frame_data["hyphae"],
                    frame_data["mycelium"],
                    i / (num_frames - 1),
                    i,
                    output_dir,
                    sequence_id,
                )
            )
            all_frames.append(frame_path)

    print(f"Generated {len(all_frames)} frames in {output_dir}")
    return all_frames


# Function to generate multiple sequences and save metadata
def generate_multiple_sequences(
    output_dir,
    num_sequences=3,
    frames_per_sequence=20,
    seed=42,
    use_parallel=True,
    num_workers=None,
    overwrite=False,
):
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    if os.path.exists(metadata_path) and not overwrite:
        raise FileExistsError(
            f"Dataset metadata already exists at {metadata_path}. Use overwrite=True to regenerate."
        )
    else:
        if os.path.exists(output_dir):
            print("Deleting existing dataset and regenerating")
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    all_sequences = []
    for sequence_id in range(num_sequences):
        # Set seed once per sequence for reproducibility
        np.random.seed(seed + sequence_id)

        print(f"Sequence {sequence_id + 1}/{num_sequences}")
        frames = generate_sequence_frames(
            output_dir,
            num_frames=frames_per_sequence,
            use_parallel=use_parallel,
            num_workers=num_workers,
            sequence_id=sequence_id,
        )
        all_sequences.append(
            {
                "sequence_id": sequence_id,
                "sequence_dir": f"sequence_{sequence_id:03d}",
                "sequence_path": os.path.join(
                    output_dir, f"sequence_{sequence_id:03d}"
                ),
                "frames": frames,
            }
        )
        print("\n", end="")

    dataset = {
        "metadata_path": metadata_path,
        "num_sequences": num_sequences,
        "frames_per_sequence": frames_per_sequence,
        "total_frames": num_sequences * frames_per_sequence,
        "seed": seed,
        "sequences": all_sequences,
    }

    with open(metadata_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Saved dataset metadata to {metadata_path}")

    return metadata_path, dataset


if __name__ == "__main__":
    output_dir = "data/test"  # Replace with your preferred path
    num_sequences = 10
    frames_per_sequence = 20

    metadata_path, dataset = generate_multiple_sequences(
        output_dir,
        num_sequences=num_sequences,
        frames_per_sequence=frames_per_sequence,
        use_parallel=True,
        num_workers=None,
        overwrite=True,
    )
