import argparse
import glob
import json
import os
import shutil

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import SmokeDataset
from model import MultirateTemporalSenseiver
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm


def denormalize(tensor, mean, std):
    return (tensor * std) + mean


def prettify_run_name(name):
    if "baseline" in name:
        display_name = "1 Step Upsampling"
    elif "2_step_upsampling" in name:
        display_name = "2 Step Upsampling"
    elif "4_step_upsampling" in name:
        display_name = "4 Step Upsampling"
    else:
        display_name = name

    details = []
    if "bilinear" in name:
        details.append("Bilinear")
    elif "nearest" in name:
        details.append("Nearest")

    if "l1" in name:
        details.append("L1")
    elif "mse" in name:
        details.append("MSE")

    if details:
        display_name += f" ({' '.join(details)})"

    return display_name


def get_grid_location(run_name):
    if "baseline" in run_name:
        row = 0
    elif "2_step" in run_name:
        row = 1
    elif "4_step" in run_name:
        row = 2
    else:
        row = 3

    if "bilinear" in run_name and "l1" in run_name:
        col = 0
        subtitle = "Bilinear MAE"
    elif "bilinear" in run_name and "mse" in run_name:
        col = 1
        subtitle = "Bilinear MSE"
    elif "nearest" in run_name and "l1" in run_name:
        col = 2
        subtitle = "Nearest MAE"
    elif "nearest" in run_name and "mse" in run_name:
        col = 3
        subtitle = "Nearest MSE"
    else:
        col = 4
        subtitle = "Other"

    return row, col, subtitle


def load_model_from_run(run_name, checkpoints_root="checkpoints", device="cuda"):
    ckpt_dir = os.path.join(checkpoints_root, run_name)

    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        print(f"Warning: No checkpoints found in {ckpt_dir}")
        return None, None

    best_ckpt = min(
        ckpts,
        key=lambda x: float(x.split("val_total_loss=")[-1].replace(".ckpt", "")),
    )

    print(f"Loading best checkpoint for '{run_name}': {os.path.basename(best_ckpt)}")

    model = MultirateTemporalSenseiver.load_from_checkpoint(best_ckpt)
    model.to(device)
    model.eval()
    return model, best_ckpt


def save_visualization_packet(
    run_name,
    indices,
    input_imgs,
    target_imgs,
    pred_imgs,
    sensor_preds,
    sensor_targets,
    sensor_pos,
    dataset_stats,
    output_dir,
    prefix="sample",
):
    os.makedirs(output_dir, exist_ok=True)

    packet = {
        "run_name": run_name,
        "indices": indices,
        "input_imgs": [t.cpu() for t in input_imgs],
        "target_imgs": [t.cpu() for t in target_imgs],
        "pred_imgs": [t.cpu() for t in pred_imgs],
        "sensor_preds": [t.cpu() for t in sensor_preds],
        "sensor_targets": [t.cpu() for t in sensor_targets],
        "sensor_pos": [t.cpu() for t in sensor_pos],
        "stats": {
            k: v.cpu() if torch.is_tensor(v) else v for k, v in dataset_stats.items()
        },
        "prefix": prefix,
    }

    save_path = os.path.join(output_dir, f"{prefix}_data.pt")
    torch.save(packet, save_path)
    print(f"Saved visualization packet: {save_path}")


def run_inference(
    exp_config,
    data_root,
    device,
    base_output_dir,
    num_worst_cases,
    num_best_cases,
    num_fixed_samples,
):
    original_run_name = exp_config["run_name"]
    print(f"Evaluating Run: {original_run_name}")

    model, ckpt_path = load_model_from_run(original_run_name, device=device)
    if model is None:
        return None

    ds = SmokeDataset(
        data_root, split="test", downsample_factor=model.hparams.downsample_factor
    )
    dataloader = DataLoader(ds, batch_size=16, shuffle=False)

    total_samples = len(ds)
    if num_fixed_samples > total_samples:
        num_fixed_samples = total_samples

    rng = np.random.RandomState(42)
    fixed_indices = rng.choice(total_samples, size=num_fixed_samples, replace=False)
    fixed_indices_set = set(fixed_indices)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    sample_errors = []

    total_metrics = {
        "sensor_loss": 0.0,
        "image_loss": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "mass_error": 0.0,
    }

    stats = {
        "smoke_mean": ds.smoke_mean,
        "smoke_std": ds.smoke_std,
        "sensor_mean": ds.sensor_mean,
        "sensor_std": ds.sensor_std,
    }
    smoke_mean_gpu = ds.smoke_mean.to(device)
    smoke_std_gpu = ds.smoke_std.to(device)

    total_processed = 0
    global_idx_counter = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Scanning {original_run_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            sensor_preds, img_preds = model(batch)

            img_pred_denorm = denormalize(
                img_preds, smoke_mean_gpu, smoke_std_gpu
            ).clamp(0, 1)
            img_target_denorm = denormalize(
                batch["target_image"], smoke_mean_gpu, smoke_std_gpu
            ).clamp(0, 1)

            s_loss = nn.functional.mse_loss(sensor_preds, batch["sensor_target_vals"])
            i_loss = nn.functional.l1_loss(img_preds, batch["target_image"])

            b_psnr = psnr_metric(img_pred_denorm, img_target_denorm)
            b_ssim = ssim_metric(img_pred_denorm, img_target_denorm)

            mass_pred = img_pred_denorm.sum(dim=[1, 2, 3])
            mass_target = img_target_denorm.sum(dim=[1, 2, 3])
            b_mass_err = torch.abs(mass_pred - mass_target) / (mass_target + 1e-6)

            bs = img_preds.size(0)
            total_metrics["sensor_loss"] += s_loss.item() * bs
            total_metrics["image_loss"] += i_loss.item() * bs
            total_metrics["psnr"] += b_psnr.item() * bs
            total_metrics["ssim"] += b_ssim.item() * bs
            total_metrics["mass_error"] += b_mass_err.mean().item() * bs
            total_processed += bs

            per_sample_mse = torch.mean(
                (img_pred_denorm - img_target_denorm) ** 2, dim=[1, 2, 3]
            )

            for i in range(bs):
                err_val = per_sample_mse[i].item()
                sample_errors.append(
                    {
                        "id": global_idx_counter,
                        "error": err_val,
                        "input": batch["current_image"][i].cpu(),
                        "target": batch["target_image"][i].cpu(),
                        "pred": img_preds[i].cpu(),
                        "s_pred": sensor_preds[i].cpu(),
                        "s_target": batch["sensor_target_vals"][i].cpu(),
                        "s_pos": batch["sensor_pos"][i].cpu(),
                    }
                )
                global_idx_counter += 1

    for k in total_metrics:
        total_metrics[k] /= total_processed

    run_out_dir = os.path.join(base_output_dir, original_run_name)

    fixed_samples = [s for s in sample_errors if s["id"] in fixed_indices_set]
    fixed_samples.sort(key=lambda x: x["id"])
    if fixed_samples:
        save_visualization_packet(
            original_run_name,
            [x["id"] for x in fixed_samples],
            [x["input"] for x in fixed_samples],
            [x["target"] for x in fixed_samples],
            [x["pred"] for x in fixed_samples],
            [x["s_pred"] for x in fixed_samples],
            [x["s_target"] for x in fixed_samples],
            [x["s_pos"] for x in fixed_samples],
            stats,
            run_out_dir,
            prefix="FIXED_SET",
        )

    sample_errors.sort(key=lambda x: x["error"], reverse=True)
    worst_samples = sample_errors[:num_worst_cases]
    best_samples = sample_errors[-num_best_cases:]

    save_visualization_packet(
        original_run_name,
        [x["id"] for x in worst_samples],
        [x["input"] for x in worst_samples],
        [x["target"] for x in worst_samples],
        [x["pred"] for x in worst_samples],
        [x["s_pred"] for x in worst_samples],
        [x["s_target"] for x in worst_samples],
        [x["s_pos"] for x in worst_samples],
        stats,
        run_out_dir,
        prefix="WORST_CASE",
    )

    save_visualization_packet(
        original_run_name,
        [x["id"] for x in best_samples],
        [x["input"] for x in best_samples],
        [x["target"] for x in best_samples],
        [x["pred"] for x in best_samples],
        [x["s_pred"] for x in best_samples],
        [x["s_target"] for x in best_samples],
        [x["s_pos"] for x in best_samples],
        stats,
        run_out_dir,
        prefix="BEST_CASE",
    )

    del model
    torch.cuda.empty_cache()

    pretty_name = prettify_run_name(original_run_name)
    summary = {"run_name": pretty_name}
    summary.update(total_metrics)
    for k, v in exp_config.items():
        if k != "run_name":
            summary[k] = v

    return summary


def plot_comparison_grid(output_dir):
    print("Generating Structured Comparison Grids...")

    packet_paths = glob.glob(
        os.path.join(output_dir, "**", "FIXED_SET_data.pt"), recursive=True
    )

    all_packets = []
    for p in packet_paths:
        all_packets.append(torch.load(p))

    reference_indices = all_packets[0]["indices"]

    for i, seq_idx in enumerate(reference_indices):
        print(f"Creating comparison grid for Sequence {seq_idx}...")

        fig, axes = plt.subplots(4, 4, figsize=(20, 13))

        stats = all_packets[0]["stats"]
        smoke_mean, smoke_std = stats["smoke_mean"], stats["smoke_std"]
        sensor_mean, sensor_std = stats["sensor_mean"], stats["sensor_std"]

        in_d = denormalize(all_packets[0]["input_imgs"][i][0], smoke_mean, smoke_std)
        axes[0, 0].imshow(in_d, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[0, 0].set_title(
            f"Input ($t$)\nSeq: {seq_idx}", fontsize=14, fontweight="bold"
        )
        axes[0, 0].axis("off")

        tgt_d = denormalize(
            all_packets[0]["target_imgs"][i][0], smoke_mean, smoke_std
        ).clamp(0, 1)
        axes[0, 1].imshow(tgt_d, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[0, 1].set_title("Ground Truth ($t+25$)", fontsize=14, fontweight="bold")
        axes[0, 1].axis("off")

        s_pos = all_packets[0]["sensor_pos"][i]
        sx = ((s_pos[:, 0] / (192 / 128)) + 1) / 2 * 192
        sy = (s_pos[:, 1] + 1) / 2 * 128
        s_target_vals = denormalize(
            all_packets[0]["sensor_targets"][i], sensor_mean, sensor_std
        )[:, -1, :]
        axes[0, 1].quiver(
            sx,
            sy,
            s_target_vals[:, 0],
            s_target_vals[:, 1],
            color="cyan",
            scale=0.01,
            width=0.005,
            angles="xy",
            scale_units="xy",
        )

        axes[0, 2].axis("off")
        axes[0, 3].axis("off")

        for packet in all_packets:
            run_name = packet["run_name"]
            pred_d = denormalize(
                packet["pred_imgs"][i][0], smoke_mean, smoke_std
            ).clamp(0, 1)
            s_pred_vals = denormalize(
                packet["sensor_preds"][i], sensor_mean, sensor_std
            )[:, -1, :]

            r, c, subtitle = get_grid_location(run_name)
            plot_row = r + 1

            ax = axes[plot_row, c]
            ax.imshow(pred_d, cmap="gray", vmin=0, vmax=1, origin="lower")

            ax.quiver(
                sx,
                sy,
                s_pred_vals[:, 0],
                s_pred_vals[:, 1],
                color="cyan",
                scale=0.01,
                width=0.005,
                angles="xy",
                scale_units="xy",
            )

            ax.set_title(subtitle, fontsize=12)
            ax.axis("off")

        rows = [
            "Reference",
            "1-Step Upsampling",
            "2-Step Upsampling",
            "4-Step Upsampling",
        ]
        for idx, label in enumerate(rows):
            axes[idx, 0].text(
                -0.2,
                0.5,
                label,
                transform=axes[idx, 0].transAxes,
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                fontsize=14,
                fontweight="bold",
            )

        plt.suptitle(f"Model Comparison | Sequence {seq_idx}", fontsize=20, y=0.95)
        plt.subplots_adjust(wspace=0.1, hspace=0.15)

        save_path = os.path.join(output_dir, f"COMPARISON_GRID_seq_{seq_idx}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_packet(packet_path):
    data = torch.load(packet_path)
    original_run_name = data["run_name"]
    display_name = prettify_run_name(original_run_name)
    indices = data["indices"]
    prefix = data["prefix"]
    stats = data["stats"]
    output_dir = os.path.dirname(packet_path)
    smoke_mean, smoke_std = stats["smoke_mean"], stats["smoke_std"]
    sensor_mean, sensor_std = stats["sensor_mean"], stats["sensor_std"]

    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        in_d = denormalize(data["input_imgs"][i][0], smoke_mean, smoke_std)
        tgt_d = denormalize(data["target_imgs"][i][0], smoke_mean, smoke_std).clamp(
            0, 1
        )
        pred_d = denormalize(data["pred_imgs"][i][0], smoke_mean, smoke_std).clamp(0, 1)

        axes[0].imshow(in_d, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[0].set_title(f"Input ($t$)\nSeq ID: {idx}")
        axes[0].axis("off")
        s_pos = data["sensor_pos"][i]
        sx = ((s_pos[:, 0] / (192 / 128)) + 1) / 2 * 192
        sy = (s_pos[:, 1] + 1) / 2 * 128
        axes[1].imshow(tgt_d, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[1].set_title("Target ($t+25$)")
        axes[1].axis("off")
        s_target_vals = denormalize(data["sensor_targets"][i], sensor_mean, sensor_std)[
            :, -1, :
        ]
        axes[1].quiver(
            sx,
            sy,
            s_target_vals[:, 0],
            s_target_vals[:, 1],
            color="cyan",
            scale=0.01,
            width=0.005,
            angles="xy",
            scale_units="xy",
        )
        axes[2].imshow(pred_d, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        s_pred_vals = denormalize(data["sensor_preds"][i], sensor_mean, sensor_std)[
            :, -1, :
        ]
        axes[2].quiver(
            sx,
            sy,
            s_pred_vals[:, 0],
            s_pred_vals[:, 1],
            color="cyan",
            scale=0.01,
            width=0.005,
            angles="xy",
            scale_units="xy",
        )
        err = torch.abs(tgt_d - pred_d)
        im_err = axes[3].imshow(err, cmap="hot", vmin=0, vmax=1.0, origin="lower")
        axes[3].set_title(r"Error $|T - P|$")
        axes[3].axis("off")
        plt.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
        plt.suptitle(f"Run: {display_name} | Sequence {idx} ({prefix})", fontsize=14)
        save_path = os.path.join(output_dir, f"{prefix}_seq_{idx}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)


def plot_model_comparison(df, output_dir):
    print("Generating Model Comparison Plot")
    plt.figure(figsize=(14, 10))

    df["image_head_type"] = df["image_head_type"].replace(
        "baseline", "1 Step Upsampling"
    )
    df["image_head_type"] = df["image_head_type"].replace(
        "2_step_upsampling", "2 Step Upsampling"
    )
    df["image_head_type"] = df["image_head_type"].replace(
        "4_step_upsampling", "4 Step Upsampling"
    )
    head_types = sorted(df["image_head_type"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(head_types)))
    color_map = dict(zip(head_types, colors))
    upsample_types = df["upsampling_type"].unique()
    available_markers = ["o", "s", "^", "D", "v", "<", ">"]
    marker_map = {
        name: available_markers[i % len(available_markers)]
        for i, name in enumerate(upsample_types)
    }
    for _, row in df.iterrows():
        ht = row["image_head_type"]
        ut = row["upsampling_type"]
        loss = row["image_loss_type"] if "image_loss_type" in row else "N/A"
        plt.scatter(
            row["psnr"],
            row["ssim"],
            s=200,
            color=color_map[ht],
            marker=marker_map[ut],
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )
        label_text = loss.upper() if "L1" in loss.upper() else "MSE"
        plt.text(
            row["psnr"] + 0.05, row["ssim"] + 0.005, label_text, fontsize=9, alpha=0.7
        )
    color_handles = [
        mlines.Line2D(
            [],
            [],
            color=color_map[h],
            marker="o",
            linestyle="None",
            markersize=10,
            label=h.replace("_", r"\_"),
        )
        for h in head_types
    ]
    first_legend = plt.legend(
        handles=color_handles,
        title=r"\textbf{Architecture}",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    plt.gca().add_artist(first_legend)
    shape_handles = [
        mlines.Line2D(
            [],
            [],
            color="gray",
            marker=marker_map[u],
            linestyle="None",
            markersize=10,
            label=u.replace("_", r"\_").capitalize(),
        )
        for u in upsample_types
    ]
    plt.legend(
        handles=shape_handles,
        title=r"\textbf{Upsampling}",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.8),
    )
    plt.title(
        r"PSNR vs SSIM",
        fontsize=16,
    )
    plt.xlabel(r"PSNR (dB) $\rightarrow$", fontsize=12)
    plt.ylabel(r"SSIM $\rightarrow$", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.subplots_adjust(right=0.8)
    plot_path = os.path.join(output_dir, "model_comparison_scatter.pdf")
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    print(f"Comparison plot saved to: {plot_path}")


def modify_parser(parser):
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "plot"],
        default="inference",
        help="Mode: 'inference' runs model & saves data. 'plot' loads data & makes PDFs.",
    )
    parser.add_argument("--experiments_json", type=str, default="experiments.json")
    parser.add_argument("--data_root", type=str, default="smoke_simulation_data")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_worst_cases", type=int, default=25)
    parser.add_argument("--num_best_cases", type=int, default=25)
    parser.add_argument(
        "--num_fixed_samples",
        type=int,
        default=10,
        help="Number of fixed, random indices to save across all models.",
    )


def main(args):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

    if args.mode == "plot":
        print("Generating plots from saved data...")
        csv_path = os.path.join(args.output_dir, "master_summary.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plot_model_comparison(df, args.output_dir)
            print("LaTeX Table:")
            print(
                df.to_latex(
                    index=False, float_format="%.4f", caption="Results", escape=False
                )
            )
        else:
            print("Warning: master_summary.csv not found.")

        plot_comparison_grid(args.output_dir)

        pt_files = glob.glob(
            os.path.join(args.output_dir, "**", "*_data.pt"), recursive=True
        )
        for pt_file in tqdm(pt_files, desc="Plotting individual packets"):
            plot_packet(pt_file)

        print("Plotting complete.")

    elif args.mode == "inference":
        print("Running inference")
        if os.path.exists(args.output_dir):
            print(f"Clearing existing output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        if not os.path.exists(args.experiments_json):
            print(f"Error: {args.experiments_json} not found.")
            return

        with open(args.experiments_json, "r") as f:
            experiments = json.load(f)["experiments"]

        all_results = []
        for exp_config in experiments:
            result = run_inference(
                exp_config,
                args.data_root,
                args.device,
                args.output_dir,
                args.num_worst_cases,
                args.num_best_cases,
                args.num_fixed_samples,
            )
            all_results.append(result)

        df = pd.DataFrame(all_results)
        cols = [
            "run_name",
            "psnr",
            "ssim",
            "mass_error",
            "image_loss",
            "sensor_loss",
            "upsampling_type",
            "image_head_type",
            "image_loss_type",
        ]
        df = df[cols]
        csv_path = os.path.join(args.output_dir, "master_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        tex_path = os.path.join(args.output_dir, "master_summary.tex")
        df.to_latex(buf=tex_path, index=False, float_format="%.4f", escape=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modify_parser(parser)
    args = parser.parse_args()
    main(args)
