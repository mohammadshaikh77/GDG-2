"""Plotting utilities for all transformer redundancy experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def parse_args() -> argparse.Namespace:
    """Returns parsed command-line arguments for plotting."""
    parser = argparse.ArgumentParser(description="Generate experiment plots from .npz metrics files.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
    return parser.parse_args()


def ensure_out_dir(out_dir: str) -> None:
    """Returns nothing and creates the output directory if needed."""
    os.makedirs(out_dir, exist_ok=True)


def save_figure(fig: plt.Figure, out_dir: str, exp_name: str, plot_num: int, description: str, fmt: str) -> str:
    """Returns the saved plot path after writing the figure to disk."""
    filename = f"{exp_name}_plot{plot_num}_{description}.{fmt}"
    path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(path, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return path


def infer_variant(exp_name: str, config: dict) -> str:
    """Returns a canonical variant label inferred from saved config."""
    if exp_name == "exp1":
        return "baseline"
    if exp_name == "exp2":
        return f"D{int(config['n_layer'])}"
    if exp_name == "exp3":
        return f"W{int(config['n_embd'])}"
    if exp_name == "exp4":
        return "component"
    if exp_name == "exp5":
        return "diagnostic"
    if exp_name == "exp6":
        alpha = float(config["residual_alpha"])
        return {
            0.5: "RS05",
            1.0: "RS10",
            1.5: "RS15",
            2.0: "RS20",
        }.get(alpha, f"RS{str(alpha).replace('.', '')}")
    return str(config.get("exp_name", exp_name))


def get_variant_order(exp_name: str) -> List[str]:
    """Returns the canonical plotting order for experiment variants."""
    mapping = {
        "exp1": ["baseline"],
        "exp2": ["D3", "D6", "D12", "D24"],
        "exp3": ["W128", "W256", "W512"],
        "exp4": ["component"],
        "exp5": ["diagnostic"],
        "exp6": ["RS05", "RS10", "RS15", "RS20"],
    }
    return mapping[exp_name]


def get_color_map(exp_name: str) -> Dict[str, tuple]:
    """Returns a consistent tab10 color assignment for the experiment variants."""
    colors = plt.get_cmap("tab10").colors
    order = get_variant_order(exp_name)
    return {variant: colors[i % len(colors)] for i, variant in enumerate(order)}


def load_records(exp_name: str, results_dir: str) -> List[dict]:
    """Returns a list of loaded metric records matching the experiment prefix."""
    pattern = os.path.join(results_dir, f"{exp_name}*_seed*_step*_metrics.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No metric files found matching {pattern}")

    records: List[dict] = []
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        config_raw = npz["config"]
        if isinstance(config_raw, np.ndarray):
            config_str = str(config_raw.item())
        else:
            config_str = str(config_raw)
        config = json.loads(config_str)
        record = {
            "path": path,
            "seed": int(config.get("seed", 0)),
            "step": int(npz["step"].item()),
            "config": config,
            "variant": infer_variant(exp_name, config),
            "metrics": {key: npz[key] for key in npz.files},
        }
        records.append(record)
    return records


def group_records_by_variant_and_step(records: Sequence[dict]) -> Dict[str, Dict[int, List[dict]]]:
    """Returns a nested mapping variant -> step -> list of records."""
    grouped: Dict[str, Dict[int, List[dict]]] = {}
    for record in records:
        grouped.setdefault(record["variant"], {}).setdefault(record["step"], []).append(record)
    return grouped


def stack_metric(records: Sequence[dict], key: str) -> np.ndarray:
    """Returns a stacked array of one metric over a list of records."""
    arrays = []
    for record in records:
        if key not in record["metrics"]:
            raise KeyError(f"Metric key '{key}' missing from {record['path']}")
        arrays.append(np.asarray(record["metrics"][key], dtype=np.float32))
    return np.stack(arrays, axis=0)


def mean_std(records: Sequence[dict], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mean and standard deviation of a metric across records."""
    stacked = stack_metric(records, key)
    return stacked.mean(axis=0), stacked.std(axis=0)


def make_line_band_plot(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: tuple,
    label: str,
) -> None:
    """Returns nothing and draws a mean line with a shaded standard-deviation band."""
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)


def normalized_depth_x(num_layers: int) -> np.ndarray:
    """Returns normalized depth coordinates for layers 1..L."""
    return np.arange(1, num_layers + 1, dtype=np.float32) / float(num_layers)


def fsrank_from_records(records: Sequence[dict], width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mean and standard deviation of fractional stable rank."""
    srank = stack_metric(records, "srank")
    fsrank = srank / float(width)
    return fsrank.mean(axis=0), fsrank.std(axis=0)


def topk_from_eigenvalues(eigenvalues: np.ndarray, ks: Sequence[int]) -> np.ndarray:
    """Returns cumulative spectral dominance values for the requested k values."""
    total = np.sum(eigenvalues)
    if total <= 0.0:
        return np.zeros(len(ks), dtype=np.float32)
    cumulative = np.cumsum(eigenvalues) / total
    values = []
    for k in ks:
        index = min(max(k, 1), len(eigenvalues)) - 1
        values.append(float(cumulative[index]))
    return np.asarray(values, dtype=np.float32)


def parse_log_variant(exp_name: str, filename: str) -> str:
    """Returns the variant label inferred from a CSV log filename."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    suffix = stem.rsplit("_seed", 1)[0]
    if exp_name == "exp6":
        mapping = {
            "exp6_alpha0_5": "RS05",
            "exp6_alpha1_0": "RS10",
            "exp6_alpha1_5": "RS15",
            "exp6_alpha2_0": "RS20",
        }
        if suffix in mapping:
            return mapping[suffix]
    return suffix


def load_csv_logs(exp_name: str, results_dir: str) -> Dict[str, Dict[int, List[dict]]]:
    """Returns variant -> seed -> rows for CSV training logs."""
    pattern = os.path.join(results_dir, f"{exp_name}*_seed*_log.csv")
    paths = sorted(glob.glob(pattern))
    grouped: Dict[str, Dict[int, List[dict]]] = {}
    for path in paths:
        variant = parse_log_variant(exp_name, path)
        seed_str = os.path.basename(path).split("_seed", 1)[1].split("_log.csv", 1)[0]
        seed = int(seed_str)
        rows: List[dict] = []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "step": int(row["step"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                        "grad_norm": float(row["grad_norm"]),
                        "lr": float(row["lr"]),
                    }
                )
        grouped.setdefault(variant, {})[seed] = rows
    return grouped


def aggregate_log_metric(rows_by_seed: Dict[int, List[dict]], key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns aligned steps, mean values, and standard deviations for a logged scalar metric."""
    seeds = sorted(rows_by_seed.keys())
    if not seeds:
        raise ValueError("No log rows available to aggregate.")
    steps = np.asarray([row["step"] for row in rows_by_seed[seeds[0]]], dtype=np.int64)
    values = []
    for seed in seeds:
        seed_rows = rows_by_seed[seed]
        values.append(np.asarray([row[key] for row in seed_rows], dtype=np.float32))
    stacked = np.stack(values, axis=0)
    return steps, stacked.mean(axis=0), stacked.std(axis=0)


def plot_exp1(records: Sequence[dict], out_dir: str, fmt: str) -> int:
    """Returns the number of saved plots for experiment 1."""
    saved = 0
    grouped = group_records_by_variant_and_step(records)["baseline"]
    colors = plt.get_cmap("tab10").colors
    steps = [500, 1000, 2000, 10000, 50000, 200000]
    available_steps = [step for step in steps if step in grouped]
    x = np.arange(stack_metric(grouped[available_steps[0]], "erank").shape[1], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, step in enumerate(available_steps):
        mean, std = mean_std(grouped[step], "erank")
        make_line_band_plot(ax, x, mean, std, colors[i], f"step {step}")
    ax.set_title("Effective Rank vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank")
    ax.set_xticks(x)
    ax.legend()
    save_figure(fig, out_dir, "exp1", 1, "erank_vs_layer", fmt)
    saved += 1

    if 200000 in grouped:
        erank_mean, _ = mean_std(grouped[200000], "erank")
        srank_mean, _ = mean_std(grouped[200000], "srank")
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(x, erank_mean, color=colors[0], linewidth=2, label="erank")
        ax2.plot(x, srank_mean, color=colors[1], linewidth=2, label="srank")
        ax1.set_title("Effective and Stable Rank at Final Checkpoint")
        ax1.set_xlabel("Layer index")
        ax1.set_ylabel("Effective rank")
        ax2.set_ylabel("Stable rank")
        ax1.set_xticks(x)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="best")
        save_figure(fig, out_dir, "exp1", 2, "srank_vs_layer", fmt)
        saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, step in enumerate(available_steps):
        mean, std = mean_std(grouped[step], "delta")
        make_line_band_plot(ax, x, mean, std, colors[i], f"step {step}")
    ax.set_title("Eigenvalue Dominance vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Top eigenvalue variance fraction")
    ax.set_xticks(x)
    ax.legend()
    save_figure(fig, out_dir, "exp1", 3, "delta_vs_layer", fmt)
    saved += 1

    if 200000 in grouped:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        final_records = grouped[200000]
        selected_layers = [0, 1, 3, 6]
        for ax, layer in zip(axes.flat, selected_layers):
            spectra = stack_metric(final_records, "eigenvalues")[:, layer, :]
            for spectrum in spectra:
                ax.plot(np.arange(1, len(spectrum) + 1), spectrum, linewidth=1)
            ax.set_title(f"Layer {layer}")
            ax.set_xlabel("Eigenvalue index")
            ax.set_ylabel("Eigenvalue magnitude")
            ax.set_yscale("log")
        save_figure(fig, out_dir, "exp1", 4, "eigenvalue_spectra", fmt)
        saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, step in enumerate(available_steps):
        mean, std = mean_std(grouped[step], "mean_cosine")
        make_line_band_plot(ax, x, mean, std, colors[i], f"step {step}")
    ax.set_title("Mean Cosine Similarity vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xticks(x)
    ax.legend()
    save_figure(fig, out_dir, "exp1", 5, "mean_cosine_vs_layer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, step in enumerate(available_steps):
        mean, std = mean_std(grouped[step], "var_cosine")
        make_line_band_plot(ax, x, mean, std, colors[i], f"step {step}")
    ax.set_title("Variance of Cosine Similarity vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity variance")
    ax.set_xticks(x)
    ax.legend()
    save_figure(fig, out_dir, "exp1", 6, "var_cosine_vs_layer", fmt)
    saved += 1

    if 200000 in grouped:
        cka_mean = stack_metric(grouped[200000], "cka_matrix").mean(axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cka_mean, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title("CKA Heatmap at Final Checkpoint")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Layer index")
        for i in range(cka_mean.shape[0]):
            for j in range(cka_mean.shape[1]):
                ax.text(j, i, f"{cka_mean[i, j]:.2f}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_figure(fig, out_dir, "exp1", 7, "cka_heatmap_final", fmt)
        saved += 1

    temporal_steps = [1000, 2000, 10000, 50000, 200000]
    present_temporal_steps = [step for step in temporal_steps if step in grouped]
    if present_temporal_steps:
        fig, axes = plt.subplots(1, len(present_temporal_steps), figsize=(4 * len(present_temporal_steps), 4))
        if len(present_temporal_steps) == 1:
            axes = [axes]
        for ax, step in zip(axes, present_temporal_steps):
            cka_mean = stack_metric(grouped[step], "cka_matrix").mean(axis=0)
            im = ax.imshow(cka_mean, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(f"step {step}")
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Layer index")
            for i in range(cka_mean.shape[0]):
                for j in range(cka_mean.shape[1]):
                    ax.text(j, i, f"{cka_mean[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
        save_figure(fig, out_dir, "exp1", 8, "cka_heatmap_temporal", fmt)
        saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, step in enumerate(available_steps):
        mean, std = mean_std(grouped[step], "redundancy_index")
        make_line_band_plot(ax, x, mean, std, colors[i], f"step {step}")
    ax.set_title("Redundancy Index vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Redundancy index")
    ax.set_xticks(x)
    ax.legend()
    save_figure(fig, out_dir, "exp1", 9, "redundancy_index_vs_layer", fmt)
    saved += 1

    return saved


def plot_exp2(records: Sequence[dict], out_dir: str, fmt: str) -> int:
    """Returns the number of saved plots for experiment 2."""
    saved = 0
    grouped = group_records_by_variant_and_step(records)
    color_map = get_color_map("exp2")
    final_step = 200000

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp2"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "erank")
        num_layers = len(mean) - 1
        x = normalized_depth_x(num_layers)
        make_line_band_plot(ax, x, mean[1:], std[1:], color_map[variant], variant)
    ax.set_title("Effective Rank vs Normalized Depth")
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Effective rank")
    ax.legend()
    save_figure(fig, out_dir, "exp2", 1, "erank_vs_normdepth", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp2"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "erank")
        x = np.arange(1, len(mean), dtype=np.int64)
        make_line_band_plot(ax, x, mean[1:], std[1:], color_map[variant], variant)
    ax.set_title("Effective Rank vs Absolute Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank")
    ax.legend()
    save_figure(fig, out_dir, "exp2", 2, "erank_vs_abslayer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp2"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "redundancy_index")
        num_layers = len(mean) - 1
        x = normalized_depth_x(num_layers)
        make_line_band_plot(ax, x, mean[1:], std[1:], color_map[variant], variant)
    ax.set_title("Redundancy Index vs Normalized Depth")
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Redundancy index")
    ax.legend()
    save_figure(fig, out_dir, "exp2", 3, "redundancy_vs_normdepth", fmt)
    saved += 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    abs_means = []
    abs_stds = []
    norm_means = []
    norm_stds = []
    labels = []
    for variant in get_variant_order("exp2"):
        if final_step not in grouped.get(variant, {}):
            continue
        erank = stack_metric(grouped[variant][final_step], "erank")[:, 1:]
        depth = erank.shape[1]
        argmin = np.argmin(erank, axis=1) + 1
        abs_means.append(argmin.mean())
        abs_stds.append(argmin.std())
        norm = argmin / float(depth)
        norm_means.append(norm.mean())
        norm_stds.append(norm.std())
        labels.append(variant)
    xpos = np.arange(len(labels))
    axes[0].bar(xpos, abs_means, yerr=abs_stds, color=[color_map[label] for label in labels], capsize=4)
    axes[0].set_title("Absolute Argmin Layer")
    axes[0].set_xlabel("Variant")
    axes[0].set_ylabel("Layer index")
    axes[0].set_xticks(xpos, labels)
    axes[1].bar(xpos, norm_means, yerr=norm_stds, color=[color_map[label] for label in labels], capsize=4)
    axes[1].set_title("Normalized Argmin Layer")
    axes[1].set_xlabel("Variant")
    axes[1].set_ylabel("Normalized layer index")
    axes[1].set_xticks(xpos, labels)
    save_figure(fig, out_dir, "exp2", 4, "argmin_layer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    depths = []
    slope_means = []
    slope_stds = []
    for variant in get_variant_order("exp2"):
        if final_step not in grouped.get(variant, {}):
            continue
        erank = stack_metric(grouped[variant][final_step], "erank")
        depth = erank.shape[1] - 1
        idx = int(math.ceil(depth / 3.0))
        slope = (erank[:, idx] - erank[:, 1]) / float(depth / 3.0)
        depths.append(depth)
        slope_means.append(slope.mean())
        slope_stds.append(slope.std())
    ax.errorbar(depths, slope_means, yerr=slope_stds, fmt="o-", linewidth=2, capsize=4)
    ax.set_title("Early Rank Slope vs Depth")
    ax.set_xlabel("Model depth")
    ax.set_ylabel("Early slope")
    save_figure(fig, out_dir, "exp2", 5, "early_slope_vs_depth", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp2"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "delta")
        num_layers = len(mean) - 1
        x = normalized_depth_x(num_layers)
        make_line_band_plot(ax, x, mean[1:], std[1:], color_map[variant], variant)
    ax.set_title("Eigenvalue Dominance vs Normalized Depth")
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Top eigenvalue variance fraction")
    ax.legend()
    save_figure(fig, out_dir, "exp2", 6, "delta_vs_normdepth", fmt)
    saved += 1

    steps = [2000, 50000, 200000]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, step in zip(axes, steps):
        for variant in ["D6", "D12"]:
            if step not in grouped.get(variant, {}):
                continue
            mean, std = mean_std(grouped[variant][step], "erank")
            depth = len(mean) - 1
            x = normalized_depth_x(depth)
            make_line_band_plot(ax, x, mean[1:], std[1:], color_map[variant], variant)
        ax.set_title(f"step {step}")
        ax.set_xlabel("Normalized depth")
        ax.set_ylabel("Effective rank")
        ax.legend()
    save_figure(fig, out_dir, "exp2", 7, "erank_temporal_D6_D12", fmt)
    saved += 1

    return saved


def plot_exp3(records: Sequence[dict], out_dir: str, fmt: str) -> int:
    """Returns the number of saved plots for experiment 3."""
    saved = 0
    grouped = group_records_by_variant_and_step(records)
    color_map = get_color_map("exp3")
    final_step = 200000

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp3"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "ferank")
        layers = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, layers, mean, std, color_map[variant], variant)
    ax.set_title("Fractional Effective Rank vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Fractional effective rank")
    ax.legend()
    save_figure(fig, out_dir, "exp3", 1, "ferank_vs_layer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp3"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "erank")
        layers = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, layers, mean, std, color_map[variant], variant)
    ax.set_title("Effective Rank vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank")
    ax.legend()
    save_figure(fig, out_dir, "exp3", 2, "erank_vs_layer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp3"):
        if final_step not in grouped.get(variant, {}):
            continue
        width = int(grouped[variant][final_step][0]["config"]["n_embd"])
        mean, std = fsrank_from_records(grouped[variant][final_step], width)
        layers = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, layers, mean, std, color_map[variant], variant)
    ax.set_title("Fractional Stable Rank vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Fractional stable rank")
    ax.legend()
    save_figure(fig, out_dir, "exp3", 3, "fsrank_vs_layer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp3"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "delta")
        layers = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, layers, mean, std, color_map[variant], variant)
    ax.set_title("Eigenvalue Dominance vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Top eigenvalue variance fraction")
    ax.legend()
    save_figure(fig, out_dir, "exp3", 4, "delta_vs_layer", fmt)
    saved += 1

    ks = [1, 5, 10, 20, 50]
    ref_records = grouped["W256"][final_step]
    ref_erank_mean = stack_metric(ref_records, "erank").mean(axis=0)
    ref_layer = int(np.argmin(ref_erank_mean[1:]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp3"):
        if final_step not in grouped.get(variant, {}):
            continue
        spectra = stack_metric(grouped[variant][final_step], "eigenvalues")[:, ref_layer, :]
        topk = np.stack([topk_from_eigenvalues(spectrum, ks) for spectrum in spectra], axis=0)
        mean = topk.mean(axis=0)
        std = topk.std(axis=0)
        ax.plot(ks, mean, color=color_map[variant], linewidth=2, label=variant)
        ax.fill_between(ks, mean - std, mean + std, color=color_map[variant], alpha=0.2)
    ax.set_title(f"Top-k Dominance at Reference Layer {ref_layer}")
    ax.set_xlabel("k")
    ax.set_ylabel("Cumulative variance fraction")
    ax.legend()
    save_figure(fig, out_dir, "exp3", 5, "topk_dominance", fmt)
    saved += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for panel_idx, layer in enumerate([1, 6]):
        ax = axes[panel_idx]
        for variant in get_variant_order("exp3"):
            if final_step not in grouped.get(variant, {}):
                continue
            spectra = stack_metric(grouped[variant][final_step], "eigenvalues")[:, layer, :]
            for spectrum in spectra:
                probs = spectrum / max(spectrum.sum(), 1e-12)
                x = np.linspace(0.0, 1.0, len(probs))
                ax.plot(x, probs, color=color_map[variant], linewidth=1, alpha=0.8)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Normalized eigenvalue index")
        ax.set_ylabel("Normalized eigenvalue magnitude")
        ax.set_yscale("log")
    save_figure(fig, out_dir, "exp3", 6, "eigenvalue_spectra", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp3"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "redundancy_index")
        layers = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, layers, mean, std, color_map[variant], variant)
    ax.set_title("Redundancy Index vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Redundancy index")
    ax.legend()
    save_figure(fig, out_dir, "exp3", 7, "redundancy_vs_layer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp3"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "mean_cosine")
        layers = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, layers, mean, std, color_map[variant], variant)
    ax.set_title("Mean Cosine Similarity vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean cosine similarity")
    ax.legend()
    save_figure(fig, out_dir, "exp3", 8, "cosine_vs_layer", fmt)
    saved += 1

    return saved


def plot_exp4(records: Sequence[dict], out_dir: str, fmt: str) -> int:
    """Returns the number of saved plots for experiment 4."""
    saved = 0
    grouped = group_records_by_variant_and_step(records)["component"]
    final_step = 200000
    if final_step not in grouped:
        raise ValueError("Experiment 4 requires final-checkpoint metrics at step 200000.")

    input_erank = stack_metric(grouped[final_step], "erank_input")
    post_erank = stack_metric(grouped[final_step], "erank_post_attn")
    output_erank = stack_metric(grouped[final_step], "erank_output")
    input_red = stack_metric(grouped[final_step], "redundancy_index_input")
    post_red = stack_metric(grouped[final_step], "redundancy_index_post_attn")
    output_red = stack_metric(grouped[final_step], "redundancy_index_output")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    stage_x = np.arange(3)
    stage_labels = ["input", "post-attn", "output"]
    for layer_idx, ax in enumerate(axes.flat):
        stage_values = np.stack(
            [input_erank[:, layer_idx], post_erank[:, layer_idx], output_erank[:, layer_idx]],
            axis=1,
        )
        mean = stage_values.mean(axis=0)
        std = stage_values.std(axis=0)
        ax.plot(stage_x, mean, color=plt.get_cmap("tab10").colors[0], linewidth=2)
        ax.fill_between(stage_x, mean - std, mean + std, color=plt.get_cmap("tab10").colors[0], alpha=0.2)
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_xticks(stage_x, stage_labels)
        ax.set_ylabel("Effective rank")
    save_figure(fig, out_dir, "exp4", 1, "rank_progression", fmt)
    saved += 1

    layers = np.arange(1, input_erank.shape[1] + 1)
    delta_rank_attn = post_erank - input_erank
    delta_rank_mlp = output_erank - post_erank
    fig, ax = plt.subplots(figsize=(8, 5))
    for values, label, color in [
        (delta_rank_attn, "delta_rank_attn", plt.get_cmap("tab10").colors[0]),
        (delta_rank_mlp, "delta_rank_mlp", plt.get_cmap("tab10").colors[1]),
    ]:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(layers, mean, color=color, linewidth=2, label=label)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.2)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Rank Change by Component")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Change in effective rank")
    ax.legend()
    save_figure(fig, out_dir, "exp4", 2, "rank_change_by_component", fmt)
    saved += 1

    novelty_attn = stack_metric(grouped[final_step], "novelty_attn")
    novelty_mlp = stack_metric(grouped[final_step], "novelty_mlp")
    fig, ax = plt.subplots(figsize=(8, 5))
    for values, label, color in [
        (novelty_attn, "novelty_attn", plt.get_cmap("tab10").colors[0]),
        (novelty_mlp, "novelty_mlp", plt.get_cmap("tab10").colors[1]),
    ]:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(layers, mean, color=color, linewidth=2, label=label)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_title("Subspace Novelty Effective Rank")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank of orthogonal component")
    ax.legend()
    save_figure(fig, out_dir, "exp4", 3, "novelty_erank", fmt)
    saved += 1

    novelty_frac_attn = stack_metric(grouped[final_step], "novelty_frac_attn")
    novelty_frac_mlp = stack_metric(grouped[final_step], "novelty_frac_mlp")
    fig, ax = plt.subplots(figsize=(8, 5))
    for values, label, color in [
        (novelty_frac_attn, "novelty_frac_attn", plt.get_cmap("tab10").colors[0]),
        (novelty_frac_mlp, "novelty_frac_mlp", plt.get_cmap("tab10").colors[1]),
    ]:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(layers, mean, color=color, linewidth=2, label=label)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_title("Novelty Fraction vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Novelty fraction")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    save_figure(fig, out_dir, "exp4", 4, "novelty_fraction", fmt)
    saved += 1

    temporal_steps = [2000, 50000, 200000]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, step in zip(axes, temporal_steps):
        values = stack_metric(grouped[step], "novelty_frac_attn")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(layers, mean, color=plt.get_cmap("tab10").colors[0], linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=plt.get_cmap("tab10").colors[0], alpha=0.2)
        ax.set_title(f"step {step}")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Attention novelty fraction")
    save_figure(fig, out_dir, "exp4", 5, "novelty_attn_temporal", fmt)
    saved += 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, step in zip(axes, temporal_steps):
        values = stack_metric(grouped[step], "novelty_frac_mlp")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(layers, mean, color=plt.get_cmap("tab10").colors[1], linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=plt.get_cmap("tab10").colors[1], alpha=0.2)
        ax.set_title(f"step {step}")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("MLP novelty fraction")
    save_figure(fig, out_dir, "exp4", 6, "novelty_mlp_temporal", fmt)
    saved += 1

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for layer_idx, ax in enumerate(axes.flat):
        stage_values = np.stack(
            [input_red[:, layer_idx], post_red[:, layer_idx], output_red[:, layer_idx]],
            axis=1,
        )
        mean = stage_values.mean(axis=0)
        std = stage_values.std(axis=0)
        ax.plot(stage_x, mean, color=plt.get_cmap("tab10").colors[2], linewidth=2)
        ax.fill_between(stage_x, mean - std, mean + std, color=plt.get_cmap("tab10").colors[2], alpha=0.2)
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_xticks(stage_x, stage_labels)
        ax.set_ylabel("Redundancy index")
    save_figure(fig, out_dir, "exp4", 7, "redundancy_through_block", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for values, label, color in [
        (stack_metric(grouped[final_step], "mean_cosine_input"), "input", plt.get_cmap("tab10").colors[0]),
        (stack_metric(grouped[final_step], "mean_cosine_post_attn"), "post-attn", plt.get_cmap("tab10").colors[1]),
        (stack_metric(grouped[final_step], "mean_cosine_output"), "output", plt.get_cmap("tab10").colors[2]),
    ]:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(layers, mean, color=color, linewidth=2, label=label)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_title("Cosine Similarity Through the Block")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean cosine similarity")
    ax.legend()
    save_figure(fig, out_dir, "exp4", 8, "cosine_through_block", fmt)
    saved += 1

    return saved


def plot_exp5(records: Sequence[dict], out_dir: str, fmt: str) -> int:
    """Returns the number of saved plots for experiment 5."""
    del records
    del out_dir
    del fmt
    raise ValueError(
        "Experiment 5 plots require position-group cosine and attention-entropy metrics, "
        "but the current measurement files do not contain those arrays."
    )


def plot_exp6(records: Sequence[dict], results_dir: str, out_dir: str, fmt: str) -> int:
    """Returns the number of saved plots for experiment 6."""
    saved = 0
    grouped = group_records_by_variant_and_step(records)
    color_map = get_color_map("exp6")
    final_step = 200000

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp6"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "erank")
        x = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, x, mean, std, color_map[variant], variant)
    ax.set_title("Effective Rank vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank")
    ax.legend()
    save_figure(fig, out_dir, "exp6", 1, "erank_vs_layer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    alphas = []
    argmin_means = []
    argmin_stds = []
    alpha_map = {"RS05": 0.5, "RS10": 1.0, "RS15": 1.5, "RS20": 2.0}
    for variant in get_variant_order("exp6"):
        if final_step not in grouped.get(variant, {}):
            continue
        erank = stack_metric(grouped[variant][final_step], "erank")[:, 1:]
        argmin = np.argmin(erank, axis=1) + 1
        alphas.append(alpha_map[variant])
        argmin_means.append(argmin.mean())
        argmin_stds.append(argmin.std())
    ax.errorbar(alphas, argmin_means, yerr=argmin_stds, fmt="o-", linewidth=2, capsize=4)
    ax.set_title("Argmin Layer vs Residual Alpha")
    ax.set_xlabel("Residual alpha")
    ax.set_ylabel("Argmin layer")
    save_figure(fig, out_dir, "exp6", 2, "argmin_vs_alpha", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp6"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "redundancy_index")
        x = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, x, mean, std, color_map[variant], variant)
    ax.set_title("Redundancy Index vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Redundancy index")
    ax.legend()
    save_figure(fig, out_dir, "exp6", 3, "redundancy_vs_layer", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp6"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "delta")
        x = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, x, mean, std, color_map[variant], variant)
    ax.set_title("Eigenvalue Dominance vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Top eigenvalue variance fraction")
    ax.legend()
    save_figure(fig, out_dir, "exp6", 4, "delta_vs_layer", fmt)
    saved += 1

    temporal_steps = [500, 1000, 2000, 10000, 50000, 200000]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, step in zip(axes.flat, temporal_steps):
        for variant in get_variant_order("exp6"):
            if step not in grouped.get(variant, {}):
                continue
            mean, std = mean_std(grouped[variant][step], "erank")
            x = np.arange(len(mean), dtype=np.int64)
            make_line_band_plot(ax, x, mean, std, color_map[variant], variant)
        ax.set_title(f"step {step}")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Effective rank")
        ax.legend()
    save_figure(fig, out_dir, "exp6", 5, "erank_temporal", fmt)
    saved += 1

    if not any("activation_norm" in record["metrics"] for record in records):
        raise ValueError("Experiment 6 activation-norm plot requires 'activation_norm' in measurement files.")
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp6"):
        if final_step not in grouped.get(variant, {}):
            continue
        mean, std = mean_std(grouped[variant][final_step], "activation_norm")
        x = np.arange(len(mean), dtype=np.int64)
        make_line_band_plot(ax, x, mean, std, color_map[variant], variant)
    ax.set_title("Activation Norm vs Layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean activation L2 norm")
    ax.legend()
    save_figure(fig, out_dir, "exp6", 6, "activation_norm", fmt)
    saved += 1

    logs = load_csv_logs("exp6", results_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp6"):
        if variant not in logs:
            continue
        steps, mean, std = aggregate_log_metric(logs[variant], "grad_norm")
        make_line_band_plot(ax, steps.astype(np.float32), mean, std, color_map[variant], variant)
    ax.set_title("Gradient Norm vs Training Step")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient norm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    save_figure(fig, out_dir, "exp6", 7, "grad_norm_vs_step", fmt)
    saved += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in get_variant_order("exp6"):
        if variant not in logs:
            continue
        steps, mean, std = aggregate_log_metric(logs[variant], "val_loss")
        make_line_band_plot(ax, steps.astype(np.float32), mean, std, color_map[variant], variant)
    ax.set_title("Validation Loss vs Training Step")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_xscale("log")
    ax.legend()
    save_figure(fig, out_dir, "exp6", 8, "val_loss_vs_step", fmt)
    saved += 1

    return saved


def dispatch(exp_name: str, records: Sequence[dict], results_dir: str, out_dir: str, fmt: str) -> int:
    """Returns the number of plots saved for the requested experiment."""
    if exp_name == "exp1":
        return plot_exp1(records, out_dir, fmt)
    if exp_name == "exp2":
        return plot_exp2(records, out_dir, fmt)
    if exp_name == "exp3":
        return plot_exp3(records, out_dir, fmt)
    if exp_name == "exp4":
        return plot_exp4(records, out_dir, fmt)
    if exp_name == "exp5":
        return plot_exp5(records, out_dir, fmt)
    if exp_name == "exp6":
        return plot_exp6(records, results_dir, out_dir, fmt)
    raise ValueError(f"Unsupported experiment name: {exp_name}")


if __name__ == "__main__":
    args = parse_args()
    ensure_out_dir(args.out_dir)
    records = load_records(args.exp_name, args.results_dir)
    n_saved = dispatch(args.exp_name, records, args.results_dir, args.out_dir, args.format)
    print(f"Saved {n_saved} plot(s) to {args.out_dir}")
