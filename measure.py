"""Unified measurement script for all transformer redundancy experiments."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.data import get_sequential_chunks, load_wikitext103
from src.metrics import compute_all_metrics, erank, linear_CKA, mean_cosine_sim
from src.model import GPT, GPTConfig


CKA_SUBSAMPLE = 4096
CKA_SEED = 42
EXP5_COSINE_PAIRS = 1000
EXP5_POSITION_GROUPS = [
    ("early", 0, 32),
    ("mid_early", 32, 64),
    ("mid_late", 64, 96),
    ("late", 96, 128),
]


def parse_args() -> argparse.Namespace:
    """Returns parsed command-line arguments for measurement."""
    parser = argparse.ArgumentParser(description="Measure hidden-state metrics from a GPT checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_sequences", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=128)
    return parser.parse_args()


def ensure_out_dir(out_dir: str) -> None:
    """Returns nothing and creates the output directory if needed."""
    os.makedirs(out_dir, exist_ok=True)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Returns the checkpoint dictionary loaded from disk."""
    return torch.load(checkpoint_path, map_location="cpu")


def make_model_from_checkpoint(checkpoint: dict, device: torch.device) -> GPT:
    """Returns a GPT model reconstructed from the checkpoint config and weights."""
    config_dict = checkpoint["config"]
    model_config = GPTConfig(
        n_layer=int(config_dict["n_layer"]),
        n_embd=int(config_dict["n_embd"]),
        n_head=int(config_dict["n_head"]),
        block_size=int(config_dict["block_size"]),
        vocab_size=int(config_dict["vocab_size"]),
        dropout=float(config_dict.get("dropout", 0.0)),
        residual_alpha=float(config_dict.get("residual_alpha", 1.0)),
    )
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model


def disable_all_dropout(model: nn.Module) -> None:
    """Returns nothing and disables dropout modules in-place."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def load_vocab_json(vocab_path: str) -> dict:
    """Returns the saved training vocabulary as a character-to-id dictionary."""
    with open(vocab_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_vocab_path(checkpoint_path: str, exp_name: str, checkpoint: dict) -> str:
    """Returns the vocabulary JSON path saved alongside the original training checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config = checkpoint.get("config", {})
    trained_exp_name = config.get("exp_name")

    candidate_names = []
    if isinstance(trained_exp_name, str) and trained_exp_name:
        candidate_names.append(trained_exp_name)
    if exp_name not in candidate_names:
        candidate_names.append(exp_name)

    for candidate in candidate_names:
        vocab_path = os.path.join(checkpoint_dir, f"{candidate}_vocab.json")
        if os.path.exists(vocab_path):
            return vocab_path

    return os.path.join(checkpoint_dir, f"{candidate_names[0]}_vocab.json")


def flatten_and_center(hidden_state: torch.Tensor) -> torch.Tensor:
    """Returns a centered 2D float32 CPU matrix of shape (tokens, width)."""
    matrix = hidden_state.reshape(-1, hidden_state.shape[-1]).float()
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    return matrix.detach().cpu().float()


def flatten_without_center(hidden_state: torch.Tensor) -> torch.Tensor:
    """Returns a 2D float32 CPU matrix of shape (tokens, width) without centering."""
    return hidden_state.reshape(-1, hidden_state.shape[-1]).detach().cpu().float()


def compute_activation_norms(residual_states: List[torch.Tensor]) -> np.ndarray:
    """Returns per-layer mean L2 activation norms for residual states."""
    values: List[float] = []
    for state in residual_states:
        flattened = flatten_without_center(state)
        mean_norm = torch.linalg.norm(flattened, dim=1).mean()
        values.append(float(mean_norm.item()))
    return np.asarray(values, dtype=np.float32)


def compute_exp5_position_metrics(residual_states_raw: List[torch.Tensor]) -> Dict[str, np.ndarray]:
    """Returns experiment-5 cosine metrics stratified by token groups and individual positions."""
    n_layers = len(residual_states_raw)
    block_size = int(residual_states_raw[0].shape[1])

    valid_groups: List[Tuple[str, int, int]] = []
    for name, start, end in EXP5_POSITION_GROUPS:
        if start >= block_size:
            continue
        valid_groups.append((name, start, min(end, block_size)))

    mean_cosine_by_group = np.zeros((n_layers, len(valid_groups)), dtype=np.float32)
    mean_cosine_by_position = np.zeros((n_layers, block_size), dtype=np.float32)

    for layer_idx, state in enumerate(residual_states_raw):
        state_cpu = state.detach().cpu().float()

        for group_idx, (_, start, end) in enumerate(valid_groups):
            group = state_cpu[:, start:end, :].reshape(-1, state_cpu.shape[-1])
            group_centered = group - group.mean(dim=0, keepdim=True)
            mean_cosine_by_group[layer_idx, group_idx] = float(
                mean_cosine_sim(group_centered, num_pairs=EXP5_COSINE_PAIRS, seed=CKA_SEED)
            )

        for position in range(block_size):
            vectors = state_cpu[:, position, :]
            vectors_centered = vectors - vectors.mean(dim=0, keepdim=True)
            mean_cosine_by_position[layer_idx, position] = float(
                mean_cosine_sim(vectors_centered, num_pairs=EXP5_COSINE_PAIRS, seed=CKA_SEED)
            )

    labels = np.asarray([name for name, _, _ in valid_groups], dtype="<U32")
    return {
        "mean_cosine_by_group": mean_cosine_by_group,
        "mean_cosine_by_position": mean_cosine_by_position,
        "position_group_labels": labels,
    }


def aggregate_metric_dicts(metric_dicts: List[dict]) -> Dict[str, np.ndarray]:
    """Returns stacked numpy arrays for a list of metric dictionaries."""
    aggregated: Dict[str, List[np.ndarray]] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                array = value.astype(np.float32, copy=False)
            else:
                array = np.asarray(value, dtype=np.float32)
            aggregated.setdefault(key, []).append(array)

    output: Dict[str, np.ndarray] = {}
    for key, values in aggregated.items():
        output[key] = np.stack(values, axis=0)
    return output


def compute_cka_matrix(residual_states: List[torch.Tensor]) -> np.ndarray:
    """Returns a symmetric CKA matrix over the provided centered residual states."""
    n_states = len(residual_states)
    matrix = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n_states):
        for j in range(i, n_states):
            score = linear_CKA(
                residual_states[i],
                residual_states[j],
                subsample=CKA_SUBSAMPLE,
                seed=CKA_SEED,
            )
            matrix[i, j] = float(score)
            matrix[j, i] = float(score)
    return matrix


def compute_subspace_novelty(reference: torch.Tensor, update: torch.Tensor) -> Tuple[float, float]:
    """Returns effective-rank novelty and variance-fraction novelty of an update relative to a reference subspace."""
    _, singular_values, vh = torch.linalg.svd(reference, full_matrices=False)
    power = singular_values.square()
    total_power = power.sum()

    if float(total_power.item()) <= 0.0:
        projector = torch.eye(reference.shape[1], dtype=reference.dtype)
    else:
        cumulative = torch.cumsum(power, dim=0) / total_power
        k = int(torch.searchsorted(cumulative, torch.tensor(0.90, dtype=cumulative.dtype)).item()) + 1
        vk = vh[:k].transpose(0, 1)
        projector = torch.eye(reference.shape[1], dtype=reference.dtype) - (vk @ vk.transpose(0, 1))

    update_perp = update @ projector
    novelty = float(erank(update_perp))

    update_norm_sq = float(torch.sum(update * update).item())
    perp_norm_sq = float(torch.sum(update_perp * update_perp).item())
    novelty_frac = 0.0 if update_norm_sq <= 0.0 else perp_norm_sq / update_norm_sq
    return novelty, novelty_frac


def make_output_path(out_dir: str, exp_name: str, seed: int, step: int) -> str:
    """Returns the metrics output path for a measured checkpoint."""
    filename = f"{exp_name}_seed{seed}_step{step}_metrics.npz"
    return os.path.join(out_dir, filename)


def measure(args: argparse.Namespace) -> None:
    """Returns nothing and runs hidden-state extraction and metric computation for one checkpoint."""
    ensure_out_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint)
    model = make_model_from_checkpoint(checkpoint, device)
    model.eval()
    disable_all_dropout(model)

    vocab_path = get_vocab_path(args.checkpoint, args.exp_name, checkpoint)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Missing vocab file saved alongside checkpoint: {vocab_path}")
    char_to_id = load_vocab_json(vocab_path)

    checkpoint_config = checkpoint["config"]
    block_size = int(checkpoint_config.get("block_size", args.block_size))
    _, val_chunks, _ = load_wikitext103(args.data_dir, block_size, vocab=char_to_id)

    n_sequences = min(args.n_sequences, len(val_chunks))
    x = get_sequential_chunks(val_chunks, start=0, n=n_sequences, device=str(device))

    with torch.no_grad():
        logits, hidden_states = model(x, extract_hidden_states=True)
    del logits

    centered_hidden_states = [flatten_and_center(state) for state in hidden_states]

    n_layer = int(checkpoint_config["n_layer"])
    residual_indices = [0] + [5 * layer_idx for layer_idx in range(1, n_layer + 1)]
    residual_states = [centered_hidden_states[index] for index in residual_indices]
    residual_states_raw = [hidden_states[index] for index in residual_indices]
    residual_metric_dicts = [compute_all_metrics(state) for state in residual_states]
    residual_metrics = aggregate_metric_dicts(residual_metric_dicts)
    residual_metrics["cka_matrix"] = compute_cka_matrix(residual_states)

    results: Dict[str, object] = dict(residual_metrics)
    results["activation_norm"] = compute_activation_norms(residual_states_raw)
    results["val_loss"] = np.asarray(checkpoint["val_loss"], dtype=np.float32)
    results["step"] = np.asarray(checkpoint["step"], dtype=np.int64)
    results["config"] = np.asarray(json.dumps(checkpoint_config, sort_keys=True), dtype="<U10000")

    if args.exp_name.startswith("exp4"):
        input_states = [centered_hidden_states[1 + 5 * layer_idx + 0] for layer_idx in range(n_layer)]
        attn_states = [centered_hidden_states[1 + 5 * layer_idx + 1] for layer_idx in range(n_layer)]
        post_attn_states = [centered_hidden_states[1 + 5 * layer_idx + 2] for layer_idx in range(n_layer)]
        mlp_states = [centered_hidden_states[1 + 5 * layer_idx + 3] for layer_idx in range(n_layer)]
        output_states = [centered_hidden_states[1 + 5 * layer_idx + 4] for layer_idx in range(n_layer)]

        attn_metric_dicts = [compute_all_metrics(state) for state in attn_states]
        post_attn_metric_dicts = [compute_all_metrics(state) for state in post_attn_states]
        mlp_metric_dicts = [compute_all_metrics(state) for state in mlp_states]
        input_metric_dicts = [compute_all_metrics(state) for state in input_states]
        output_metric_dicts = [compute_all_metrics(state) for state in output_states]

        for key, value in aggregate_metric_dicts(input_metric_dicts).items():
            results[f"{key}_input"] = value
        for key, value in aggregate_metric_dicts(attn_metric_dicts).items():
            results[f"{key}_attn_out"] = value
        for key, value in aggregate_metric_dicts(post_attn_metric_dicts).items():
            results[f"{key}_post_attn"] = value
        for key, value in aggregate_metric_dicts(mlp_metric_dicts).items():
            results[f"{key}_mlp_out"] = value
        for key, value in aggregate_metric_dicts(output_metric_dicts).items():
            results[f"{key}_output"] = value

        results["metrics_attn_out"] = np.asarray(attn_metric_dicts, dtype=object)
        results["metrics_post_attn"] = np.asarray(post_attn_metric_dicts, dtype=object)
        results["metrics_mlp_out"] = np.asarray(mlp_metric_dicts, dtype=object)

        novelty_attn = []
        novelty_mlp = []
        novelty_frac_attn = []
        novelty_frac_mlp = []

        for input_state, attn_state, post_attn_state, mlp_state in zip(
            input_states,
            attn_states,
            post_attn_states,
            mlp_states,
        ):
            novelty_a, novelty_frac_a = compute_subspace_novelty(input_state, attn_state)
            novelty_m, novelty_frac_m = compute_subspace_novelty(post_attn_state, mlp_state)
            novelty_attn.append(novelty_a)
            novelty_mlp.append(novelty_m)
            novelty_frac_attn.append(novelty_frac_a)
            novelty_frac_mlp.append(novelty_frac_m)

        results["novelty_attn"] = np.asarray(novelty_attn, dtype=np.float32)
        results["novelty_mlp"] = np.asarray(novelty_mlp, dtype=np.float32)
        results["novelty_frac_attn"] = np.asarray(novelty_frac_attn, dtype=np.float32)
        results["novelty_frac_mlp"] = np.asarray(novelty_frac_mlp, dtype=np.float32)

    if args.exp_name.startswith("exp5"):
        results.update(compute_exp5_position_metrics(residual_states_raw))

    output_path = make_output_path(args.out_dir, args.exp_name, args.seed, int(checkpoint["step"]))
    np.savez(output_path, **results)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    measure(parse_args())
