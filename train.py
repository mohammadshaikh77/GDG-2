"""Unified training script for all transformer redundancy experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Optional

import numpy as np
import torch

from src.data import get_batch, get_sequential_chunks, load_wikitext103
from src.model import GPT, GPTConfig


CHECKPOINT_STEPS = [500, 1000, 2000, 10000, 50000, 200000]
LOG_INTERVAL = 100
PRINT_INTERVAL = 500
VAL_CHUNKS = 64


def parse_args() -> argparse.Namespace:
    """Returns parsed command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a GPT model for redundancy experiments.")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual_alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=200000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Returns nothing and seeds torch, numpy, and python random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Returns the training device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> GPT:
    """Returns a GPT model moved onto the requested device."""
    config = GPTConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        block_size=args.block_size,
        vocab_size=vocab_size,
        dropout=args.dropout,
        residual_alpha=args.residual_alpha,
    )
    model = GPT(config)
    return model.to(device)


def get_lr(step: int, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Returns the learning rate for a given training step using linear warmup and cosine decay."""
    if step <= warmup_steps:
        return base_lr * float(step) / float(max(warmup_steps, 1))
    if step >= total_steps:
        return min_lr

    decay_progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + cosine_coeff * (base_lr - min_lr)


def build_targets_for_chunk_indices(chunks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Returns next-token targets for the provided chunk indices."""
    x_np = chunks[indices]
    y_np = np.roll(x_np, shift=-1, axis=1).copy()
    next_indices = (indices + 1) % len(chunks)
    y_np[:, -1] = chunks[next_indices, 0]
    return y_np


def evaluate_val_loss(model: GPT, val_chunks: np.ndarray, block_size: int, device: torch.device) -> float:
    """Returns validation loss on the first 64 sequential validation chunks."""
    del block_size
    model.eval()
    with torch.no_grad():
        x = get_sequential_chunks(val_chunks, start=0, n=VAL_CHUNKS, device=str(device))
        indices = np.arange(min(VAL_CHUNKS, len(val_chunks)), dtype=np.int64)
        y_np = build_targets_for_chunk_indices(val_chunks, indices)
        y = torch.as_tensor(y_np, dtype=torch.long, device=device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    model.train()
    return float(loss.item())


def ensure_out_dir(out_dir: str) -> None:
    """Returns nothing and creates the output directory if needed."""
    os.makedirs(out_dir, exist_ok=True)


def make_checkpoint_path(out_dir: str, exp_name: str, seed: int, step: int) -> str:
    """Returns the checkpoint path for a given experiment, seed, and step."""
    filename = f"{exp_name}_seed{seed}_step{step}.pt"
    return os.path.join(out_dir, filename)


def make_log_path(out_dir: str, exp_name: str, seed: int) -> str:
    """Returns the CSV log path for a given experiment and seed."""
    filename = f"{exp_name}_seed{seed}_log.csv"
    return os.path.join(out_dir, filename)


def make_vocab_path(out_dir: str, exp_name: str) -> str:
    """Returns the vocabulary JSON path for a given experiment."""
    filename = f"{exp_name}_vocab.json"
    return os.path.join(out_dir, filename)


def init_csv_log(log_path: str) -> None:
    """Returns nothing and initializes the CSV log file with a header."""
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "train_loss", "val_loss", "grad_norm", "lr"])


def save_vocab_json(vocab_path: str, char_to_id: dict) -> None:
    """Returns nothing and saves the training vocabulary as JSON."""
    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(char_to_id, handle, ensure_ascii=False, sort_keys=True, indent=2)


def append_csv_log(log_path: str, step: int, train_loss: float, val_loss: float, grad_norm: float, lr: float) -> None:
    """Returns nothing and appends one metrics row to the CSV log."""
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([step, train_loss, val_loss, grad_norm, lr])


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    """Returns nothing and saves a training checkpoint."""
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "config": vars(args),
    }
    checkpoint_path = make_checkpoint_path(args.out_dir, args.exp_name, args.seed, step)
    torch.save(checkpoint, checkpoint_path)


def train(args: argparse.Namespace) -> None:
    """Returns nothing and runs model training, logging, validation, and checkpointing."""
    seed_everything(args.seed)
    ensure_out_dir(args.out_dir)

    train_chunks, val_chunks, char_to_id = load_wikitext103(args.data_dir, args.block_size)
    args.vocab_size = len(char_to_id)
    save_vocab_json(make_vocab_path(args.out_dir, args.exp_name), char_to_id)

    device = get_device()
    model = make_model(args, vocab_size=len(char_to_id), device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    train_rng = np.random.default_rng(args.seed)
    log_path = make_log_path(args.out_dir, args.exp_name, args.seed)
    init_csv_log(log_path)

    last_val_loss: Optional[float] = None
    last_grad_norm: Optional[float] = None

    for step in range(1, args.total_steps + 1):
        lr = get_lr(step, args.lr, args.min_lr, args.warmup_steps, args.total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_chunks, batch_size=args.batch_size, device=str(device), rng=train_rng)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        train_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        train_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        last_grad_norm = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

        if step % LOG_INTERVAL == 0:
            last_val_loss = evaluate_val_loss(model, val_chunks, args.block_size, device)
            append_csv_log(
                log_path=log_path,
                step=step,
                train_loss=float(train_loss.item()),
                val_loss=last_val_loss,
                grad_norm=last_grad_norm,
                lr=lr,
            )

        if step % PRINT_INTERVAL == 0:
            val_to_print = float("nan") if last_val_loss is None else last_val_loss
            grad_to_print = float("nan") if last_grad_norm is None else last_grad_norm
            print(
                f"step={step} "
                f"train_loss={train_loss.item():.6f} "
                f"val_loss={val_to_print:.6f} "
                f"grad_norm={grad_to_print:.6f}"
            )

        if step in CHECKPOINT_STEPS:
            if last_val_loss is None:
                last_val_loss = evaluate_val_loss(model, val_chunks, args.block_size, device)
            save_checkpoint(model, optimizer, step, last_val_loss, args)


if __name__ == "__main__":
    train(parse_args())
