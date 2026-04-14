"""Character-level WikiText-103 data loading utilities."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from typing import List, Optional, Tuple

import numpy as np
import torch


def build_vocab(train_path: str) -> dict:
    """Returns a character-to-id vocabulary built from the train split, reserving id 0 for UNK."""
    with open(train_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    unique_chars = sorted(set(text))
    char_to_id = {"<UNK>": 0}
    for index, char in enumerate(unique_chars, start=1):
        char_to_id[char] = index

    print(f"Vocab size: {len(char_to_id)}")
    return char_to_id


def encode(text: str, char_to_id: dict) -> List[int]:
    """Returns a list of token ids for the input text, mapping unknown characters to 0."""
    unk_id = 0
    return [char_to_id.get(char, unk_id) for char in text]


def make_chunks(token_ids: List[int], block_size: int) -> np.ndarray:
    """Returns non-overlapping token chunks of shape (n_chunks, block_size) with dtype int32."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    usable_tokens = (len(token_ids) // block_size) * block_size
    if usable_tokens == 0:
        return np.zeros((0, block_size), dtype=np.int32)

    array = np.asarray(token_ids[:usable_tokens], dtype=np.int32)
    return array.reshape(-1, block_size)


def load_wikitext103(
    data_dir: str,
    block_size: int,
    vocab: dict = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Returns train chunks, validation chunks, and the character vocabulary from raw WikiText-103 files."""
    train_path = os.path.join(data_dir, "wiki.train.raw")
    valid_path = os.path.join(data_dir, "wiki.valid.raw")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Missing validation split: {valid_path}")

    char_to_id = build_vocab(train_path) if vocab is None else vocab

    with open(train_path, "r", encoding="utf-8") as handle:
        train_text = handle.read()
    with open(valid_path, "r", encoding="utf-8") as handle:
        valid_text = handle.read()

    train_ids = encode(train_text, char_to_id)
    valid_ids = encode(valid_text, char_to_id)

    train_chunks = make_chunks(train_ids, block_size)
    val_chunks = make_chunks(valid_ids, block_size)
    return train_chunks, val_chunks, char_to_id


def get_batch(
    chunks: np.ndarray,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns random training inputs and next-token targets as torch.long tensors on the requested device."""
    if chunks.ndim != 2:
        raise ValueError(f"chunks must have shape (n_chunks, block_size), got {chunks.shape}.")
    if len(chunks) == 0:
        raise ValueError("chunks must contain at least one chunk.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    indices = rng.integers(0, len(chunks), size=batch_size)
    x_np = chunks[indices]
    y_np = np.roll(x_np, shift=-1, axis=1).copy()

    next_indices = (indices + 1) % len(chunks)
    y_np[:, -1] = chunks[next_indices, 0]

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y


def get_sequential_chunks(
    chunks: np.ndarray,
    start: int,
    n: int,
    device: str,
) -> torch.Tensor:
    """Returns a deterministic slice of chunk rows as a torch.long tensor on the requested device."""
    if chunks.ndim != 2:
        raise ValueError(f"chunks must have shape (n_chunks, block_size), got {chunks.shape}.")
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}.")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")

    sliced = chunks[start : start + n]
    return torch.as_tensor(sliced, dtype=torch.long, device=device)


def _find_data_dir() -> Optional[str]:
    """Returns a directory containing WikiText-103 raw files when one is found locally."""
    candidate_dirs = [
        os.getcwd(),
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.getcwd(), "wikitext-103-raw"),
        os.path.join(os.getcwd(), "wikitext-103-raw-v1"),
    ]

    for directory in candidate_dirs:
        train_path = os.path.join(directory, "wiki.train.raw")
        valid_path = os.path.join(directory, "wiki.valid.raw")
        if os.path.exists(train_path) and os.path.exists(valid_path):
            return directory
    return None


def _download_wikitext103_raw(target_root: str) -> str:
    """Returns the directory containing downloaded WikiText-103 raw files."""
    os.makedirs(target_root, exist_ok=True)
    final_dir = os.path.join(target_root, "wikitext-103-raw")
    os.makedirs(final_dir, exist_ok=True)

    train_path = os.path.join(final_dir, "wiki.train.raw")
    valid_path = os.path.join(final_dir, "wiki.valid.raw")
    if os.path.exists(train_path) and os.path.exists(valid_path):
        return final_dir

    try:
        from datasets import load_dataset

        print("Downloading WikiText-103 raw via Hugging Face datasets")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        with open(train_path, "w", encoding="utf-8") as handle:
            handle.write("".join(dataset["train"]["text"]))
        with open(valid_path, "w", encoding="utf-8") as handle:
            handle.write("".join(dataset["validation"]["text"]))
        return final_dir
    except Exception:
        pass

    file_urls = {
        "wiki.train.raw": "https://raw.githubusercontent.com/salesforce/wikitext/master/wikitext-103-raw/wiki.train.raw",
        "wiki.valid.raw": "https://raw.githubusercontent.com/salesforce/wikitext/master/wikitext-103-raw/wiki.valid.raw",
    }

    last_error = None
    for filename, url in file_urls.items():
        target_path = os.path.join(final_dir, filename)
        if os.path.exists(target_path):
            continue
        try:
            print(f"Downloading {filename} from {url}")
            urllib.request.urlretrieve(url, target_path)
        except Exception as exc:
            last_error = exc
            break

    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        raise RuntimeError("Failed to download WikiText-103 raw files.") from last_error

    return final_dir


if __name__ == "__main__":
    data_dir = _find_data_dir()
    if data_dir is None:
        temp_root = os.path.join(tempfile.gettempdir(), "wikitext103_data")
        data_dir = _download_wikitext103_raw(temp_root)

    train_chunks, val_chunks, char_to_id = load_wikitext103(data_dir=data_dir, block_size=128)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Vocab size: {len(char_to_id)}")

    seq = get_sequential_chunks(val_chunks, start=0, n=4, device="cpu")
    assert seq.shape == (4, 128)
    assert seq.dtype == torch.long

    rng = np.random.default_rng(42)
    x, y = get_batch(train_chunks, batch_size=2, device="cpu", rng=rng)
    assert x.shape == y.shape == (2, 128)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    print("All data sanity checks passed.")
