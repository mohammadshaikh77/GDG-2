"""Shared metric functions for redundancy experiments."""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]
ScalarLike = Union[float, np.floating, torch.Tensor]


def _to_torch(x: ArrayLike) -> Tuple[torch.Tensor, bool]:
    """Returns the input as a float32 torch tensor and whether the original input was a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.float(), False
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    raise TypeError(f"Unsupported input type: {type(x)!r}")


def _to_output(x: torch.Tensor, as_numpy: bool) -> Union[np.ndarray, torch.Tensor]:
    """Returns a tensor as numpy when the input source was numpy, else returns a torch tensor."""
    if as_numpy:
        return x.detach().cpu().numpy()
    return x


def _to_scalar(x: torch.Tensor, as_numpy: bool) -> Union[float, np.floating]:
    """Returns a 0-d tensor as a Python or numpy scalar."""
    value = x.detach().cpu().item()
    if as_numpy:
        return np.float64(value)
    return float(value)


def _validate_matrix(x: torch.Tensor) -> None:
    """Returns nothing and raises if the input is not a rank-2 matrix."""
    if x.ndim != 2:
        raise ValueError(f"Expected a rank-2 matrix, got shape {tuple(x.shape)}.")
    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"Expected shape (N, D) with N >= 2 and D >= 1, got {tuple(x.shape)}.")


def _cov_eigvals(x: torch.Tensor) -> torch.Tensor:
    """Returns covariance eigenvalues sorted in descending order."""
    _validate_matrix(x)
    n = x.shape[0]
    cov = (x.transpose(0, 1) @ x) / float(n)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    return torch.flip(eigvals, dims=[0])


def _normalized_spectrum(eigvals: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Returns eigenvalues normalized into a probability vector."""
    total = eigvals.sum()
    if total <= eps:
        d = eigvals.numel()
        return torch.full_like(eigvals, 1.0 / max(d, 1))
    probs = eigvals / total
    return torch.clamp(probs, min=eps)


def erank(x: ArrayLike, eps: float = 1e-12) -> ScalarLike:
    """Returns the effective rank of a centered matrix."""
    x_t, as_numpy = _to_torch(x)
    eigvals = _cov_eigvals(x_t)
    probs = _normalized_spectrum(eigvals, eps=eps)
    entropy = -(probs * torch.log(probs)).sum()
    return _to_scalar(torch.exp(entropy), as_numpy)


def srank(x: ArrayLike, eps: float = 1e-12) -> ScalarLike:
    """Returns the stable rank of a centered matrix."""
    x_t, as_numpy = _to_torch(x)
    eigvals = _cov_eigvals(x_t)
    denom = torch.clamp(eigvals[0], min=eps)
    return _to_scalar(eigvals.sum() / denom, as_numpy)


def ferank(x: ArrayLike, eps: float = 1e-12) -> ScalarLike:
    """Returns the fractional effective rank, equal to effective rank divided by feature dimension."""
    x_t, as_numpy = _to_torch(x)
    value = torch.as_tensor(erank(x_t, eps=eps), device=x_t.device, dtype=x_t.dtype) / x_t.shape[1]
    return _to_scalar(value, as_numpy)


def delta(x: ArrayLike, eps: float = 1e-12) -> ScalarLike:
    """Returns the fraction of variance explained by the top eigenvalue."""
    x_t, as_numpy = _to_torch(x)
    eigvals = _cov_eigvals(x_t)
    denom = torch.clamp(eigvals.sum(), min=eps)
    return _to_scalar(eigvals[0] / denom, as_numpy)


def top_k_dominance(
    x: ArrayLike,
    k: Union[int, Sequence[int]],
    eps: float = 1e-12,
) -> Union[ScalarLike, np.ndarray, torch.Tensor]:
    """Returns cumulative eigenvalue dominance for one or more top-k values."""
    x_t, as_numpy = _to_torch(x)
    eigvals = _cov_eigvals(x_t)
    denom = torch.clamp(eigvals.sum(), min=eps)
    cum = torch.cumsum(eigvals, dim=0) / denom

    if isinstance(k, int):
        if k < 1 or k > eigvals.numel():
            raise ValueError(f"k must be in [1, {eigvals.numel()}], got {k}.")
        return _to_scalar(cum[k - 1], as_numpy)

    k_list = list(k)
    if len(k_list) == 0:
        raise ValueError("k sequence must be non-empty.")
    indices = []
    for value in k_list:
        if value < 1 or value > eigvals.numel():
            raise ValueError(f"k must be in [1, {eigvals.numel()}], got {value}.")
        indices.append(value - 1)
    result = cum[torch.tensor(indices, device=cum.device, dtype=torch.long)]
    return _to_output(result, as_numpy)


def _sample_pair_indices(num_rows: int, num_pairs: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns two index tensors for distinct row pairs sampled uniformly with replacement."""
    if num_rows < 2:
        raise ValueError("Need at least two rows to sample cosine similarity pairs.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    i = torch.randint(0, num_rows, (num_pairs,), generator=generator)
    offsets = torch.randint(1, num_rows, (num_pairs,), generator=generator)
    j = (i + offsets) % num_rows
    return i, j


def _pairwise_cosines(x: torch.Tensor, num_pairs: int, seed: int, eps: float = 1e-12) -> torch.Tensor:
    """Returns sampled pairwise cosine similarities between matrix rows."""
    _validate_matrix(x)
    i, j = _sample_pair_indices(x.shape[0], num_pairs, seed)
    xi = x[i.to(x.device)]
    xj = x[j.to(x.device)]
    xi = xi / torch.clamp(torch.linalg.norm(xi, dim=-1, keepdim=True), min=eps)
    xj = xj / torch.clamp(torch.linalg.norm(xj, dim=-1, keepdim=True), min=eps)
    return (xi * xj).sum(dim=-1)


def mean_cosine_sim(x: ArrayLike, num_pairs: int = 2000, seed: int = 42, eps: float = 1e-12) -> ScalarLike:
    """Returns the mean sampled pairwise cosine similarity between matrix rows."""
    x_t, as_numpy = _to_torch(x)
    cosines = _pairwise_cosines(x_t, num_pairs=num_pairs, seed=seed, eps=eps)
    return _to_scalar(cosines.mean(), as_numpy)


def var_cosine_sim(
    x: ArrayLike,
    num_pairs: int = 2000,
    seed: int = 42,
    eps: float = 1e-12,
) -> ScalarLike:
    """Returns the variance of sampled pairwise cosine similarity between matrix rows."""
    x_t, as_numpy = _to_torch(x)
    cosines = _pairwise_cosines(x_t, num_pairs=num_pairs, seed=seed, eps=eps)
    return _to_scalar(cosines.var(unbiased=False), as_numpy)


def linear_CKA(
    x: ArrayLike,
    y: ArrayLike,
    subsample: Optional[int] = None,
    seed: int = 42,
    eps: float = 1e-12,
) -> ScalarLike:
    """Returns the linear CKA similarity between two centered matrices."""
    x_t, as_numpy_x = _to_torch(x)
    y_t, as_numpy_y = _to_torch(y)
    if as_numpy_x != as_numpy_y:
        raise TypeError("x and y must both be numpy arrays or both be torch tensors.")
    _validate_matrix(x_t)
    _validate_matrix(y_t)
    if x_t.shape != y_t.shape:
        raise ValueError(f"x and y must have identical shapes, got {tuple(x_t.shape)} and {tuple(y_t.shape)}.")

    if subsample is not None:
        if subsample < 1:
            raise ValueError(f"subsample must be positive, got {subsample}.")
        subsample = min(subsample, x_t.shape[0])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(x_t.shape[0], generator=generator)[:subsample]
        indices = indices.to(x_t.device)
        x_t = x_t[indices]
        y_t = y_t[indices]

    cross = y_t.transpose(0, 1) @ x_t
    gram_x = x_t.transpose(0, 1) @ x_t
    gram_y = y_t.transpose(0, 1) @ y_t
    numerator = torch.sum(cross * cross)
    denominator = torch.clamp(torch.linalg.norm(gram_x) * torch.linalg.norm(gram_y), min=eps)
    return _to_scalar(numerator / denominator, as_numpy_x)


def redundancy_index(
    x: ArrayLike,
    num_pairs: int = 2000,
    seed: int = 42,
    eps: float = 1e-12,
) -> ScalarLike:
    """Returns the redundancy index defined as delta plus mean cosine similarity."""
    x_t, as_numpy = _to_torch(x)
    value = torch.as_tensor(delta(x_t, eps=eps), device=x_t.device, dtype=x_t.dtype)
    value = value + torch.as_tensor(
        mean_cosine_sim(x_t, num_pairs=num_pairs, seed=seed, eps=eps),
        device=x_t.device,
        dtype=x_t.dtype,
    )
    return _to_scalar(value, as_numpy)


def compute_all_metrics(
    x: ArrayLike,
    num_pairs: int = 2000,
    cosine_seed: int = 42,
    eps: float = 1e-12,
) -> dict:
    """Returns a dictionary of all shared scalar metrics plus the covariance eigenvalue spectrum."""
    x_t, as_numpy = _to_torch(x)
    eigvals = _cov_eigvals(x_t)
    eigvals_out = _to_output(eigvals, as_numpy)

    metrics = {
        "erank": erank(x_t, eps=eps),
        "srank": srank(x_t, eps=eps),
        "ferank": ferank(x_t, eps=eps),
        "delta": delta(x_t, eps=eps),
        "top_5_dominance": top_k_dominance(x_t, 5 if x_t.shape[1] >= 5 else x_t.shape[1], eps=eps),
        "top_10_dominance": top_k_dominance(x_t, 10 if x_t.shape[1] >= 10 else x_t.shape[1], eps=eps),
        "mean_cosine": mean_cosine_sim(x_t, num_pairs=num_pairs, seed=cosine_seed, eps=eps),
        "var_cosine": var_cosine_sim(x_t, num_pairs=num_pairs, seed=cosine_seed, eps=eps),
        "redundancy_index": redundancy_index(x_t, num_pairs=num_pairs, seed=cosine_seed, eps=eps),
        "eigenvalues": eigvals_out,
    }
    return metrics


def _seed_all(seed: int) -> None:
    """Returns nothing and seeds python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    _seed_all(123)

    x = torch.randn(512, 256)
    y = torch.randn(512, 256)
    x_np = x.numpy()

    er = erank(x)
    sr = srank(x)
    fr = ferank(x)
    de = delta(x)
    dk = top_k_dominance(x, [1, 5, 10])
    mc = mean_cosine_sim(x, num_pairs=128, seed=42)
    vc = var_cosine_sim(x, num_pairs=128, seed=42)
    cka = linear_CKA(x, y, subsample=256, seed=42)
    red = redundancy_index(x, num_pairs=128, seed=42)
    all_metrics = compute_all_metrics(x, num_pairs=128, cosine_seed=42)

    assert isinstance(er, float)
    assert isinstance(sr, float)
    assert isinstance(fr, float)
    assert isinstance(de, float)
    assert isinstance(mc, float)
    assert isinstance(vc, float)
    assert isinstance(cka, float)
    assert isinstance(red, float)
    assert tuple(dk.shape) == (3,)

    assert 1.0 <= er <= 256.0
    assert 1.0 <= sr <= 256.0
    assert (1.0 / 256.0) <= fr <= 1.0
    assert 0.0 <= de <= 1.0
    assert torch.all(dk[:-1] <= dk[1:])
    assert torch.all((0.0 <= dk) & (dk <= 1.0))
    assert -1.0 <= mc <= 1.0
    assert 0.0 <= vc <= 1.0
    assert 0.0 <= cka <= 1.0 + 1e-5
    assert 0.0 <= red <= 2.0
    assert all_metrics["eigenvalues"].shape == (256,)

    er_np = erank(x_np)
    dk_np = top_k_dominance(x_np, [1, 5, 10])
    assert np.isscalar(er_np)
    assert dk_np.shape == (3,)
    assert np.all((0.0 <= dk_np) & (dk_np <= 1.0))

    print("metrics.py sanity check passed.")
