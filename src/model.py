"""Pre-LN GPT-style transformer model used across experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Returns a dataclass holding GPT hyperparameters."""

    n_layer: int
    n_embd: int
    n_head: int
    block_size: int
    vocab_size: int
    dropout: float = 0.0
    residual_alpha: float = 1.0


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention for a decoder-only transformer."""

    def __init__(self, config: GPTConfig) -> None:
        """Returns an initialized causal self-attention module."""
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd must be divisible by n_head, got {config.n_embd} and {config.n_head}."
            )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns attention output and optionally attention weights of shape (batch, heads, seq_len, seq_len)."""
        batch_size, seq_len, channels = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        if return_attn_weights:
            return y, att
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: GPTConfig) -> None:
        """Returns an initialized MLP module."""
        super().__init__()
        inner_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, inner_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(inner_dim, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the MLP output tensor of shape (batch, seq_len, n_embd)."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Single Pre-LN GPT transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        """Returns an initialized transformer block."""
        super().__init__()
        self.residual_alpha = config.residual_alpha
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        extract_hidden_states: bool = False,
        return_attn_weights: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor],
    ]:
        """Returns the block output, and optionally the five experiment extraction tensors."""
        if not extract_hidden_states:
            if return_attn_weights:
                attn_out, attn_weights = self.attn(self.ln_1(x), return_attn_weights=True)
            else:
                attn_out = self.attn(self.ln_1(x))
            x = x + self.residual_alpha * attn_out
            mlp_out = self.mlp(self.ln_2(x))
            x = x + self.residual_alpha * mlp_out
            if return_attn_weights:
                return x, attn_weights
            return x

        h_l = x
        if return_attn_weights:
            a_l, attn_weights = self.attn(self.ln_1(h_l), return_attn_weights=True)
        else:
            a_l = self.attn(self.ln_1(h_l))
        r_l = h_l + self.residual_alpha * a_l
        m_l = self.mlp(self.ln_2(r_l))
        h_next = r_l + self.residual_alpha * m_l
        if return_attn_weights:
            return h_next, [h_l, a_l, r_l, m_l, h_next], attn_weights
        return h_next, [h_l, a_l, r_l, m_l, h_next]


class GPT(nn.Module):
    """Pre-LN GPT-style decoder-only transformer."""

    def __init__(self, config: GPTConfig) -> None:
        """Returns an initialized GPT model."""
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Returns nothing and initializes module parameters."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        extract_hidden_states: bool = False,
        return_attn_weights: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
        Tuple[torch.Tensor, List[torch.Tensor]],
    ]:
        """Returns logits, and optionally the ordered hidden-state extraction list."""
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block size {self.config.block_size}."
            )

        pos = torch.arange(0, seq_len, device=idx.device, dtype=torch.long)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)[None, :, :]
        x = self.drop(tok_emb + pos_emb)

        if not extract_hidden_states:
            attn_weights_all: List[torch.Tensor] = []
            for block in self.blocks:
                if return_attn_weights:
                    x, attn_weights = block(x, extract_hidden_states=False, return_attn_weights=True)
                    attn_weights_all.append(attn_weights)
                else:
                    x = block(x, extract_hidden_states=False)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            if return_attn_weights:
                return logits, attn_weights_all
            return logits

        hidden_states: List[torch.Tensor] = [x]
        attn_weights_all: List[torch.Tensor] = []
        for block in self.blocks:
            if return_attn_weights:
                x, block_states, attn_weights = block(x, extract_hidden_states=True, return_attn_weights=True)
                attn_weights_all.append(attn_weights)
            else:
                x, block_states = block(x, extract_hidden_states=True)
            hidden_states.extend(block_states)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if return_attn_weights:
            return logits, hidden_states, attn_weights_all
        return logits, hidden_states


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable and non-trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    from dataclasses import dataclass

    cfg = GPTConfig(n_layer=6, n_embd=256, n_head=4, block_size=128, vocab_size=100)
    model = GPT(cfg)
    x = torch.randint(0, 100, (2, 128))
    logits, hs = model(x, extract_hidden_states=True)
    assert logits.shape == (2, 128, 100)
    assert len(hs) == 31
    assert hs[0].shape == (2, 128, 256)
    print(f"Model parameters: {count_parameters(model):,}")
    print("All model sanity checks passed.")
