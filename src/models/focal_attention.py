"""Attention-specific modules for focal conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalCrossAttention(nn.Module):
    """Cross-attention used to inject focal features into the SD3 transformer."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_size)

        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(self, hidden_states: torch.Tensor, encoder_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_states)
        v = self.to_v(encoder_states)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attn = torch.matmul(attention_probs, v)

        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.to_out(attn)


__all__ = ["FocalCrossAttention"]
