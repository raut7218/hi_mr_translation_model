"""
Bahdanau (Additive) Attention mechanism.

Reference: Bahdanau et al., "Neural Machine Translation by Jointly Learning
to Align and Translate", ICLR 2015.

score(s_t, h_j) = V^T * tanh(W_h * h_j + W_s * s_t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention.

    Computes attention weights over encoder outputs given the current
    decoder hidden state, then produces a context vector.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Args:
            encoder_dim: Dimensionality of encoder output vectors.
            decoder_dim: Dimensionality of decoder hidden state.
            attention_dim: Internal dimensionality of the attention layer.
        """
        super().__init__()
        self.W_h = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_s = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        projected_encoder_outputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: (batch, decoder_dim) — current decoder hidden state.
            encoder_outputs: (batch, src_len, encoder_dim) — all encoder outputs.
            src_mask: (batch, src_len) — boolean mask, True for PAD positions.

        Returns:
            context: (batch, encoder_dim) — weighted sum of encoder outputs.
            attn_weights: (batch, src_len) — attention distribution.
        """
        # decoder_hidden: (batch, decoder_dim) → (batch, 1, attention_dim)
        query = self.W_s(decoder_hidden).unsqueeze(1)
        # encoder_outputs: (batch, src_len, encoder_dim) → (batch, src_len, attention_dim)
        keys = (
            projected_encoder_outputs
            if projected_encoder_outputs is not None
            else self.W_h(encoder_outputs)
        )

        # (batch, src_len, attention_dim) → (batch, src_len, 1) → (batch, src_len)
        energy = self.V(torch.tanh(query + keys)).squeeze(-1)

        # Mask padding positions with -inf before softmax
        if src_mask is not None:
            energy = energy.masked_fill(src_mask, float("-inf"))

        attn_weights = F.softmax(energy, dim=-1)

        # (batch, 1, src_len) @ (batch, src_len, encoder_dim) → (batch, 1, encoder_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights

    def project_encoder(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """Precompute encoder-side attention keys once per source batch."""
        return self.W_h(encoder_outputs)
