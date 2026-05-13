"""
LSTM Decoder with Attention for Seq2Seq NMT.

At each decoding step:
  1. Embed the previous token
  2. Compute attention over encoder outputs
  3. Concatenate [embedding, context] → feed through LSTM
  4. Project to vocabulary logits
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import BahdanauAttention


class LSTMDecoder(nn.Module):
    """LSTM decoder with Bahdanau attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        encoder_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 256,
        padding_idx: int = 0,
        embedding: nn.Embedding | None = None,
    ):
        """
        Args:
            vocab_size: Size of the target vocabulary.
            embedding_dim: Dimensionality of token embeddings.
            hidden_dim: LSTM hidden size.
            encoder_dim: Dimensionality of encoder output vectors.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate.
            attention_dim: Internal attention dimensionality.
            padding_idx: Padding token index.
            embedding: Optional pre-built embedding layer.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx,
            )

        # Attention
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)

        # LSTM: input is [embedding; context_vector]
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection: [hidden; context; embedding] → vocab
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embedding_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single decoding step.

        Args:
            token: (batch,) — current input token IDs.
            hidden: (num_layers, batch, hidden_dim) — LSTM hidden state.
            cell: (num_layers, batch, hidden_dim) — LSTM cell state.
            encoder_outputs: (batch, src_len, encoder_dim) — encoder outputs.
            src_mask: (batch, src_len) — True for PAD positions.

        Returns:
            logits: (batch, vocab_size) — output logits.
            hidden: updated hidden state.
            cell: updated cell state.
            attn_weights: (batch, src_len) — attention distribution.
        """
        # (batch,) → (batch, embedding_dim)
        embedded = self.dropout(self.embedding(token))

        # Attention: use top-layer hidden state as query
        # hidden[-1]: (batch, hidden_dim)
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, src_mask)

        # LSTM input: [embedding; context] → (batch, 1, embedding_dim + encoder_dim)
        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_output = lstm_output.squeeze(1)  # (batch, hidden_dim)

        # Output projection: [lstm_output; context; embedded]
        combined = torch.cat([lstm_output, context, embedded], dim=-1)
        logits = self.fc_out(combined)

        return logits, hidden, cell, attn_weights
