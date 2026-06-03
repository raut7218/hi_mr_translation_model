"""
LSTM Encoder for Seq2Seq NMT.

Supports bidirectional encoding and projects hidden states
to be compatible with the decoder's expected dimensions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Bidirectional (or unidirectional) LSTM encoder.

    Embeds source tokens, runs through multi-layer LSTM, and projects
    the final hidden/cell states for decoder initialization.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = 0,
        embedding: nn.Embedding | None = None,
    ):
        """
        Args:
            vocab_size: Size of the source vocabulary.
            embedding_dim: Dimensionality of token embeddings.
            hidden_dim: LSTM hidden size (per direction).
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between LSTM layers.
            bidirectional: Whether to use a bidirectional LSTM.
            padding_idx: Index of the padding token.
            embedding: Optional pre-built embedding layer (e.g., from BERT).
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx,
            )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # Project bidirectional hidden states → decoder-compatible size
        # (hidden_dim * 2) → hidden_dim for both hidden and cell
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
            self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: (batch, src_len) — source token IDs.
            src_lengths: (batch,) — actual lengths for packing (optional).

        Returns:
            outputs: (batch, src_len, hidden_dim * num_directions) — encoder outputs.
            (hidden, cell): each (num_layers, batch, hidden_dim) — for decoder init.
        """
        # (batch, src_len) → (batch, src_len, embedding_dim)
        embedded = self.dropout(self.embedding(src))

        # Pack padded sequences for efficiency
        if src_lengths is not None:
            # Ensure lengths are on CPU for pack_padded_sequence
            src_lengths_cpu = src_lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=src.size(1),
            )
        else:
            outputs, (hidden, cell) = self.lstm(embedded)

        # Project bidirectional states for the decoder
        if self.bidirectional:
            # hidden: (num_layers * 2, batch, hidden_dim)
            # Reshape to (num_layers, batch, hidden_dim * 2)
            hidden = self._merge_directions(hidden)
            cell = self._merge_directions(cell)
            # Project to decoder size
            hidden = torch.tanh(self.fc_hidden(hidden))
            cell = torch.tanh(self.fc_cell(cell))

        return outputs, (hidden, cell)

    def _merge_directions(self, state: torch.Tensor) -> torch.Tensor:
        """Merge forward and backward hidden states.

        Input:  (num_layers * 2, batch, hidden_dim)
        Output: (num_layers, batch, hidden_dim * 2)
        """
        num_layers = self.num_layers
        batch_size = state.size(1)
        # Reshape: (num_layers, 2, batch, hidden_dim)
        state = state.view(num_layers, 2, batch_size, self.hidden_dim)
        # Concatenate forward and backward: (num_layers, batch, hidden_dim * 2)
        state = torch.cat([state[:, 0, :, :], state[:, 1, :, :]], dim=-1)
        return state
