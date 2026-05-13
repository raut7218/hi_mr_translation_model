"""
Seq2Seq model: composes Encoder + Decoder.

Handles the full forward pass with teacher forcing, and provides
a factory function to build the model from config.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.data.tokenizer import SharedTokenizer
from src.model.decoder import LSTMDecoder
from src.model.encoder import LSTMEncoder


class Seq2Seq(nn.Module):
    """Full sequence-to-sequence model with attention."""

    def __init__(
        self,
        encoder: LSTMEncoder,
        decoder: LSTMDecoder,
        pad_id: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create a boolean mask: True where src == pad_id."""
        return src == self.pad_id

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            src: (batch, src_len) — source token IDs.
            tgt: (batch, tgt_len) — target token IDs (with BOS prefix).
            src_lengths: (batch,) — actual source lengths.
            teacher_forcing_ratio: probability of using ground truth at each step.

        Returns:
            outputs: (batch, tgt_len - 1, vocab_size) — logits for each step.
                     We skip the first BOS token in targets, so output length = tgt_len - 1.
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size

        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        src_mask = self.create_src_mask(src)

        # Prepare output tensor
        # We predict tgt_len - 1 tokens (skip BOS)
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)

        # First decoder input is BOS token
        current_token = tgt[:, 0]  # (batch,)

        for t in range(tgt_len - 1):
            logits, hidden, cell, _ = self.decoder.forward_step(
                current_token, hidden, cell, encoder_outputs, src_mask,
            )
            outputs[:, t, :] = logits

            # Teacher forcing: use ground truth or model prediction
            if torch.rand(1).item() < teacher_forcing_ratio:
                current_token = tgt[:, t + 1]
            else:
                current_token = logits.argmax(dim=-1)

        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    config: ModelConfig,
    tokenizer: SharedTokenizer,
) -> Seq2Seq:
    """Build a Seq2Seq model from config.

    Handles shared embeddings when tie_embeddings is True.
    """
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    encoder_dim = config.hidden_dim * (2 if config.bidirectional_encoder else 1)

    # Build shared or separate embeddings
    shared_embedding = None
    if config.tie_embeddings and config.embedding_type == "random":
        shared_embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=pad_id,
        )

    encoder = LSTMEncoder(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional_encoder,
        padding_idx=pad_id,
        embedding=shared_embedding,
    )

    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        encoder_dim=encoder_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        attention_dim=config.hidden_dim,  # attention dim = hidden dim
        padding_idx=pad_id,
        embedding=shared_embedding,
    )

    model = Seq2Seq(encoder, decoder, pad_id=pad_id)

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel built:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Embedding type:       {config.embedding_type}")
    print(f"  Tied embeddings:      {config.tie_embeddings}")
    print(f"  Encoder dim:          {encoder_dim}")
    print(f"  Hidden dim:           {config.hidden_dim}")
    print(f"  Num layers:           {config.num_layers}")
    print()

    return model
