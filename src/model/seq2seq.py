"""
Seq2Seq model: composes Encoder + Decoder.

Handles the full forward pass with teacher forcing, and provides
a factory function to build the model from config.
"""

from __future__ import annotations

import gc

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
        projected_encoder_outputs = self.decoder.attention.project_encoder(
            encoder_outputs,
        )

        # Prepare output tensor
        # We predict tgt_len - 1 tokens (skip BOS)
        outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size, device=src.device)

        # First decoder input is BOS token
        current_token = tgt[:, 0]  # (batch,)

        for t in range(tgt_len - 1):
            logits, hidden, cell, _ = self.decoder.forward_step(
                current_token,
                hidden,
                cell,
                encoder_outputs,
                src_mask,
                projected_encoder_outputs=projected_encoder_outputs,
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
    encoder_embedding = None
    decoder_embedding = None
    if config.tie_embeddings and config.embedding_type == "random":
        shared_embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=pad_id,
        )
    elif config.embedding_type == "bert":
        print("Building BERT-initialized embedding tables ...")
        encoder_embedding = _build_bert_initialized_embedding(
            tokenizer=tokenizer,
            model_name=config.bert_hi_model,
            embedding_dim=config.embedding_dim,
            padding_idx=pad_id,
            freeze=config.freeze_bert_embeddings,
            seed=17,
        )
        decoder_embedding = _build_bert_initialized_embedding(
            tokenizer=tokenizer,
            model_name=config.bert_mr_model,
            embedding_dim=config.embedding_dim,
            padding_idx=pad_id,
            freeze=config.freeze_bert_embeddings,
            seed=23,
        )
    elif config.embedding_type != "random":
        raise ValueError(f"Unknown embedding_type: {config.embedding_type}")

    encoder = LSTMEncoder(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional_encoder,
        padding_idx=pad_id,
        embedding=encoder_embedding or shared_embedding,
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
        embedding=decoder_embedding or shared_embedding,
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


def _build_bert_initialized_embedding(
    tokenizer: SharedTokenizer,
    model_name: str,
    embedding_dim: int,
    padding_idx: int,
    freeze: bool,
    seed: int,
) -> nn.Embedding:
    """Create an embedding table by averaging BERT vectors for each BPE token.

    BERT is used once at model construction time. Training still uses a normal
    nn.Embedding layer, which keeps the LSTM experiment fast on Colab/T4.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "BERT embeddings require transformers. Install requirements.txt first."
        ) from exc

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModel.from_pretrained(model_name)
    hf_model.eval()

    source_weight = hf_model.get_input_embeddings().weight.detach().cpu()
    bert_dim = source_weight.size(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    init_std = float(source_weight.std().item())
    table = torch.empty(
        tokenizer.vocab_size,
        embedding_dim,
        dtype=torch.float32,
    )
    table.normal_(mean=0.0, std=init_std, generator=generator)
    table[padding_idx].zero_()

    special_ids = {
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.unk_id,
    }

    with torch.no_grad():
        for idx in range(tokenizer.vocab_size):
            if idx in special_ids:
                continue

            text = _sentencepiece_piece_to_text(tokenizer.id_to_piece(idx))
            if not text:
                continue

            bert_token_ids = hf_tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if not bert_token_ids:
                bert_token_ids = [hf_tokenizer.unk_token_id]
            bert_token_ids = [
                token_id for token_id in bert_token_ids
                if token_id is not None and token_id >= 0
            ]
            if not bert_token_ids:
                continue

            vector = source_weight[bert_token_ids].mean(dim=0)
            table[idx] = _fit_embedding_dim(vector, embedding_dim, generator, init_std)

    embedding = nn.Embedding.from_pretrained(
        table,
        freeze=freeze,
        padding_idx=padding_idx,
    )
    print(
        f"  Initialized {tokenizer.vocab_size:,} embeddings from {model_name} "
        f"(bert_dim={bert_dim}, trainable={not freeze})"
    )

    del hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embedding


def _sentencepiece_piece_to_text(piece: str) -> str:
    """Convert a SentencePiece token into text suitable for a BERT tokenizer."""
    if piece.startswith("<") and piece.endswith(">"):
        return ""
    return piece.replace("▁", " ").strip()


def _fit_embedding_dim(
    vector: torch.Tensor,
    embedding_dim: int,
    generator: torch.Generator,
    init_std: float,
) -> torch.Tensor:
    """Resize a pretrained vector when config.embedding_dim differs."""
    current_dim = vector.numel()
    if current_dim == embedding_dim:
        return vector.to(torch.float32)
    if current_dim > embedding_dim:
        return vector[:embedding_dim].to(torch.float32)

    fitted = torch.empty(embedding_dim, dtype=torch.float32)
    fitted[:current_dim] = vector.to(torch.float32)
    fitted[current_dim:].normal_(mean=0.0, std=init_std, generator=generator)
    return fitted
