"""
Inference: greedy and beam search decoding for the Seq2Seq model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.tokenizer import SharedTokenizer
from src.model.seq2seq import Seq2Seq


@torch.no_grad()
def greedy_decode(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    tokenizer: SharedTokenizer,
    max_len: int = 150,
) -> list[list[int]]:
    """Greedy decoding for a batch of source sequences.

    Args:
        model: Trained Seq2Seq model.
        src: (batch, src_len) — source token IDs.
        src_lengths: (batch,) — source lengths.
        tokenizer: Tokenizer for special token IDs.
        max_len: Maximum decoding length.

    Returns:
        List of decoded token ID sequences (one per batch item).
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device

    # Encode
    encoder_outputs, (hidden, cell) = model.encoder(src, src_lengths)
    src_mask = model.create_src_mask(src)

    # Start with BOS
    current_token = torch.full(
        (batch_size,), tokenizer.bos_id, dtype=torch.long, device=device,
    )

    # Track decoded tokens and finished status
    decoded = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    for _ in range(max_len):
        logits, hidden, cell, _ = model.decoder.forward_step(
            current_token, hidden, cell, encoder_outputs, src_mask,
        )
        # Greedy: pick the most likely token
        current_token = logits.argmax(dim=-1)  # (batch,)

        for i in range(batch_size):
            if not finished[i]:
                tok = current_token[i].item()
                if tok == tokenizer.eos_id:
                    finished[i] = True
                else:
                    decoded[i].append(tok)

        if all(finished):
            break

    return decoded


@torch.no_grad()
def beam_search_decode(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    tokenizer: SharedTokenizer,
    beam_size: int = 5,
    max_len: int = 150,
    length_penalty: float = 0.6,
) -> list[list[int]]:
    """Beam search decoding (processes one sentence at a time).

    For batch inference, call this in a loop over individual sentences.

    Args:
        src: (1, src_len) — single source sentence.
        src_lengths: (1,) — source length.
        tokenizer: Tokenizer for special token IDs.
        beam_size: Number of beams.
        max_len: Maximum decoding length.
        length_penalty: Length normalization factor.

    Returns:
        List containing one decoded token ID sequence (best beam).
    """
    model.eval()
    device = src.device

    # Encode
    encoder_outputs, (hidden, cell) = model.encoder(src, src_lengths)
    src_mask = model.create_src_mask(src)

    # Expand for beam search: (1, ...) → (beam_size, ...)
    encoder_outputs = encoder_outputs.repeat(beam_size, 1, 1)
    src_mask = src_mask.repeat(beam_size, 1)
    hidden = hidden.repeat(1, beam_size, 1)
    cell = cell.repeat(1, beam_size, 1)

    # Initialize beams
    current_token = torch.full(
        (beam_size,), tokenizer.bos_id, dtype=torch.long, device=device,
    )
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = float("-inf")  # Only first beam is active initially

    # Store completed beams
    completed_beams: list[tuple[float, list[int]]] = []
    beam_sequences: list[list[int]] = [[] for _ in range(beam_size)]

    for step in range(max_len):
        logits, hidden, cell, _ = model.decoder.forward_step(
            current_token, hidden, cell, encoder_outputs, src_mask,
        )
        log_probs = F.log_softmax(logits, dim=-1)  # (beam_size, vocab_size)

        # Compute scores for all possible next tokens
        vocab_size = log_probs.size(-1)
        next_scores = beam_scores.unsqueeze(-1) + log_probs  # (beam_size, vocab_size)
        next_scores = next_scores.view(-1)  # (beam_size * vocab_size,)

        # Pick top-k
        top_scores, top_indices = next_scores.topk(beam_size, dim=-1)
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        # Update beams
        new_sequences: list[list[int]] = []
        new_hidden = hidden[:, beam_indices, :]
        new_cell = cell[:, beam_indices, :]

        active_beams = 0
        temp_scores = []

        for i in range(beam_size):
            bi = beam_indices[i].item()
            ti = token_indices[i].item()
            score = top_scores[i].item()
            seq = beam_sequences[bi] + [ti]

            if ti == tokenizer.eos_id:
                # Normalize score by length
                normalized_score = score / (len(seq) ** length_penalty)
                completed_beams.append((normalized_score, seq[:-1]))  # exclude EOS
            else:
                new_sequences.append(seq)
                temp_scores.append(score)
                active_beams += 1

        if active_beams == 0 or len(completed_beams) >= beam_size:
            break

        # Pad to beam_size if needed
        while len(new_sequences) < beam_size:
            new_sequences.append(new_sequences[-1])
            temp_scores.append(float("-inf"))

        beam_sequences = new_sequences[:beam_size]
        beam_scores = torch.tensor(temp_scores[:beam_size], device=device)
        current_token = torch.tensor(
            [seq[-1] for seq in beam_sequences[:beam_size]],
            dtype=torch.long, device=device,
        )
        hidden = new_hidden
        cell = new_cell

    # If no beam completed, take the best active one
    if not completed_beams:
        completed_beams = [
            (beam_scores[i].item() / max(1, len(beam_sequences[i])) ** length_penalty,
             beam_sequences[i])
            for i in range(len(beam_sequences))
        ]

    # Sort by score and return the best
    completed_beams.sort(key=lambda x: x[0], reverse=True)
    return [completed_beams[0][1]]


def translate_batch(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    tokenizer: SharedTokenizer,
    strategy: str = "greedy",
    beam_size: int = 5,
    max_len: int = 150,
    length_penalty: float = 0.6,
) -> list[str]:
    """Translate a batch of source sentences to text.

    Args:
        model: Trained model.
        src: (batch, src_len) source token IDs.
        src_lengths: (batch,) source lengths.
        tokenizer: Tokenizer.
        strategy: "greedy" or "beam".
        beam_size: Beam width (if strategy == "beam").
        max_len: Max decoding length.
        length_penalty: Length penalty for beam search.

    Returns:
        List of translated strings.
    """
    if strategy == "greedy":
        decoded_ids = greedy_decode(model, src, src_lengths, tokenizer, max_len)
    elif strategy == "beam":
        # Beam search processes one sentence at a time
        decoded_ids = []
        for i in range(src.size(0)):
            ids = beam_search_decode(
                model,
                src[i : i + 1],
                src_lengths[i : i + 1],
                tokenizer,
                beam_size=beam_size,
                max_len=max_len,
                length_penalty=length_penalty,
            )
            decoded_ids.extend(ids)
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")

    # Decode token IDs to text
    return [tokenizer.decode(ids) for ids in decoded_ids]
