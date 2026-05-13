"""
Evaluation metrics: BLEU-100 and CHRF++-100 using sacrebleu.

Scores are reported on a 0–100 scale as required by the assignment.
"""

from __future__ import annotations

import sacrebleu


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute BLEU score on a 0–100 scale.

    Args:
        hypotheses: List of predicted translations.
        references: List of reference translations.

    Returns:
        BLEU score (0–100).
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    """Compute CHRF++ score on a 0–100 scale.

    Args:
        hypotheses: List of predicted translations.
        references: List of reference translations.

    Returns:
        CHRF++ score (0–100).
    """
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return chrf.score


def compute_all_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Returns:
        Dict with "bleu_100" and "chrf_100" keys.
    """
    return {
        "bleu_100": compute_bleu(hypotheses, references),
        "chrf_100": compute_chrf(hypotheses, references),
    }
