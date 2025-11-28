from __future__ import annotations

import re
import string
from typing import Iterable, List, Tuple

from decoding.vanilla import VanillaPrediction


# --- Text normalization helpers (SQuAD-style) ------------------------------


def _normalize_answer(text: str) -> str:
    """
    Lowercase, remove punctuation, articles, and extra whitespace.
    """
    def lower(t: str) -> str:
        return t.lower()

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in string.punctuation)

    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    text = lower(text)
    text = remove_punc(text)
    text = remove_articles(text)
    text = white_space_fix(text)
    return text


def _f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 between a single prediction and a single gold answer.
    """
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = {}
    for tok in pred_tokens:
        common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))

    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    Exact match after normalization.
    """
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _max_over_ground_truths(
    metric_fn,
    prediction: str,
    gold_answers: List[str],
) -> float:
    """
    For questions with multiple gold answers, take the best score.
    """
    if not gold_answers:
        # If no gold answers exist, treat as 0 score.
        return 0.0
    scores = [metric_fn(prediction, gt) for gt in gold_answers]
    return max(scores)


# --- Public API ------------------------------------------------------------


def compute_em_f1(
    predictions: Iterable[VanillaPrediction],
) -> Tuple[float, float]:
    """
    Compute dataset-level Exact Match (EM) and F1.

    Args:
        predictions: iterable of VanillaPrediction objects

    Returns:
        (em, f1) as percentages, e.g., (34.2, 47.8)
    """
    total = 0
    em_sum = 0.0
    f1_sum = 0.0

    for pred in predictions:
        total += 1
        em = _max_over_ground_truths(
            _exact_match_score,
            pred.prediction,
            pred.gold_answers,
        )
        f1 = _max_over_ground_truths(
            _f1_score,
            pred.prediction,
            pred.gold_answers,
        )
        em_sum += em
        f1_sum += f1

    if total == 0:
        return 0.0, 0.0

    em = 100.0 * em_sum / total
    f1 = 100.0 * f1_sum / total
    return em, f1
