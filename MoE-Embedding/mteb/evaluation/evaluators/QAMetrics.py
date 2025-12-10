"""
QA Metrics for Natural Questions evaluation.

Provides Exact Match (EM) and token-level F1 metrics for comparing
generated answers against ground truth answers.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for comparison.
    
    Performs:
    - Lowercase conversion
    - Punctuation removal
    - Article removal (a, an, the)
    - Extra whitespace removal
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Tokenize normalized string into words."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute Exact Match score.
    
    Returns 1.0 if the normalized prediction exactly matches any of the
    normalized ground truth answers, else 0.0.
    
    Args:
        prediction: The model's generated answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        1.0 if exact match found, 0.0 otherwise
    """
    if not prediction or not ground_truths:
        return 0.0
    
    normalized_prediction = normalize_answer(prediction)
    
    for ground_truth in ground_truths:
        if normalize_answer(ground_truth) == normalized_prediction:
            return 1.0
    
    return 0.0


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute token-level F1 score.
    
    Computes F1 between the prediction and each ground truth,
    returning the maximum F1 score.
    
    Args:
        prediction: The model's generated answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        Maximum F1 score across all ground truths (0.0 to 1.0)
    """
    if not prediction or not ground_truths:
        return 0.0
    
    prediction_tokens = get_tokens(prediction)
    
    if not prediction_tokens:
        return 0.0
    
    max_f1 = 0.0
    
    for ground_truth in ground_truths:
        gold_tokens = get_tokens(ground_truth)
        
        if not gold_tokens:
            continue
        
        # Count common tokens
        common = Counter(prediction_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            continue
        
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def compute_qa_metrics(
    predictions: List[str],
    ground_truths_list: List[List[str]]
) -> dict:
    """
    Compute aggregated QA metrics over a list of predictions.
    
    Args:
        predictions: List of model predictions
        ground_truths_list: List of ground truth lists (one per prediction)
        
    Returns:
        Dictionary with 'exact_match' and 'f1' average scores
    """
    if len(predictions) != len(ground_truths_list):
        raise ValueError("predictions and ground_truths_list must have same length")
    
    if not predictions:
        return {"exact_match": 0.0, "f1": 0.0}
    
    total_em = 0.0
    total_f1 = 0.0
    
    for pred, gts in zip(predictions, ground_truths_list):
        total_em += compute_exact_match(pred, gts)
        total_f1 += compute_f1(pred, gts)
    
    n = len(predictions)
    
    return {
        "exact_match": round(total_em / n * 100, 2),  # As percentage
        "f1": round(total_f1 / n * 100, 2)  # As percentage
    }

