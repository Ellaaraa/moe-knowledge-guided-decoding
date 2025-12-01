from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset


@dataclass
class QAExample:
    id: str
    question: str
    answers: list[str]
    context: str | None = None


def _tokens_to_text(tokens, start_token: Optional[int] = None, end_token: Optional[int] = None) -> str:
    """
    Convert NQ 'document.tokens' into plain text, dropping HTML tokens.
    Works for both:
      - list of dicts: [{"token": ..., "is_html": ...}, ...]
      - dict of lists: {"token": [...], "is_html": [...], ...}
    """
    if start_token is None:
        start_token = 0

    # Case 1: dict-of-lists (HuggingFace NQ actual format)
    if isinstance(tokens, dict):
        token_list = tokens["token"]
        is_html_list = tokens.get("is_html", [False] * len(token_list))

        if end_token is None:
            end_token = len(token_list)

        words = []
        end_token = min(end_token, len(token_list))
        for i in range(start_token, end_token):
            if not is_html_list[i]:
                words.append(token_list[i])
        return " ".join(words)

    # Case 2: list-of-dicts (our earlier toy example)
    if end_token is None:
        end_token = len(tokens)

    words = []
    for t in tokens[start_token:end_token]:
        if not t.get("is_html", False):
            words.append(t["token"])
    return " ".join(words)


def _extract_answers(row) -> list[str]:
    annotations = row["annotations"]

    # If there is no annotation, just return empty list
    if annotations is None:
        return []

    # HF NQ can sometimes store a *single* annotation as a dict,
    # or as a list/sequence of annotations.
    if isinstance(annotations, dict):
        ann = annotations
    else:
        # try to treat it as a sequence
        try:
            annotations_list = list(annotations)
        except TypeError:
            # fallback: just use it as-is
            ann = annotations
        else:
            if len(annotations_list) == 0:
                return []
            ann = annotations_list[0]

    answers: list[str] = []

    # short_answers is usually a list of dicts with a "text" field
    short_answers = ann.get("short_answers") or []
    for sa in short_answers:
        text = sa.get("text")
        # Some formats put text as a list of strings
        if isinstance(text, list):
            for t in text:
                if t:
                    answers.append(t)
        elif text:
            answers.append(text)

    # Handle yes/no questions if there are no span answers
    yes_no = ann.get("yes_no_answer", None)
    if not answers and yes_no is not None and yes_no != -1:
        # NQ uses 1 = yes, 0 = no (sometimes), -1 = none
        if yes_no == 1:
            answers.append("yes")
        elif yes_no == 0:
            answers.append("no")

    return answers


def load_nq(
    split: str = "train",
    config: str = "dev",
    max_examples: Optional[int] = None,
) -> List[QAExample]:
    """
    Load Natural Questions from HuggingFace (google-research-datasets/natural_questions)
    and convert to QAExample objects.

    Args:
        split: "train" or "validation" (depending on config).
        config: "default" or "dev" (per NQ HF builder).
        max_examples: if set, limit to first N examples.

    Returns:
        List[QAExample]
    """
    hf_ds = load_dataset(
        "google-research-datasets/natural_questions",
        config,
        split=split,
    )

    if max_examples is not None:
        hf_ds = hf_ds.select(range(max_examples))

    examples: List[QAExample] = []
    for row in hf_ds:
        q_text = row["question"]["text"]
        doc_tokens = row["document"]["tokens"]

        # full document context (you could later restrict to long_answer span)
        context_text = _tokens_to_text(doc_tokens)
        answers = _extract_answers(row)

        # if there are truly no answers, either skip or keep with []
        if not answers:
            # skip pathological examples
            continue

        examples.append(
            QAExample(
                id=str(row["id"]),
                question=q_text,
                answers=answers,
                context=context_text,
            )
        )

    return examples
