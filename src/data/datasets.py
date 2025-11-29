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
    and restrict to [start_token, end_token).
    """
    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = len(tokens)

    words: List[str] = []
    for t in tokens[start_token:end_token]:
        # tokens have shape: {"token": str, "start_byte": int, "end_byte": int, "is_html": bool}
        if not t.get("is_html", False):
            words.append(t["token"])
    return " ".join(words)


def _extract_answers(row) -> list[str]:
    """
    Extract short answer texts from NQ annotations.
    If no short answers, fall back to the long answer span (as a single string),
    otherwise return [].
    """
    annotations = row.get("annotations", [])
    if not annotations:
        return []

    ann = annotations[0]

    # 1) Prefer short answers if present
    short_answers = ann.get("short_answers", []) or []
    answers = [sa["text"] for sa in short_answers if sa.get("text")]
    if answers:
        return answers

    # 2) Fallback: use long answer tokens if available
    long_ans = ann.get("long_answer", {}) or {}
    cand_idx = long_ans.get("candidate_index", -1)
    if cand_idx is None or cand_idx < 0:
        return []

    cands = row.get("long_answer_candidates", []) or []
    if not (0 <= cand_idx < len(cands)):
        return []

    cand = cands[cand_idx]
    start_tok = cand.get("start_token", 0)
    end_tok = cand.get("end_token", 0)

    doc_tokens = row["document"]["tokens"]
    long_text = _tokens_to_text(doc_tokens, start_tok, end_tok)
    return [long_text] if long_text.strip() else []


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
