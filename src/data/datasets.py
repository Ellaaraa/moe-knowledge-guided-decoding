from __future__ import annotations
from dataclasses import dataclass
from typing import List
from datasets import load_dataset


@dataclass
class QAExample:
    id: str
    question: str
    answers: list[str]
    context: str | None = None


def load_nq(split: str = "train") -> List[QAExample]:
    """
    Load Natural Questions from HuggingFace and convert to QAExample objects.

    The dataset schema (for the subset you're using):
        - query: str
        - answer:   str
    """

    hf_ds = load_dataset("sentence-transformers/natural-questions", split=split)

    examples: List[QAExample] = []
    for idx, row in enumerate(hf_ds):
        question = row["query"]
        answer = row["answer"]      # single string
        answers = [answer]          # wrap as list for consistency with QAExample

        examples.append(
            QAExample(
                id=str(idx), 
                question=question,
                answers=answers,
                context=None,
            )
        )

    return examples