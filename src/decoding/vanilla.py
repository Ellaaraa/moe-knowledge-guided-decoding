from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from models.olmoe import OLMoELM
from data.datasets import QAExample

from data.datasets import load_nq


@dataclass
class VanillaPrediction:
    """Container for a single vanilla decoding prediction."""
    id: str
    question: str
    prediction: str
    gold_answers: list[str]


def build_vanilla_prompt(example: QAExample) -> str:
    """
    Build a simple prompt for vanilla QA.

    You can tweak the instruction text here, but keep it consistent
    across all baselines so comparisons with KGD are fair.
    """
    return (
        "You are a factual question-answering assistant. "
        "Answer briefly with just the final answer phrase.\n\n"
        f"Question: {example.question}\n"
        "Answer:"
    )


def decode_vanilla(
    model: OLMoELM,
    examples: Iterable[QAExample],
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    show_progress: bool = True,
) -> List[VanillaPrediction]:
    """
    Run vanilla decoding on a sequence of QAExample objects.

    Args:
        model: An initialized OLMoELM model.
        examples: Iterable of QAExample.
        max_new_tokens: Max tokens to generate for each answer.
        temperature: 0.0 = greedy; >0 enables sampling.
        show_progress: If True, use tqdm if available.

    Returns:
        List of VanillaPrediction objects.
    """
    iterator = examples

    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(examples, desc="Decoding (vanilla)")
        except ImportError:
            # tqdm is optional; silently fall back if not installed
            iterator = examples

    preds: List[VanillaPrediction] = []

    for ex in iterator:
        prompt = build_vanilla_prompt(ex)
        raw_output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Simple post-processing: take the first line as the answer.
        # You can refine this later if needed.
        answer = raw_output.split("\n")[0].strip()

        preds.append(
            VanillaPrediction(
                id=ex.id,
                question=ex.question,
                prediction=answer,
                gold_answers=ex.answers,
            )
        )

    return preds


def main():
    model = OLMoELM()
    dev_examples = load_nq(split="train[:100]")  # small subset for testing

    preds = decode_vanilla(model, dev_examples)

    # later:
    # em, f1 = compute_em_f1(preds)
    # print({"em": em, "f1": f1})

if __name__ == "__main__":
    main()
