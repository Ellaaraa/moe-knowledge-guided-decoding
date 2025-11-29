from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional

import torch

from models.olmoe import OLMoELM
from data.datasets import QAExample
from decoding.vanilla import build_vanilla_prompt


@dataclass
class KGDPrediction:
    """Prediction container for Knowledge-Guided Decoding."""
    id: str
    question: str
    prediction: str
    gold_answers: list[str]
    alpha: float      # interpolation weight
    num_docs: int     # how many docs were used


# -------------------------- retrieval stub -------------------------- #

def retrieve_docs(example: QAExample, top_k: int = 1) -> List[str]:
    """
    Placeholder retriever.

    For now we just use `example.context` if it exists.
    Later you will replace this with a real retriever over your corpus
    (e.g., BM25 / dense retriever that returns top-k passages).
    """
    docs: List[str] = []
    if example.context:
        docs.append(example.context)
    return docs[:top_k]


# ------------------- knowledge logits computation ------------------- #

def _compute_knowledge_logits(
    model: OLMoELM,
    base_prompt: str,
    docs: Sequence[str],
) -> Optional[torch.Tensor]:
    """
    Run the LM once with (context + question) to get a 'knowledge prior'
    over the vocabulary. We then mix this with the base LM logits.

    Returns:
        Tensor of shape (1, vocab_size), or None if no docs.
    """
    if not docs:
        return None

    device = model.device
    all_logits = []

    for doc in docs:
        knowledge_prompt = (
            "Use the following background knowledge to answer the question.\n\n"
            f"Context: {doc}\n\n"
            f"{base_prompt}"
        )
        inputs = model.tokenizer(
            knowledge_prompt,
            return_tensors="pt",
            max_length=model.max_length,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            out = model.model(**inputs)
            # last-token logits
            logits = out.logits[:, -1, :]   # (1, vocab)
        all_logits.append(logits)

    # average logits over docs
    stacked = torch.stack(all_logits, dim=0)  # (num_docs, 1, vocab)
    return stacked.mean(dim=0)                # (1, vocab)


# -------------------------- KGD decoding ---------------------------- #

def kgd_decode_single(
    model: OLMoELM,
    example: QAExample,
    alpha: float = 0.5,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> KGDPrediction:
    """
    Decode one QAExample with Knowledge-Guided Decoding.

    alpha = 0.0  -> pure vanilla
    alpha = 1.0  -> purely knowledge-prior-driven
    """
    device = model.device
    tokenizer = model.tokenizer

    # vanilla QA prompt
    base_prompt = build_vanilla_prompt(example)

    # retrieve docs (you'll replace this with real retrieval later)
    docs = retrieve_docs(example, top_k=1)
    knowledge_logits = _compute_knowledge_logits(model, base_prompt, docs)

    # tokenize base prompt
    inputs = tokenizer(
        base_prompt,
        return_tensors="pt",
        max_length=model.max_length,
        truncation=True,
    ).to(device)

    input_ids = inputs["input_ids"]  # (1, L)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )
            base_logits = out.logits[:, -1, :]   # (1, vocab)

        # interpolate with knowledge prior if available
        logits = base_logits
        if knowledge_logits is not None:
            logits = (1.0 - alpha) * base_logits + alpha * knowledge_logits

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs[0], num_samples=1)  # (1,)
        else:
            next_token_id = torch.argmax(logits, dim=-1)  # (1,)

        token_id = next_token_id.item()
        # append token
        input_ids = torch.cat(
            [input_ids, next_token_id.unsqueeze(0)], dim=1
        )  # (1, L+1)

        if token_id == tokenizer.eos_token_id:
            break

    # decode full text and extract answer like vanilla
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # everything after "Answer:" first line = prediction
    if "Answer:" in full_text:
        answer = full_text.split("Answer:", 1)[1].strip().split("\n")[0]
    else:
        answer = full_text.strip().split("\n")[-1]

    return KGDPrediction(
        id=example.id,
        question=example.question,
        prediction=answer,
        gold_answers=example.answers,
        alpha=alpha,
        num_docs=len(docs),
    )


def kgd_decode(
    model: OLMoELM,
    examples: Iterable[QAExample],
    alpha: float = 0.5,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    show_progress: bool = True,
) -> List[KGDPrediction]:
    """
    Run KGD on a collection of QAExample objects.
    """
    iterator = examples
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(examples, desc=f"Decoding (KGD, alpha={alpha})")
        except ImportError:
            iterator = examples

    preds: List[KGDPrediction] = []
    for ex in iterator:
        pred = kgd_decode_single(
            model,
            ex,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        preds.append(pred)

    return preds
