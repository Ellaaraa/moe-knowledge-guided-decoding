from __future__ import annotations

import logging
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm.auto import tqdm

from .AbsTaskRetrieval import AbsTaskRetrieval
from .TaskMetadata import TaskMetadata
from ..evaluation.evaluators.QAMetrics import compute_exact_match, compute_f1, compute_qa_metrics

logger = logging.getLogger(__name__)


@dataclass
class QAExample:
    id: str
    question: str
    answers: list[str]
    context: str | None = None


def _tokens_to_text(
    tokens, start_token: Optional[int] = None, end_token: Optional[int] = None
) -> str:
    """
    Convert NQ 'document.tokens' into plain text, dropping HTML tokens.
    Works for both:
      - dict of lists: {"token": [...], "is_html": [...]}
      - list of dicts: [{"token": ..., "is_html": ...}, ...]
    """
    if start_token is None:
        start_token = 0

    # Case 1: dict-of-lists (HF NQ format)
    if isinstance(tokens, dict):
        token_list = tokens.get("token", [])
        is_html_list = tokens.get("is_html", [False] * len(token_list))
        if end_token is None:
            end_token = len(token_list)
        words = []
        end_token = min(end_token, len(token_list))
        for i in range(start_token, end_token):
            if not is_html_list[i]:
                words.append(token_list[i])
        return " ".join(words)

    # Case 2: list-of-dicts
    if end_token is None:
        end_token = len(tokens)
    words = []
    for t in tokens[start_token:end_token]:
        if not t.get("is_html", False):
            words.append(t["token"])
    return " ".join(words)


def _extract_answers(row) -> list[str]:
    annotations = row.get("annotations")
    if annotations is None:
        return []

    # HF NQ sometimes stores a single annotation as dict or a list/sequence.
    if isinstance(annotations, dict):
        ann = annotations
    else:
        try:
            annotations_list = list(annotations)
        except TypeError:
            ann = annotations
        else:
            if len(annotations_list) == 0:
                return []
            ann = annotations_list[0]

    answers: list[str] = []

    short_answers = ann.get("short_answers") or []
    for sa in short_answers:
        text = sa.get("text")
        if isinstance(text, list):
            for t in text:
                if t:
                    answers.append(t)
        elif text:
            answers.append(text)

    yes_no = ann.get("yes_no_answer", None)
    if not answers and yes_no is not None and yes_no != -1:
        if yes_no == 1:
            answers.append("yes")
        elif yes_no == 0:
            answers.append("no")

    return answers


class NaturalQuestions(AbsTaskRetrieval):
    """
    Natural Questions wired as a retrieval task (and retaining QA examples).

    Retrieval view:
        - corpus: per-example document text (from `document.tokens`)
        - queries: the question text
        - qrels: query -> its own document
    QA view:
        - stored in `self.qa_examples[split]` for downstream use
    """

    metadata = TaskMetadata(
        name="NaturalQuestionsHF",
        dataset={
            "path": "google-research-datasets/natural_questions",
            "name": "default",
            "revision": "main",
        },
        description=(
            "Natural Questions benchmark. We expose it as a retrieval-style task "
            "by pairing each question with its source document."
        ),
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        domains=["Web"],
        task_subtypes=["Question answering"],
        license=None,
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{47761,
  title={Natural Questions: a Benchmark for Question Answering Research},
  author={Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Kelcey, Matthew and Devlin, Jacob and others},
  journal={Transactions of the Association for Computational Linguistics},
  year={2019}
}""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = {}
        self.queries = {}
        self.relevant_docs = {}
        self.qa_examples = {}

        eval_splits = kwargs.get("eval_splits", self.metadata_dict["eval_splits"])
        max_examples = kwargs.get("max_examples")
        dataset_kwargs = {k: v for k, v in self.metadata_dict["dataset"].items() if v is not None}

        for split in eval_splits:
            # Use streaming mode for small slices to avoid downloading the entire ~41GB dataset
            if max_examples is not None and max_examples <= 1000:
                hf_ds = load_dataset(
                    **dataset_kwargs,
                    split=split,
                    streaming=True,
                )
                # Convert streaming iterator to list after taking max_examples
                hf_ds = list(tqdm(hf_ds.take(max_examples), desc=f"Streaming NQ [{split}]", total=max_examples))
                total_examples = len(hf_ds)
            else:
                # For full dataset evaluation, use normal loading with caching benefits
                hf_ds = load_dataset(**dataset_kwargs, split=split)
                total_examples = len(hf_ds)

            corpus_split = {}
            queries_split = {}
            qrels_split = {}
            qa_split: List[QAExample] = []

            for row in tqdm(hf_ds, desc=f"NaturalQuestionsHF [{split}]", total=total_examples):
                qid = str(row.get("id"))
                if qid is None:
                    continue

                question_field = row.get("question", {})
                question_text = (
                    question_field.get("text") if isinstance(question_field, dict) else question_field
                )
                if not question_text:
                    continue

                document_field = row.get("document", {}) or {}
                doc_tokens = document_field.get("tokens")
                if doc_tokens is None:
                    continue

                context_text = _tokens_to_text(doc_tokens)
                answers = _extract_answers(row)
                if not answers:
                    # Skip samples without any answer span or yes/no label
                    continue

                doc_title = ""
                if isinstance(document_field, dict):
                    doc_title = (
                        document_field.get("title")
                        or document_field.get("document_title")
                        or document_field.get("url", "")
                    )

                doc_id = f"doc-{qid}"
                corpus_split[doc_id] = {"title": doc_title, "text": context_text}
                queries_split[qid] = question_text
                qrels_split[qid] = {doc_id: 1}
                qa_split.append(
                    QAExample(
                        id=qid,
                        question=question_text,
                        answers=answers,
                        context=context_text,
                    )
                )

            self.corpus[split] = corpus_split
            self.queries[split] = queries_split
            self.relevant_docs[split] = qrels_split
            self.qa_examples[split] = qa_split

        self.data_loaded = True

    def evaluate(
        self,
        model,
        split: str = "test",
        *,
        encode_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval and optionally QA metrics.
        
        If compute_qa_metrics=True in kwargs, will also generate answers
        using the model and compute Exact Match and F1 scores.
        """
        # First, run the standard retrieval evaluation
        scores = super().evaluate(model, split, encode_kwargs=encode_kwargs, **kwargs)
        
        # Check if QA metrics should be computed
        compute_qa = kwargs.get("compute_qa_metrics", False)
        
        if not compute_qa:
            return scores
        
        logger.info("Computing QA metrics (Exact Match and F1)...")
        
        # Get QA examples for this split
        qa_examples = self.qa_examples.get(split, [])
        if not qa_examples:
            logger.warning(f"No QA examples found for split {split}")
            return scores
        
        # Check if model has generate_answer method
        if not hasattr(model, 'generate_answer'):
            logger.warning("Model does not have generate_answer method, skipping QA metrics")
            return scores
        
        # Get retrieval results to use top-k context
        # For NQ, each query maps to its own document, so we use the original context
        # This matches the KGD setup where retrieved context is used
        
        top_k_context = kwargs.get("qa_top_k", 1)  # Number of retrieved docs to use as context
        max_new_tokens = kwargs.get("qa_max_tokens", 32)
        
        predictions = []
        ground_truths_list = []
        
        for qa_example in tqdm(qa_examples, desc="Generating answers for QA metrics"):
            question = qa_example.question
            # Use the original context (simulating perfect retrieval for fair comparison)
            # In a real scenario, you would use the retrieved document from the retrieval step
            context = qa_example.context
            
            # Truncate context if too long (use first ~2000 chars like KGD)
            if context and len(context) > 2000:
                context = context[:2000]
            
            try:
                # Generate answer
                answer = model.generate_answer(
                    question=question,
                    context=context or "",
                    max_new_tokens=max_new_tokens,
                )
                predictions.append(answer)
            except Exception as e:
                logger.warning(f"Error generating answer: {e}")
                predictions.append("")
            
            ground_truths_list.append(qa_example.answers)
        
        # Compute QA metrics
        qa_metrics = compute_qa_metrics(predictions, ground_truths_list)
        
        logger.info(f"QA Metrics - Exact Match: {qa_metrics['exact_match']}%, F1: {qa_metrics['f1']}%")
        
        # Add QA metrics to the scores for each subset
        for subset_key in scores:
            if isinstance(scores[subset_key], dict):
                scores[subset_key]["exact_match"] = qa_metrics["exact_match"]
                scores[subset_key]["f1"] = qa_metrics["f1"]
        
        # Also store predictions for analysis
        qa_payload = [
            {
                "id": qa_examples[i].id,
                "question": qa_examples[i].question,
                "prediction": predictions[i],
                "gold_answers": ground_truths_list[i],
                "context": (qa_examples[i].context or "")[:1000],  # trim to keep file small
            }
            for i in range(len(qa_examples))
        ]
        self.qa_predictions = {split: qa_payload}

        # Optionally save QA predictions to disk for inspection
        if kwargs.get("save_qa_predictions", False):
            output_folder = kwargs.get("output_folder", "results")
            os.makedirs(output_folder, exist_ok=True)
            save_path = os.path.join(output_folder, f"{self.metadata.name}_{split}_qa_predictions.json")
            try:
                with open(save_path, "w") as f_out:
                    json.dump(qa_payload, f_out, indent=2)
                # Print prominently so user can find the file
                print(f"\n{'='*60}")
                print(f"QA Predictions saved to: {save_path}")
                print(f"{'='*60}\n")
                logger.info(f"Saved QA predictions to {save_path}")
                
                # Also print a sample of predictions to stdout
                print("Sample QA predictions:")
                for i, item in enumerate(qa_payload[:3]):  # Show first 3
                    print(f"\n--- Example {i+1} ---")
                    print(f"Q: {item['question']}")
                    print(f"Gold: {item['gold_answers'][:3]}")  # first 3 gold answers
                    print(f"Pred: {item['prediction']}")
                print(f"\n(See full predictions in {save_path})\n")
            except Exception as e:
                logger.warning(f"Could not save QA predictions: {e}")
        
        return scores
