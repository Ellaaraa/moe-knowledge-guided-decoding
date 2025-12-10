from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset
from tqdm.auto import tqdm

from .AbsTaskRetrieval import AbsTaskRetrieval
from .TaskMetadata import TaskMetadata


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
            "revision": None,
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
            split_spec = split
            if max_examples is not None:
                # Limit download size on remote (e.g., Colab) to avoid filling disk.
                split_spec = f"{split}[:{max_examples}]"
            hf_ds = load_dataset(**dataset_kwargs, split=split_spec)
            if max_examples is not None and hasattr(hf_ds, "select"):
                hf_ds = hf_ds.select(range(max_examples))

            corpus_split = {}
            queries_split = {}
            qrels_split = {}
            qa_split: List[QAExample] = []

            for row in tqdm(hf_ds, desc=f"NaturalQuestionsHF [{split}]", total=len(hf_ds)):
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
