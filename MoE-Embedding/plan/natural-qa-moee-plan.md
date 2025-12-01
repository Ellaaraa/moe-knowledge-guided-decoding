# Natural Questions + MoEE (OLMoE) Plan

Goal
Build a clean, self-contained Natural Questions (NQ) pipeline that evaluates MoEE embeddings (using OLMoE) on NQ without depending on the KGD code. Keep it aligned with the MoE-Embedding repo layout.

Scope choices
- Model: OLMoE (default: `allenai/OLMoE-1B-7B-0924` or similar), using existing MoEE embedding extraction.
- Retrieval: Implement a minimal in-repo retrieval/eval loop (FAISS or torch top-k) rather than relying on the KGD code; MTEB is optional, but we’ll keep the interface similar.
- Data: HuggingFace `google-research-datasets/natural_questions`, validation split, configurable slicing for quick runs.

Plan
1) Data ingestion
- Add `datasets/nq.py` to load NQ (train/validation) directly from HF.
- Parse tokens -> text (strip HTML), pick a document context (full doc or long-answer window), keep question id, question text, and short/yes-no answers.
- Provide sampling/slicing (e.g., `--max_examples`, `--split_slice`) for fast smoke runs.

2) Passage construction
- Build a passage corpus from each example’s context: either full doc or a window around the long-answer span; optional character-level chunking (size/overlap).
- Keep mappings: passage_id -> raw text; question_id -> list of relevant passage_ids (for labeling retrieval correctness).

3) Embedding + retrieval pipeline
- Use `MOEE` with OLMoE to embed queries and passages; expose pooling/normalization options already supported by MoEE.
- Implement a lightweight retriever (FAISS or torch top-k over in-memory embeddings) to rank passages for each question.
- Make retrieval config-driven: `--top_k`, `--chunk_size/overlap`, `--use_4bit`, `--embed_method`.

4) Evaluation
- Compute retrieval metrics (Recall@k, MRR@k; optionally nDCG@10) using labels from passage_id mapping.
- Dump predictions to JSONL (question_id, top_k passage_ids, scores) and metrics to a results JSON for inspection.

5) CLI wiring
- New entry script `eval_nq_moee.py` in `MoE-Embedding/` to run end-to-end: load NQ, build passages, embed, retrieve, score, save outputs.
- Arguments: dataset split/config, sampling, passage construction options, model/base args, embedding options, retrieval k, output paths, device.

6) Validation & tests
- Add small unit tests for token-to-text parsing and answer extraction.
- Add a smoke test on `validation[:20]` that runs embed + retrieval on CPU (small model) to ensure the loop executes.

7) Docs
- Update `MoE-Embedding/README.md` (or a short `docs/nq.md`) with usage examples for `eval_nq_moee.py`, expected runtime/VRAM, and notes about HF download/auth if needed.

Notes on reuse
- We reuse the MoEE model code and general embedding interface, but the NQ loader/retrieval/eval is new and self-contained (no dependency on `src/` KGD code). If desired, MTEB’s NQ task can be swapped in later via a thin adapter.
