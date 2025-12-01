# Natural Questions Support Plan

Context
- Repo has KGD + vanilla QA over HF datasets; `load_nq` exists but parsing/config is minimal and run scripts are hard-coded.
- MoE-Embedding side focuses on MTEB; NQ appears as a retrieval task but not wired into the QA decoding pipeline.

Goals
- First-class Natural Questions support with robust parsing, configurable loading, retrieval-aware prompting, evaluation, and docs.

Plan
1) Data layer cleanup
- Extend `src/data/datasets.py` with a small registry + `DatasetConfig` (dataset name, split, config, max_examples) and a public `load_dataset(name, ...)` entry point.
- Harden NQ parsing: prefer long-answer spans when available, keep short answers (plus yes/no), strip HTML tokens, allow context truncation and sampling for quick runs.
- Add optional caching/slicing helpers so scripts can quickly grab `validation[:N]` without ad-hoc string splits.

2) Retrieval & preprocessing
- Add helper to build a context corpus from NQ (full doc vs. long-answer window) and chunk it with configurable size/overlap.
- Expose retrieval knobs (BM25 vs embedding retriever, model name, top_k) and plumb them into KGD so context quality is controllable per dataset.

3) Decoding pipeline integration
- Update `src/run_kgd.py` and `src/run_vanilla.py` to take dataset args (`--dataset natural_questions`, `--split`, `--config`, `--max-examples`, retrieval/chunk params) instead of hard-coded NQ slices.
- Fix the `kgd_decode` API mismatch (callers pass `use_chat_template`; function should accept and forward tokenizer/chat-template choice cleanly).
- Ensure prompts handle missing context gracefully and log dataset metadata (name/split/config) with each run.

4) Evaluation outputs
- Keep EM/F1; include dataset name/split in printed summaries and optional JSONL dumps of predictions/metadata for error analysis.
- If MoE-Embedding experiments need NQ retrieval, add a config entry so MTEB runners can invoke it consistently.

5) Testing/validation
- Add unit tests for `_tokens_to_text` and `_extract_answers` on dict/list formats, HTML tokens, and yes/no cases.
- Add an integration smoke test on `validation[:10]` to ensure end-to-end decode + metric runs without GPU-only features.

6) Documentation
- Document NQ support and CLI usage in `README.md`/`KGD_IMPLEMENTATION.md` (dataset options, retrieval knobs, resource needs, HF cache/auth notes).
- Add a short example command for both vanilla and KGD runs on a small NQ slice.

Dependencies/infra notes
- Uses existing `datasets` and `sentence-transformers`; no new deps anticipated beyond optional BM25 if enabled.
