# llm_p5: Qwen on the P5 (text) purpose

One notebook, `qwen_p5_eval.ipynb`, runs `qwen3:8b` locally through Ollama as a span detector on the shared dev/test data and
checks your machine. P5 policy: **keep** public/historical figures, organizations, cities/regions/venues, non-identifying dates;
**redact** private persons, emails, phones, IDs, street addresses, personal URLs, usernames, dates of birth, demographics.

## Setup (from the repo root)
1. Install [Ollama](https://ollama.com) and start it, then `ollama pull qwen3:8b`.
2. `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r llm_p5/requirements.txt`
3. Unzip the shared data into `data/processed/` so that `data/processed/dev.jsonl` and `data/processed/test.jsonl` exist.
   The data is gitignored: never commit it.
4. `jupyter lab llm_p5/qwen_p5_eval.ipynb`. Run **Sections 0-3 first** (config, helpers, data check, system check), then 4-10.
   Optional helper check (no model needed): `pip install pytest && python -m pytest llm_p5/tests`.

Keep `RUN_SPLITS = ["dev"]` until the prompt is frozen. Runs are resumable; delete `llm_p5/results/qwen_*.jsonl` to start over.
Runtimes printed in Section 4 are estimates (windows x median seconds per window from the system check); real windows are longer.

## What to send back
From `llm_p5/results/`: `summary_<chip>_<ram>gb.md`, `summary_<chip>_<ram>gb.csv` and `system_check_<chip>_<ram>gb.json`.
Do not send the data or the per-document `qwen_*.jsonl` rows.

## Definitions
- **Sampling** (seed 42, document level): CRAPII 150 dev / 150 test, stratified so half the docs have at least one gold span;
  TAB 10 dev / 10 test, random. `SAMPLE_IDS_FILE` (JSON list of doc_ids) overrides this to use the same docs as Presidio/GLiNER.
- **CRAPII gold**: every span is redact.
- **TAB gold, scope `direct+quasi`**: DIRECT and QUASI = redact; NO_MASK (incl. PERSON_PUBLIC) = keep.
- **TAB gold, scope `direct`**: DIRECT = redact; QUASI = ignore (predictions fully inside ignored regions are excluded from
  precision and over-redaction); NO_MASK = keep.
- MISC/QUANTITY spans are ignored in both scopes.
- **TAB annotator**: `first` = the primary annotator (the `spans` field); `union` = primary plus `other_annotations`.
- Metrics are computed per source and split (never pooled), whitespace excluded, with 95% bootstrap CIs over documents
  (1000 resamples, seed 42). Definitions are in notebook Section 6; `keep_false_redaction_rate` is the share of gold keep spans
  overlapped by a REDACT prediction.
