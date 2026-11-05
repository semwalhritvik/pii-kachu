# pii-kachu
Local PII detection research. Baselines: Presidio and GLiNER. The LLM layer is **P5 only (text purpose)**.

## LLM layer
- `llm_p5/` is the teammate-facing folder: one notebook (`qwen_p5_eval.ipynb`), a synthetic system-check file, tests, README.
- `src/judge/` and `pilot/` are the completed six-purpose bake-off, kept as reference. Do not move or delete them.
- The P5 policy is inlined in the notebook. No config reads from the notebook.

## Rules
- Never read or print full files in `data/`. Use `head -c 1500` or sample 3 records, or small scripts.
- Never commit `data/`, `llm_p5/results/` or predictions (`results/preds/`). Before a commit: `git ls-files | grep -E '^(data/|llm_p5/results/)'` must print nothing but `.gitkeep`.
- Run `pytest llm_p5/tests` before commits. Tests use a fake local server.
- Never run models (Ollama) unless the user asks.
- Never push.
- Keep outputs terse; no summaries of what you did beyond 5 lines.
- Python 3.11, venv at `.venv`. Notebook deps: stdlib + numpy, pandas, matplotlib.
- Document-level splits only, seed=42.

## Data
Unified JSONL, one doc per line:
`{"doc_id","source","text","spans":[{"start","end","label","attrs":{}}],"split"}` (TAB also has `ignored_spans`, `other_annotations`).

## Layout
configs/ src/ingest/ src/baselines/ src/eval/ src/judge/ (reference) pilot/ llm_p5/ notebooks/ results/ data/ (gitignored)
