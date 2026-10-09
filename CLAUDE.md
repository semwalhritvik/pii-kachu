# pii-kachu
Local PII detection research (Presidio vs GLiNER baselines; later a purpose-aware over/under-redaction judge).
Rules:
- Never read/print full files in data/. Use head -c 1500 or sample 3 records.
- Never commit data/ or predictions. Keep outputs terse; no summaries of what you did beyond 5 lines.
- Python 3.11, venv at .venv, deps in requirements.txt. Prefer stdlib+pandas.
- Unified format: JSONL, one doc per line:
  {"doc_id","source","text","spans":[{"start","end","label","attrs":{}}],"split"}
- Document-level splits only, seed=42.
Layout: configs/ src/ingest/ src/baselines/ src/eval/ notebooks/ results/ data/(gitignored)