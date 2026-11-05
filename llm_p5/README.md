# llm_p5: Qwen on the P5 (text) purpose

One notebook, `qwen_p5_eval.ipynb`, runs `qwen3:8b` locally through Ollama as a span detector on the shared **dev** and **train**
samples and checks your machine. P5 policy: **keep** public/historical figures, organizations, cities/regions/venues,
non-identifying dates; **redact** private persons, emails, phones, IDs, street addresses, personal URLs, usernames, dates of birth, demographics.
**Test is locked** (the notebook refuses to run it).

## Setup (from the repo root)
1. Install [Ollama](https://ollama.com), start it, then `ollama pull qwen3:8b`. Check the digest later: the system check prints it and
   stores it in its JSON; it must match the one the team uses (a mismatch makes machine comparisons invalid).
2. `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r llm_p5/requirements.txt`
3. Unzip the shared data into `data/processed/` so that `dev.jsonl`, `train.jsonl` and `test.jsonl` exist there. Never commit it.
4. `jupyter lab llm_p5/qwen_p5_eval.ipynb`. Optional helper check (no model needed): `pip install pytest && python -m pytest llm_p5/tests`.

## Order of runs
Run the notebook top to bottom. Sections: Setup, Helpers, Data check, System check, DEV run, Dev metrics, Compare machines on dev,
TRAIN run, Train benchmark, Export.
1. **Setup**: set `MAX_HOURS_TRAIN` (e.g. `8`) if you want a wall-clock cap for the train run. Leave `ALLOW_PROMPT_CHANGE = False`.
2. **System check first**, then send `llm_p5/results/system_check_<chip>_<ram>gb.json` to the team before the long runs.
3. **DEV run**, then Dev metrics (this is the machine-comparison run). Then **TRAIN run** and Train benchmark.
4. Compare machines on dev: put the other machine's `qwen_standalone_dev.jsonl` in `llm_p5/results/incoming/` and set `OTHER_DEV_FILE`.

**Stop and resume:** interrupt the running cell (Kernel > Interrupt) at any time and run it again: finished docs are skipped. A time
budget (`MAX_HOURS_*`) only stops starting new docs; finished docs form a stratified random subset. To enlarge a sample, only increase its
size in `SAMPLE_SIZES` (a larger sample extends the smaller one). Delete `llm_p5/results/qwen_*.jsonl` to start over.
Runtimes printed before each run are estimates (windows x median s/window from the system check); real windows are longer.

## What to send back
From `llm_p5/results/`: `system_check_<chip>_<ram>gb.json` (first), `qwen_standalone_dev.jsonl`, `qwen_standalone_train.jsonl`,
`summary_<chip>_<ram>gb.md` and `.csv`, and `machine_comparison_dev.png` if you made it.
**Warning: the `qwen_*.jsonl` rows contain document text** (and `train_worst30_errors.csv` has snippets). Send them privately like the
data zip, never via git or any public channel. `llm_p5/results/` is gitignored; keep it that way.

## Definitions
- **Sampling** (seed 42, document level): dev = CRAPII 150 + TAB 20, train = CRAPII 300 + TAB 25. CRAPII uses two seeded permutations
  (docs with at least one gold span, docs with none) interleaved 1:1; TAB uses one seeded permutation. Sources are interleaved in
  proportion to their sizes. `SAMPLE_IDS_FILE` (JSON list of doc_ids) overrides this.
- **CRAPII gold**: every span is redact.
- **TAB gold, scope `direct+quasi`**: DIRECT and QUASI = redact; NO_MASK (incl. PERSON_PUBLIC) = keep.
- **TAB gold, scope `direct`**: DIRECT = redact; QUASI = ignore (predictions fully inside ignored regions are excluded from
  precision and over-redaction); NO_MASK = keep. MISC/QUANTITY spans are ignored in both scopes.
- **TAB annotator**: `first` = the primary annotator (the `spans` field); `union` = primary plus `other_annotations`.
- Metrics are per source and split (never pooled), whitespace excluded, 95% bootstrap CIs over documents (1000 resamples, seed 42).
  `keep_false_redaction_rate` is the share of gold keep spans overlapped by a REDACT prediction.
- **Prompt freeze**: every row records `prompt_version`, `prompt_hash`, model digest, machine (chip, RAM, OS, Ollama version) and
  `sample_fingerprint`. Editing the prompt stops the run unless `ALLOW_PROMPT_CHANGE = True` (rows are then stamped `prompt_modified`).
- Train is for error mining and choosing few-shot examples; docs used as few-shot examples must later be excluded from train scoring.
  Train numbers are the pre-tuning reference, not report numbers.
