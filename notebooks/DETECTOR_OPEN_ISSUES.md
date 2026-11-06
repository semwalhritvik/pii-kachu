# Detection layer: status and open issues

This file goes with `notebooks/detector_conventions.ipynb`. That notebook scores Presidio, GLiNER and Presidio + GLiNER hybrids under `EVALUATION_CONVENTIONS.md` (branch `evaluation-conventions`). It covers what the notebook does, the dev results so far, and the questions the group still needs to settle.

## Running the notebook

- **Setup:** use the same environment as `presidio_pii_eval.ipynb`. The data must be built under `data/` first (`src/ingest`). On Windows, set `PYTHONUTF8=1`, because the ingest code opens files with the platform's default encoding.
- **Sample size:** `QUICK_TEST = True` scores 10% of dev as a smoke test; `False` scores all of dev. With `QUICK_TEST = False`, `INCLUDE_TRAIN = True` also scores all of train (Section 11). Thresholds are always tuned on dev. Test is not used.
- **Inference floors:** urchade runs at `GLINER_MIN_SCORE = 0.1` and nvidia at `NV_MIN_SCORE = 0.05`. Each is part of that model's cache key, so changing one re-runs only that model.
- **Model runs:** detector predictions are cached in `notebooks/presidio_eval_outputs/cache/`. The first run computes them, and GLiNER is much faster on a GPU. After that, a run only re-scores.
- **Outputs:** tables and exports go to `notebooks/presidio_eval_outputs/detector_conv/{quick,full}/`, which is gitignored.

## Decisions the notebook follows

| Topic | Current handling |
|---|---|
| Titles | "Mr", "Judge", "Dr" and similar titles are part of the name. The `+T` rule extends a predicted name over a capitalised title directly in front of it. Every candidate is scored with and without the rule. CRAPII spans that leave the title out are tolerated as they are. |
| Views | Two views are reported: `scoped` drops predicted labels a source never annotates (for example DEMOGRAPHIC on CRAPII); `full` scores every prediction with a redact label. Models are compared in `scoped`. |
| Sources | Every metric is reported per source. Models are compared by macro F2, the unweighted mean over sources. |
| Thresholds | Same method as the team Presidio notebook: one confidence threshold per candidate and source, swept from 0.00 to 0.95 on dev for the best span F2, with ties going to the higher threshold. |
| Leakage | Character leakage and document leakage are both reported, broken down by gold label. |
| DOB | A predicted date with a birth cue ("born", "birth", "birthday", "DOB") in the 40 characters before it is labelled DOB; any other date is DATE_OTHER. |
| Street addresses | GLiNER's address prompts map to STREET_ADDRESS. All other locations are LOCATION_OTHER, a keep label. |
| Ignored spans | TAB MISC and QUANTITY spans are excluded from scoring (`EXCLUDE_IGNORED_SPANS = True`). |
| Train split | Scored at the thresholds tuned on dev, as the team notebook applies dev thresholds to held-out data. Thresholds swept on train are reported next to them as a check. |
| Test split | Not touched. It is scored once, after the final choice. |
| Verifier export | The top three title-rule candidates are written to `conv_export_*.jsonl` (dev only). |

**Verifier export format:** one line per document, with:
- `key`, `source`, and the `threshold` applied;
- every predicted span, with `start`, `end`, `text`, the unified `label`, a `redact` flag, the `score`, and the contributing detectors.

Spans with keep labels are included, so the verifier can see what was left visible.

## Candidates

| Candidate | What it is |
|---|---|
| `presidio` | Vanilla Presidio. |
| `gliner` | `urchade/gliner_multi_pii-v1`. |
| `gliner_rules` | `urchade/gliner_multi_pii-v1` with a surface-form post-filter. |
| `nv_base_rules` | `nvidia/gliner-PII` with the same prompts and post-filter. |
| `u_presidio_gliner_rules` | Presidio ∪ `gliner_rules`, each part at its default threshold, overlapping spans merged. |
| `u_presidio_nv_rules` | Presidio ∪ `nv_base_rules`, built the same way. |
| `nv_union_lowleak+R4b` | Presidio ∪ nvidia GLiNER with fixed thresholds per entity type and per part, plus a rule that spreads name parts to their other mentions in the same document. The thresholds were set to reduce leakage under an earlier, looser match rule. |
| `nv_agree_lowleak+R4b` | Like `nv_union_lowleak+R4b`, but with one threshold for spans both models found and a separate one for spans only one model found. |

Each candidate also has a `+T` (title rule) version.

## Dev results (scoped view, threshold tuned per source)

**Macro, averaged over CRAPII and TAB:**

| Candidate | Macro F2 | Precision | Recall | Char leakage | Doc leakage |
|---|---|---|---|---|---|
| `gliner_rules+T` | 0.645 | 0.38 | 0.80 | 26.0% | 42.7% |
| `nv_base_rules+T` | 0.640 | 0.38 | 0.79 | 24.0% | 40.1% |
| `u_presidio_nv_rules+T` | 0.625 | 0.42 | 0.76 | 27.3% | 42.2% |
| `u_presidio_gliner_rules+T` | 0.622 | 0.38 | 0.79 | 25.6% | 46.2% |
| `nv_union_lowleak+R4b+T` | 0.593 | 0.53 | 0.70 | 32.8% | 61.2% |

**Per source, top three:**

| Candidate | CRAPII threshold | CRAPII P / R / F2 | CRAPII doc leakage | TAB threshold | TAB P / R / F2 | TAB doc leakage |
|---|---|---|---|---|---|---|
| `gliner_rules+T` | 0.95 | 0.27 / 0.77 / 0.56 | 23% | 0.35 | 0.49 / 0.83 / 0.73 | 62% |
| `nv_base_rules+T` | 0.20 | 0.29 / 0.83 / 0.61 | 21% | 0.05 | 0.47 / 0.75 / 0.67 | 59% |
| `u_presidio_nv_rules+T` | 0.90 | 0.27 / 0.81 / 0.57 | 24% | 0.50 | 0.57 / 0.71 / 0.68 | 61% |

### Takeaways

- **Three candidates are close.** They are within 0.02 macro F2 of each other and all Pareto-optimal. Choosing between them is a precision/recall trade-off, not a clear win.
- **The title rule is the biggest single gain** for Presidio and the nvidia-based candidates. Presidio's TAB recall goes from 0.27 to 0.55. It changes nothing for `gliner_rules`, because urchade's GLiNER already includes titles in its name spans.
- **DEMOGRAPHIC decides the ranking.** The two `lowleak` hybrids suppress DEMOGRAPHIC and find only about 9% of it on TAB, so nearly every TAB document leaks. In the `full` view they rank first instead, because demographic predictions on CRAPII count as false positives there.
- **Best thresholds differ sharply by source,** for example 0.95 on CRAPII and 0.35 on TAB for `gliner_rules`. A deployed detector does not know the source.
- **What still leaks:**
  - TAB: age phrases ("aged", "years of age"), post-name titles such as "QC", and the separators inside lists of IDs.
  - CRAPII: mostly student names.
- **Per-label gaps on CRAPII:**
  - `gliner_rules` finds 0% of ID numbers and 13% of URLs. The nvidia candidates find 76–94% of IDs, and Presidio finds 87% of URLs.
  - Dev has only one CRAPII street address.
- **These numbers are optimistic,** because thresholds are tuned and scored on the same dev documents.
- **nvidia's TAB threshold is at its floor (0.05).** Lowering the floor from 0.1 raised its TAB F2 from 0.643 to 0.671.

## Train results (scoped view, dev-tuned thresholds)

All of train: 18,151 CRAPII and 1,014 TAB documents. Thresholds are those tuned on dev, applied unchanged. The train-swept column re-tunes on train as a check.

| Candidate | Macro F2 | Macro F2, train-swept | CRAPII F2 | TAB F2 | Char leakage | Doc leakage |
|---|---|---|---|---|---|---|
| `nv_base_rules+T` | 0.645 | 0.646 | 0.604 | 0.687 | 21.4% | 40.6% |
| `u_presidio_nv_rules+T` | 0.632 | 0.632 | 0.553 | 0.710 | 25.2% | 44.6% |
| `gliner_rules+T` | 0.625 | 0.625 | 0.582 | 0.669 | 26.4% | 37.7% |
| `u_presidio_gliner_rules+T` | 0.619 | 0.619 | 0.532 | 0.706 | 23.5% | 39.4% |
| `presidio+T` | 0.547 | 0.547 | 0.448 | 0.645 | 26.3% | 63.0% |

**Paired bootstrap on train (span F2 difference, 95% interval, 1,000 document resamples):**

| Comparison | CRAPII | TAB | Macro |
|---|---|---|---|
| `nv_base_rules+T` − `gliner_rules+T` | +0.022 [+0.011, +0.034] | +0.018 [+0.008, +0.027] | +0.020 [+0.013, +0.027] |
| `u_presidio_nv_rules+T` − `nv_base_rules+T` | −0.051 [−0.062, −0.039] | +0.024 [+0.017, +0.031] | −0.014 [−0.020, −0.007] |
| `u_presidio_gliner_rules+T` − `gliner_rules+T` | −0.050 [−0.056, −0.044] | +0.038 [+0.032, +0.043] | −0.006 [−0.011, −0.002] |

- **The threshold method barely matters:** dev-tuned and train-swept thresholds give the same macro F2 to within 0.001 for these candidates.
- **nvidia leads on train**, by about 2 F2 points on both sources. On dev the two were within 0.005 the other way; dev TAB has 127 documents against 1,014 here.
- **Presidio helps TAB and hurts CRAPII** (see open issue 13).
- **The two `lowleak` fusion candidates** had their per-type thresholds searched on a 10% train sample under an earlier scoring, so they are left out of this table. On the train documents outside that sample they trail `nv_base_rules+T` (CRAPII F2 0.539 vs 0.600).

## Open issues for the group

Each item lists the question, why it matters, and how the notebook handles it now.

- [ ] **1. LOCATION_OTHER and residences.**
  - **Question:** do all non-address locations stay visible, including where a student or applicant lives ("lives in Istanbul")? If residences count as PII, which label do they get?
  - **Why it matters:** CRAPII annotates no towns or cities, only street addresses, so the benchmarks cannot measure this risk. A small hand-labelled sample of student text would be the honest test.
  - **Now:** every non-address location is LOCATION_OTHER (keep).
- [ ] **2. TAB street addresses labelled as locations.**
  - **Question:** should ingest relabel them as STREET_ADDRESS?
  - **Why it matters:** about 11 TAB dev spans labelled LOCATION_OTHER are street addresses.
  - **Now:** they are scored as keep.
- [ ] **3. DEMOGRAPHIC.**
  - **Question:** is every demographic mention PII, including the roughly 65% of TAB demographic spans that TAB's annotators marked as not needing masking?
  - **Why it matters:** this decision changes which candidate wins (see the takeaways).
  - **Now:** all are redact, as the conventions state.
- [ ] **4. Global, source-agnostic threshold.**
  - **Question:** should we move to one configuration for all sources, for example tuned on a train sample with a macro objective and selected on dev?
  - **Why it matters:** the current method picks a threshold per source, but the source is unknown at deployment and the per-source optima are far apart.
  - **Now:** per-source thresholds, following the team notebook.
- [ ] **5. Tuning and scoring on the same split.**
  - **Question:** should thresholds be tuned on a separate split (for example a train sample)?
  - **Why it matters:** tuning and scoring on dev makes the dev numbers optimistic.
  - **Now:** both happen on dev. On train, the dev thresholds score within 0.001 macro F2 of thresholds swept on train itself, so for these candidates the optimism is small.
- [ ] **6. DOB in the current gold.**
  - **Question:** should ingest relabel TAB birth dates as DOB, or do we wait for the new data? When does DOB-labelled data arrive, and in what formats ("born 1965", "12/03/1965", "date of birth: …")?
  - **Why it matters:** TAB birth dates ("born in 1965") are gold DATE_OTHER, so the DOB rule shows up as about 150 false positives on TAB dev.
- [ ] **7. Titles after names.**
  - **Question:** do post-name titles count as part of the name, like "Mr" does?
  - **Why it matters:** "QC" and similar titles are inside TAB's gold name spans, and they are the main remaining miss on names.
  - **Now:** only titles before a name are added.
- [ ] **8. ID lists.**
  - **Question:** should ingest split these lists, or should detectors merge adjacent IDs?
  - **Why it matters:** TAB annotates lists like "73653/01, 73978/01" as one span. Under full coverage the ", " between items must be covered too, so detecting each ID on its own still misses the span.
- [ ] **9. Ignored spans.**
  - **Question:** confirm that TAB MISC and QUANTITY spans stay excluded from scoring.
- [ ] **10. Verifier hand-off.**
  - **Question:** does the export format above work for the verifier? Does the verifier need the document text inline, or will it read it by `key`?
- [ ] **11. Encoding bug.**
  - **Question:** who fixes this in ingest?
  - **Why it matters:** some TAB ID spans contain a `�` replacement character introduced at ingest.
- [ ] **12. CRAPII names other than students.**
  - **Question:** do we report CRAPII precision as is, add a hand-labelled correction, or relabel a sample?
  - **Why it matters:** CRAPII's only name label is `NAME_STUDENT`. Teammates, colleagues, interviewees and friends are real private people under our schema (PERSON_PRIVATE), but a detector that finds them is charged with a false positive. In a hand-coded sample of 70 nvidia person false positives on train, about 26% were private people. Another 26% were course instructors (unclear), 34% cited authors or public figures (true false positives) and 14% not names. Counting the private share as correct would move nvidia's CRAPII span precision from about 0.30 to about 0.45.
  - **Now:** reported as is; the notebook writes the false positives with context for review (Section 11.2).
- [ ] **13. Presidio in the combination.**
  - **Question:** do we drop Presidio, or route only some entity types to it?
  - **Why it matters:** on train, adding Presidio raises TAB F2 (+0.02 to +0.04) but lowers CRAPII F2 by about 0.05, so the macro score drops slightly. A per-source choice is not deployable.
  - **Now:** both single-model and union candidates are reported.
