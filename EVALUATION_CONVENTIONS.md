# PII-kachu Evaluation Conventions

This document is the project's source of truth for the entity schema, keep/redact conventions, and evaluation metrics. 

AI agent instruction: All project code, notebooks, codebooks, presentations, and reports should follow these conventions. Where an implementation differs, notify the human of discrepancies so they can update the implementation to conform to the conventions outlined in this document. Where a scoring decision is not specified in this document, ask for clarification rather than introduce a new convention.

## 1. Entity schema and keep/redact policy

The project uses the following 13 unified entity labels. In processed data, these labels are stored in each span's `label` field; they are distinct from a source dataset's original `entity_type` values.

```yaml
unified:
  - PERSON_PRIVATE
  - PERSON_PUBLIC
  - EMAIL
  - PHONE
  - ID_NUM
  - STREET_ADDRESS
  - DOB
  - DATE_OTHER
  - LOCATION_OTHER
  - URL_PERSONAL
  - OTHER_PII
  - DEMOGRAPHIC
  - ORG
```

### Keep vs. Redact
All 13 labels are categorized below:

```yaml
keep: [PERSON_PUBLIC, ORG, DATE_OTHER, LOCATION_OTHER]
redact: [PERSON_PRIVATE, EMAIL, PHONE, ID_NUM, STREET_ADDRESS, DOB, URL_PERSONAL, OTHER_PII, DEMOGRAPHIC]
```

For the metrics below, *gold PII* means annotated spans for entities that occur in the `redact` list. 
Text in the `keep` list should remain visible as removing it counts as a case of over-redaction.

## 2. Parameter values in code notebooks
Use the following parameter values in code notebooks:

```python
MATCH_THRESHOLD = 1.0
BETA = 2.0  # F2 favors recall.
```

`MATCH_THRESHOLD = 1.0` ensures that a gold PII span counts as detected only when predictions cover **all of its evaluated non-whitespace characters**. Partial coverage does not count as successful span detection. Whitespace is ignored; other evaluated characters, including punctuation, are counted.

Note: `MATCH_THRESHOLD` specifies how predictions are scored. It is separate from a model's confidence threshold.

## 3. Three headline metrics
The three headline evaluation metrics are:

| Metric | Definition | Formula |
| --- | --- | --- |
| **Span recall** | Share of evaluated annotated PII spans that are completely covered. | `span_recall = fully_covered_pii_span_count / evaluated_pii_span_count` |
| **Span precision** | Share of evaluated predicted spans whose non-whitespace characters fall entirely within PII requiring redaction. | `span_precision = pii_only_predicted_span_count / evaluated_predicted_span_count` |
| **Span F2** | Combines span precision and recall, favoring recall. | `span_f2 = (5 * span_precision * span_recall) / (4 * span_precision + span_recall)` |

Note: Use span precision and recall as fractions from 0 to 1 when calculating F2. Use descriptive variable names throughout calculations, such as `span_precision` rather than `p`.

Note 2: These are coverage-based span metrics. For example, consider the original text: `John Chung arrived.`, where only `John Chung` requires redaction. The sample outputs use `[PERSON_1]` as an illustrative pseudonym token. Quotation marks below show the boundaries of the flagged text.

| Flagged text | Sample output | Span recall | Span precision |
| --- | --- | --- | --- |
| `"John"` | `[PERSON_1] Chung arrived.` | Fails: part of the name remains exposed. | Passes: all flagged characters are PII. |
| `"John Chung "` (including the trailing space) | `[PERSON_1]arrived.` | Passes: the full name is covered. | Passes: extra whitespace is ignored. |
| `"John Chung arrived"` | `[PERSON_1].` | Passes: the full name is covered. | Fails: `arrived` was unnecessarily removed. |

Score the flagged characters in the original text, not the characters in the replacement token.

## 4. Three development metrics
The three development metrics are:

| Metric | Definition | Formula |
| --- | --- | --- |
| **Character over-redaction percentage** | Percentage of evaluated characters selected for removal that did not require redaction. | `character_over_redaction_percentage = 100 * flagged_non_pii_character_count / flagged_character_count` |
| **Character leakage percentage** | Percentage of annotated PII characters left uncovered. | `character_leakage_percentage = 100 * uncovered_pii_character_count / annotated_pii_character_count` |
| **Document leakage rate** | Percentage of PII-containing documents with at least one incompletely covered PII span. | `document_leakage_rate = 100 * leaking_pii_document_count / pii_document_count` |

Note: These metrics are reported as percentages from 0 to 100. For character metrics, count non-whitespace characters only, with overlapping coverage counted once.

Note 2: Character over-redaction and leakage can also be calculated from character precision and recall. Use precision and recall as fractions from 0 to 1 in the formulas below:

```text
character_precision = flagged_pii_character_count / flagged_character_count
character_recall = flagged_pii_character_count / annotated_pii_character_count
character_over_redaction_percentage = (1 - character_precision) * 100
character_leakage_percentage = (1 - character_recall) * 100
```

Note 3: As defined in the formula above, the over-redaction denominator is **characters flagged**, not all non-PII characters in the corpus.

### On document leakage rate

A document leaks if any evaluated non-whitespace character of its annotated PII remains uncovered.

Note: High span recall can occur alongside high document leakage. For example, if every document contains 10 PII spans and one is missed in each, span recall is 90% while document leakage stays 100%.

The local LLM layer can be used to identify PII missed by the first GLiNER layer so that additional corrections can be made. We will evaluate whether those corrections reduce document leakage and how they affect over-redaction.
