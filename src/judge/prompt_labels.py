"""Prompt for the "labels" judge mode: the LLM only labels spans; rules decide the verdict (no purpose/policy shown)."""
from src.judge.prompt import LABEL_DEFS, span_table

LABELS = list(LABEL_DEFS)
SYSTEM = """You are a privacy auditor. You are given an original text, its redacted version, and a table of the spans a detector found with the action taken on each.

1. Classify each span in the table into exactly one label, using the span's meaning in the original text.
2. List any personal information visible in the ORIGINAL text that is missing from the span table (exact text as written in the original, with its label). Leave the list empty if nothing is missing.

Labels:
{defs}

Answer with JSON: labels (a list of {{id, label}} with ids from the span table), missed (a list of {{text, label}})."""


def output_schema_labels():
    label = {"type": "string", "enum": LABELS}
    props = {
        "labels": {"type": "array", "items": {"type": "object", "required": ["id", "label"],
                                              "properties": {"id": {"type": "string"}, "label": label}}},
        "missed": {"type": "array", "items": {"type": "object", "required": ["text", "label"],
                                              "properties": {"text": {"type": "string"}, "label": label}}},
    }
    return {"type": "object", "properties": props, "required": list(props)}


def system_labels():
    return SYSTEM.format(defs="\n".join(f"- {k}: {v}" for k, v in LABEL_DEFS.items()))


def build_messages_labels(case):
    user = (
        f"ORIGINAL TEXT:\n{case['original']}\n\n"
        f"REDACTED TEXT:\n{case['redacted']}\n\n"
        f"SPAN TABLE (spans found by the detector):\n{span_table(case)}"
    )
    return [{"role": "system", "content": system_labels()}, {"role": "user", "content": user}]
