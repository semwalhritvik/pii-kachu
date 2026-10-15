"""Build chat messages and the output schema for the over/under-redaction judge."""
from src.judge.derive import ACTIONS, load_purposes

VERDICTS = ["OK", "OVER", "UNDER", "SHOULD_PSEUDONYMIZE"]
ACTION_WORD = {"keep": "KEPT", "redact": "REDACTED", "pseudonymize": "PSEUDONYMIZED", "generalize": "GENERALIZED"}
LABEL_DEFS = {
    "DOB": "a person's date of birth",
    "DATE_OTHER": "any other date",
    "PERSON_PUBLIC": "a well-known public or historical figure",
    "PERSON_PRIVATE": "any other person",
    "STREET_ADDRESS": "a home-level address",
    "LOCATION_OTHER": "city, region, venue or non-home place",
    "DEMOGRAPHIC": "age, gender or similar",
    "ORG": "an organization",
    "ID_NUM": "an identification number",
    "EMAIL": "an email address",
    "PHONE": "a phone number",
    "URL_PERSONAL": "a personal URL",
    "OTHER_PII": "any other personally identifying detail",
}
SYSTEM = f"""You are a privacy-redaction auditor. You are given a purpose policy, an original text, its redacted version, and a table of the spans the detector found with the action taken on each. Decide whether the redaction matches the purpose policy.

Verdicts:
- OK: the redaction matches the purpose policy.
- OVER: something the purpose needs was removed or generalized too far.
- UNDER: something that should be removed or generalized was left, including PII missing from the span table and quasi-identifier groups left fully intact.
- SHOULD_PSEUDONYMIZE: a stable linkage key was redacted but should be consistently pseudonymized.
Precedence when several apply: UNDER > SHOULD_PSEUDONYMIZE > OVER > OK.

Quasi rule: if every span of a combination of rare event details stays unmasked, it is UNDER.

Answer with JSON: verdict, offending_span_ids (ids from the span table; empty if none), missed_text (exact text left unmasked that is not in the span table; empty if none)."""


def label_defs_text():
    return "\n".join(f"- {k}: {v}" for k, v in LABEL_DEFS.items())


def render_policy(purposes, purpose):
    p = purposes[purpose]
    lines = [f"Purpose {purpose} policy:"]
    for a in ACTIONS:
        lines.append(f"- {a}: {', '.join(p.get(a) or []) or '(none)'}")
    return "\n".join(lines)


def span_table(case):
    rows = ["id | text | detector_label | action"]
    for s in case["spans"]:
        if s.get("detected", True) is False:
            continue
        rows.append(f"{s['id']} | {s['text']} | {s['detector_label']} | {ACTION_WORD[s['action']]}")
    return "\n".join(rows)


def output_schema(with_reason=False):
    props = {}
    if with_reason:
        props["reason"] = {"type": "string"}
    props["verdict"] = {"type": "string", "enum": VERDICTS}
    props["offending_span_ids"] = {"type": "array", "items": {"type": "string"}}
    props["missed_text"] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props)}


def build_messages(case, purposes=None, with_reason=False):
    purposes = purposes or load_purposes()
    system = f"{SYSTEM}\n\nLabel definitions:\n{label_defs_text()}"
    if with_reason:
        system += "\n\nGive a short `reason` first, then the verdict fields."
    user = (
        f"{render_policy(purposes, case['purpose'])}\n\n"
        f"Purpose id: {case['purpose']}\n\n"
        f"ORIGINAL TEXT:\n{case['original']}\n\n"
        f"REDACTED TEXT:\n{case['redacted']}\n\n"
        f"SPAN TABLE (spans found by the detector):\n{span_table(case)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
