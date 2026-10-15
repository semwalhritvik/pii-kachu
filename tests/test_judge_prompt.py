import json

from src.judge.prompt import build_messages, output_schema

CASE = {
    "case_id": "c1", "purpose": "P3_longitudinal", "format": "x", "original": "Ann met Bob on 1 May.",
    "redacted": "<PERSON_1> met Bob on 1 May.", "hard_case_type": "HCT_ZZ", "pair_id": "PAIR_ZZ",
    "gold_verdict": "SHOULD_PSEUDONYMIZE", "fault_type": {"s1": "FAULT_ZZ"}, "offending_span_ids": ["s1"],
    "spans": [
        {"id": "s1", "text": "Ann", "gold_label": "GOLD_ZZ", "action": "redact", "role": "ROLE_ZZ",
         "detector_label": "PERSON", "entity": "ENTITY_ZZ", "detected": True},
        {"id": "s9", "text": "SECRETMISSED", "gold_label": "ID_NUM", "action": "keep", "role": "other",
         "detector_label": "ID", "detected": False},
    ],
}


def prompt_text(**kw):
    return "\n".join(m["content"] for m in build_messages(CASE, **kw))


def test_no_forbidden_leak():
    t = prompt_text()
    for bad in ["GOLD_ZZ", "ROLE_ZZ", "ENTITY_ZZ", "PAIR_ZZ", "HCT_ZZ", "FAULT_ZZ", "gold_label", "gold_verdict",
                "pair_id", "hard_case_type", "fault_type", "detected", "entity", "role:"]:
        assert bad not in t, bad
    assert "SECRETMISSED" not in t and "s9" not in t
    assert "s1 | Ann | PERSON | REDACTED" in t


def test_policy_and_schema():
    t = prompt_text()
    assert "pseudonymize: PERSON_PRIVATE, ID_NUM" in t and "UNDER > SHOULD_PSEUDONYMIZE > OVER > OK" in t
    assert list(output_schema(True)["properties"])[0] == "reason"
    assert "reason" not in output_schema()["properties"]
    json.dumps(output_schema(True))
