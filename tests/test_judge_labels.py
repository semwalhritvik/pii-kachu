import json

from src.judge.derive import load_purposes
from src.judge.hybrid import hybrid_verdict, labels_oracle
from src.judge.prompt_labels import LABELS, build_messages_labels, output_schema_labels
from tests.test_judge_prompt import CASE

P = load_purposes()
PURPOSE = CASE["purpose"]


def text():
    return "\n".join(m["content"] for m in build_messages_labels(CASE))


def test_no_forbidden_leak():
    t = text()
    for bad in ["GOLD_ZZ", "ROLE_ZZ", "ENTITY_ZZ", "PAIR_ZZ", "HCT_ZZ", "FAULT_ZZ", "gold_label", "gold_verdict", "pair_id",
                "hard_case_type", "fault_type", "family", "detected", "entity", "role:", "purpose", "policy", PURPOSE,
                "SHOULD_PSEUDONYMIZE", "OVER", "UNDER"]:
        assert bad not in t, bad
    assert "SECRETMISSED" not in t and "s9" not in t
    assert "s1 | Ann | PERSON | REDACTED" in t
    for lab in LABELS:
        assert lab in t


def test_schema():
    sch = output_schema_labels()
    json.dumps(sch)
    assert sch["properties"]["labels"]["items"]["properties"]["label"]["enum"] == LABELS
    assert list(sch["properties"]) == ["labels", "missed"]


def _case(action="keep"):
    return {"case_id": "c", "purpose": PURPOSE, "original": "Ann met SECRET on 1 May.", "redacted": "Ann met SECRET on 1 May.", "quasi_groups": [["s1", "s2"]],
            "spans": [{"id": "s1", "text": "Ann", "gold_label": "PERSON_PRIVATE", "action": action, "detector_label": "PERSON"},
                      {"id": "s2", "text": "1 May", "gold_label": "DATE_OTHER", "action": "keep", "detector_label": "DATE"}]}


def test_hybrid_uses_predicted_label_and_falls_back():
    c = _case("keep")
    coarse = hybrid_verdict(c, None, P)
    assert coarse["verdict"] == hybrid_verdict(c, {"labels": [{"id": "bad", "label": "DOB"}, {"id": "s1", "label": "NOPE"}]}, P)["verdict"]
    pub = hybrid_verdict(c, {"labels": [{"id": "s1", "label": "PERSON_PUBLIC"}], "missed": []}, P)
    assert coarse["verdict"] == "UNDER" and pub["verdict"] == "OK"


def test_missed_only_if_in_original_and_quasi_ignored():
    c = _case("keep")
    r = hybrid_verdict(c, {"labels": [], "missed": [{"text": "NOT THERE", "label": "ID_NUM"}]}, P)
    assert r["missed_text"] == []
    r = hybrid_verdict(c, {"labels": [], "missed": [{"text": "SECRET", "label": "ID_NUM"}]}, P)
    assert r["verdict"] == "UNDER" and r["missed_text"] == ["SECRET"] and "__missed1" not in r["offending_span_ids"]


def test_labels_oracle_adds_undetected_gold():
    c = _case("redact")
    c["spans"].append({"id": "s3", "text": "SECRET", "gold_label": "ID_NUM", "action": "keep", "detector_label": "ID", "detected": False})
    r = labels_oracle(c, P)
    assert r["missed_text"] == ["SECRET"] and r["verdict"] == "UNDER"


def test_already_redacted_missed_entry_changes_nothing():
    c = _case("redact")
    c["redacted"] = "<PERSON_1> met SECRET on 1 May."
    base = hybrid_verdict(c, {"labels": [], "missed": []}, P)
    r = hybrid_verdict(c, {"labels": [], "missed": [{"text": "Ann", "label": "PERSON_PRIVATE"}]}, P)
    assert r == base and r["missed_text"] == []
    # a text that equals a detected span's text is ignored even if it still appears in the redacted text
    c2 = _case("keep")
    r2 = hybrid_verdict(c2, {"labels": [], "missed": [{"text": "1 May", "label": "DOB"}]}, P)
    assert r2 == hybrid_verdict(c2, None, P)
