"""Non-LLM baselines. Each returns {"verdict", "offending_span_ids", "missed_text"} for a case."""
from collections import Counter

from src.judge.derive import derive_verdict, load_purposes

DETECTOR_TO_LABEL = {
    "PERSON": "PERSON_PRIVATE", "DATE": "DATE_OTHER", "LOCATION": "LOCATION_OTHER", "ID": "ID_NUM",
    "EMAIL": "EMAIL", "PHONE": "PHONE", "ORG": "ORG", "URL": "URL_PERSONAL", "USERNAME": "OTHER_PII",
    "OTHER": "DEMOGRAPHIC",
}


def _rule(case, purposes, use_quasi):
    """Derive the verdict from detector labels alone (gold_label/role ignored); undetected spans are absent."""
    purposes = purposes or load_purposes()
    spans = [dict(s, gold_label=DETECTOR_TO_LABEL[s["detector_label"]])
             for s in case["spans"] if s.get("detected", True)]
    ids = {s["id"] for s in spans}
    groups = [g for g in ([i for i in g if i in ids] for g in case.get("quasi_groups", []) if use_quasi) if g]
    d = derive_verdict(dict(case, spans=spans, quasi_groups=groups), purposes)
    return {"verdict": d["verdict"], "offending_span_ids": d["offending_span_ids"], "missed_text": []}


def rule_oracle(case, purposes=None):
    """Rule baseline that also reads the gold-derived quasi_groups (upper bound for rules)."""
    return _rule(case, purposes, True)


def rule_blind(case, purposes=None):
    """Same logic as rule_oracle without quasi_groups: the honest non-LLM baseline."""
    return _rule(case, purposes, False)


def always_ok(case):
    return {"verdict": "OK", "offending_span_ids": [], "missed_text": []}


def majority_class(train_cases):
    """Predictor for the most common gold verdict of train_cases (use dev)."""
    top = Counter(c["gold_verdict"] for c in train_cases).most_common(1)[0][0]
    return lambda case: {"verdict": top, "offending_span_ids": [], "missed_text": []}
