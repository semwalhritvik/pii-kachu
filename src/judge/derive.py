"""Derive a case verdict from purposes.yaml classes and the case's span actions."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ("keep", "generalize", "redact", "pseudonymize")
PRECEDENCE = ("UNDER", "SHOULD_PSEUDONYMIZE", "OVER", "OK")
TABLE = {
    "keep": {"keep": "OK", "generalize": "OVER", "redact": "OVER", "pseudonymize": "OVER"},
    "generalize": {"keep": "UNDER", "generalize": "OK", "redact": "OVER", "pseudonymize": "OVER"},
    "redact": {"keep": "UNDER", "generalize": "UNDER", "redact": "OK", "pseudonymize": "OK"},
    "pseudonymize": {"keep": "UNDER", "generalize": "UNDER", "redact": "SHOULD_PSEUDONYMIZE", "pseudonymize": "OK"},
}
FAULT = {"UNDER": "under", "OVER": "over", "SHOULD_PSEUDONYMIZE": "should_pseudonymize"}


def load_purposes(path=ROOT / "configs/purposes.yaml"):
    return {k: v for k, v in yaml.safe_load(Path(path).read_text()).items() if k[0] == "P" and k[1].isdigit()}


def purpose_class(purposes, purpose, label):
    p = purposes[purpose]
    for a in ACTIONS:
        if label in p.get(a, []):
            return a
    raise KeyError(f"{purpose}: label {label} in no list")


def span_verdict(purposes, purpose, span):
    return TABLE[purpose_class(purposes, purpose, span["gold_label"])][span["action"]]


def judge_spans(case):
    """Spans the judge sees: detected ones only."""
    return [s for s in case["spans"] if s.get("detected", True)]


def derive_verdict(case, purposes):
    """Return {"verdict", "offending_span_ids", "fault_type": {span_id: fault}}."""
    per = {}
    for s in case["spans"]:
        cls = purpose_class(purposes, case["purpose"], s["gold_label"])
        if not s.get("detected", True):
            per[s["id"]] = ("UNDER", "missed") if cls != "keep" else ("OK", None)
        else:
            v = TABLE[cls][s["action"]]
            per[s["id"]] = (v, FAULT.get(v))
    act = {s["id"]: s["action"] for s in case["spans"]}
    for group in case.get("quasi_groups", []):
        if any(act[i] in ("redact", "generalize") and case_detected(case, i) for i in group):
            for i in group:  # mitigated: no per-span OVER for members
                if per[i][0] == "OVER":
                    per[i] = ("OK", None)
        elif all(act[i] == "keep" or not case_detected(case, i) for i in group):
            for i in group:
                if per[i][0] == "OK":
                    per[i] = ("UNDER", "quasi")
    faults = {i: f for i, (v, f) in per.items() if v != "OK"}
    verdicts = {v for v, _ in per.values()}
    verdict = next(v for v in PRECEDENCE if v in verdicts or v == "OK")
    return {"verdict": verdict, "offending_span_ids": sorted(faults), "fault_type": faults}


def case_detected(case, span_id):
    return next(s for s in case["spans"] if s["id"] == span_id).get("detected", True)
