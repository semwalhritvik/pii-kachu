"""Validate the pilot set: schema, span offsets, derived verdicts, rendered text; print purpose x verdict counts and pair contrasts."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.judge.derive import ACTIONS, PRECEDENCE, derive_verdict, load_purposes  # noqa: E402
from src.judge.render import DETECTOR, render  # noqa: E402

ROLES = {"student", "parent", "staff", "public_figure", "other"}
CASE_KEYS = {"case_id", "purpose", "format", "hard_case_type", "original", "spans", "gold_verdict", "offending_span_ids", "fault_type", "redacted"}
OPT_CASE_KEYS = {"pair_id", "quasi_groups"}
SPAN_KEYS = {"id", "text", "gold_label", "action", "start", "end", "role", "detector_label"}
OPT_SPAN_KEYS = {"replacement", "entity", "detected"}
GOLD_ONLY = ("PERSON_PUBLIC", "PERSON_PRIVATE", "DOB", "DATE_OTHER", "LOCATION_OTHER", "DEMOGRAPHIC", "STREET_ADDRESS", "OTHER_PII")


def main(path):
    purposes = load_purposes()
    labels = set(yaml.safe_load((ROOT / "configs/labels.yaml").read_text())["unified"])
    cases = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    errs, seen = [], set()
    for c in cases:
        cid = c.get("case_id", "?")
        if not CASE_KEYS <= set(c) or set(c) - CASE_KEYS - OPT_CASE_KEYS:
            errs.append(f"{cid}: keys {sorted(set(c) ^ CASE_KEYS)}")
            continue
        if cid in seen:
            errs.append(f"{cid}: duplicate id")
        seen.add(cid)
        if c["purpose"] not in purposes:
            errs.append(f"{cid}: purpose {c['purpose']}")
            continue
        sids = set()
        for s in c["spans"]:
            if not SPAN_KEYS <= set(s) or set(s) - SPAN_KEYS - OPT_SPAN_KEYS:
                errs.append(f"{cid}: span keys {sorted(set(s) ^ SPAN_KEYS)}")
                continue
            if s["id"] in sids:
                errs.append(f"{cid}: duplicate span id {s['id']}")
            sids.add(s["id"])
            if s["text"] not in c["original"] or c["original"][s["start"]:s["end"]] != s["text"]:
                errs.append(f"{cid}/{s['id']}: text/offsets not in original")
            if s["gold_label"] not in labels:
                errs.append(f"{cid}/{s['id']}: label {s['gold_label']}")
            if s["action"] not in ACTIONS:
                errs.append(f"{cid}/{s['id']}: action {s['action']}")
            if s["action"] == "generalize" and not s.get("replacement"):
                errs.append(f"{cid}/{s['id']}: generalize without replacement")
            if s["role"] not in ROLES:
                errs.append(f"{cid}/{s['id']}: role {s['role']}")
            if (s["gold_label"] == "PERSON_PUBLIC") != (s["role"] == "public_figure"):
                errs.append(f"{cid}/{s['id']}: role {s['role']} vs {s['gold_label']}")
            if s["gold_label"] not in ("PERSON_PRIVATE", "PERSON_PUBLIC") and s["role"] != "other":
                errs.append(f"{cid}/{s['id']}: role on non-person")
            if s.get("detected", True) is False and s["action"] != "keep":
                errs.append(f"{cid}/{s['id']}: undetected span must have action keep")
            if s["detector_label"] != DETECTOR.get(s["gold_label"]):
                errs.append(f"{cid}/{s['id']}: detector_label {s['detector_label']}")
        spans = sorted(c["spans"], key=lambda s: s.get("start", 0))
        if any(a["end"] > b["start"] for a, b in zip(spans, spans[1:]) if "end" in a and "start" in b):
            errs.append(f"{cid}: overlapping spans")
        if any(i not in sids for g in c.get("quasi_groups", []) for i in g):
            errs.append(f"{cid}: quasi_groups ids")
        try:
            d = derive_verdict(c, purposes)
            if (d["verdict"], d["offending_span_ids"], d["fault_type"]) != (c["gold_verdict"], c["offending_span_ids"], c["fault_type"]):
                errs.append(f"{cid}: gold {c['gold_verdict']} != derived {d['verdict']}")
            if any(g in c["redacted"] and g not in c["original"] for g in GOLD_ONLY):  # "DOB" can be source text
                errs.append(f"{cid}: gold-only label in redacted")
            if render(c) != c["redacted"]:
                errs.append(f"{cid}: redacted does not match actions")
        except (KeyError, ValueError) as e:
            errs.append(f"{cid}: {e!r}")
    print(f"cases={len(cases)} schema_errors={len(errs)}")
    for e in errs:
        print("  ERR", e)
    if errs:
        sys.exit(1)
    n = len(cases)
    vc = Counter(c["gold_verdict"] for c in cases)
    print("verdict:", {v: f"{vc[v]} ({100 * vc[v] / n:.0f}%)" for v in PRECEDENCE})
    tab = defaultdict(Counter)
    for c in cases:
        tab[c["purpose"][:2]][c["gold_verdict"]] += 1
    print("purpose  " + " ".join(f"{v[:5]:>5}" for v in PRECEDENCE))
    for p in sorted(tab):
        print(f"{p:<8} " + " ".join(f"{tab[p][v]:>5}" for v in PRECEDENCE))
    pairs = defaultdict(list)
    for c in cases:
        if "pair_id" in c:
            pairs[c["pair_id"]].append(c)
    print(f"pairs={len(pairs)}")
    for pid in sorted(pairs):
        a, b = pairs[pid]
        same = a["original"] == b["original"] and a["redacted"] == b["redacted"] and a["purpose"] != b["purpose"]
        print(f"  {pid}: {a['purpose'][:2]}={a['gold_verdict']} vs {b['purpose'][:2]}={b['gold_verdict']} {'' if same and a['gold_verdict'] != b['gold_verdict'] else 'BAD'}")
    print("P4 SHOULD_PSEUDONYMIZE:", sum(c["purpose"][:2] == "P4" and c["gold_verdict"] == "SHOULD_PSEUDONYMIZE" for c in cases))
    print("missed:", sorted(c["case_id"][-3:] for c in cases if any(f == "missed" for f in c["fault_type"].values())))
    print("quasi:", [(c["case_id"][-3:], c["gold_verdict"]) for c in cases if c.get("quasi_groups")])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "pilot/pilot_cases.jsonl")
