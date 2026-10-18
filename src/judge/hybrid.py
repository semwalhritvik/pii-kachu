"""Hybrid judge: the LLM labels spans, derive_verdict decides (no quasi_groups)."""
from src.judge.baselines import DETECTOR_TO_LABEL
from src.judge.derive import derive_verdict, load_purposes
from src.judge.prompt_labels import LABELS


def predicted_labels(parsed):
    """{span id: label} from a parsed labels-mode answer; invalid entries are dropped."""
    items = parsed.get("labels") if isinstance(parsed, dict) else None
    return {x["id"]: x["label"] for x in items if isinstance(x, dict) and isinstance(x.get("id"), str)
            and x.get("label") in LABELS} if isinstance(items, list) else {}


def predicted_missed(case, parsed):
    """Valid, de-duplicated [(text, label)] missed entries whose text occurs in the original."""
    items = parsed.get("missed") if isinstance(parsed, dict) else None
    out = []
    for x in items if isinstance(items, list) else []:
        if isinstance(x, dict) and isinstance(x.get("text"), str) and x["text"].strip() and x.get("label") in LABELS \
                and x["text"] in case["original"] and (x["text"], x["label"]) not in out:
            out.append((x["text"], x["label"]))
    return out


def effective_missed(case, parsed):
    """predicted_missed entries that still occur in the redacted text and are not a detected span's text."""
    seen = {s["text"].strip().casefold() for s in case["spans"] if s.get("detected", True)}
    return [(t, lab) for t, lab in predicted_missed(case, parsed)
            if t in case["redacted"] and t.strip().casefold() not in seen]


def _verdict(case, spans, extra, purposes):
    """derive_verdict over `spans` + keep-action `extra` [(text, label)]; missed ids map back to missed_text."""
    extra_spans = [{"id": f"__missed{i}", "text": t, "gold_label": lab, "action": "keep", "detected": True}
                   for i, (t, lab) in enumerate(extra, 1)]
    d = derive_verdict(dict(case, spans=spans + extra_spans, quasi_groups=[]), purposes or load_purposes())
    ids, mids = d["offending_span_ids"], {s["id"] for s in extra_spans}
    return {"verdict": d["verdict"], "offending_span_ids": [i for i in ids if i not in mids],
            "missed_text": [s["text"] for s in extra_spans if s["id"] in ids]}


def hybrid_verdict(case, parsed, purposes=None):
    labels = predicted_labels(parsed)
    spans = [dict(s, gold_label=labels.get(s["id"]) or DETECTOR_TO_LABEL[s["detector_label"]])
             for s in case["spans"] if s.get("detected", True)]
    return _verdict(case, spans, effective_missed(case, parsed), purposes)


def labels_oracle(case, purposes=None):
    """Ceiling: gold labels on detected spans, gold undetected spans added as keep, no quasi_groups."""
    spans = [s for s in case["spans"] if s.get("detected", True)]
    missed = [(s["text"], s["gold_label"]) for s in case["spans"] if not s.get("detected", True)]
    return _verdict(case, spans, missed, purposes)
