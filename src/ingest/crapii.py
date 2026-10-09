import json
from .labels import map_label

PATH = "data/raw/crapii/crapii.json"


def _spans(doc):
    text, cursor, spans, cur, rebuilt = doc["full_text"], 0, [], None, []
    for tok, lab, ws in zip(doc["tokens"], doc["labels"], doc["trailing_whitespace"]):
        start, end = cursor, cursor + len(tok)
        rebuilt.append(tok + (" " if ws else ""))
        cursor = end + (1 if ws else 0)
        if lab == "O":
            cur = None
            continue
        pre, ent = lab.split("-", 1)
        if cur is not None and pre == "I" and cur["ent"] == ent:
            cur["end"] = end
        else:
            cur = {"ent": ent, "start": start, "end": end}
            spans.append(cur)
    assert "".join(rebuilt) == text, f"text mismatch in doc {doc['document']}"
    return [{"start": s["start"], "end": s["end"], "label": map_label("crapii", s["ent"]), "attrs": {}} for s in spans]


def load(path=PATH):
    for d in json.load(open(path)):
        yield {"doc_id": str(d["document"]), "source": "crapii", "text": d["full_text"], "spans": _spans(d), "split": None}
