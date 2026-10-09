import json
from .labels import map_label, IGNORE

DIR = "data/raw/tab"


def _span(m):
    ent, idt = m["entity_type"], m["identifier_type"]
    return map_label("tab", ent, idt), {"start": m["start_offset"], "end": m["end_offset"],
                                        "attrs": {"entity_type": ent, "identifier_type": idt}}


def load(directory=DIR):
    for split in ("train", "dev", "test"):
        for d in json.load(open(f"{directory}/echr_{split}.json")):
            keys = sorted(d["annotations"])
            spans, ignored, other = [], [], {}
            for i, k in enumerate(keys):
                out = []
                for m in d["annotations"][k]["entity_mentions"]:
                    lab, s = _span(m)
                    s["label"] = lab if lab != IGNORE else m["entity_type"]
                    out.append((lab, s))
                if i == 0:
                    spans = [s for lab, s in out if lab != IGNORE]
                    ignored = [s for lab, s in out if lab == IGNORE]
                else:
                    other[k] = [s for _, s in out]
            yield {"doc_id": str(d["doc_id"]), "source": "tab", "text": d["text"], "spans": spans,
                   "ignored_spans": ignored, "other_annotations": other, "split": split}
