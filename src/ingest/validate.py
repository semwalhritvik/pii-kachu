import json
import pathlib
import sys

import pandas as pd
import yaml

from .labels import UNIFIED

SPLITS = ("train", "dev", "test")
KNOWN = pathlib.Path(__file__).parents[2] / "configs" / "known_issues.yaml"


def load_allowlist(path=KNOWN):
    cfg = yaml.safe_load(open(path)) or {}
    return {(e["source"], str(e["doc_id"]), tuple(sorted(map(tuple, e["spans"]))))
            for e in cfg.get("crossing_spans", [])}


def is_allowlisted(d, a, b, allow):
    key = (d["source"], d["doc_id"], tuple(sorted([(a["start"], a["end"]), (b["start"], b["end"])])))
    return key in allow


def pair_stats(d):
    """Return (nested, crossing, crossing_examples). Nested = overlapping but not crossing."""
    spans = sorted(d["spans"], key=lambda s: (s["start"], -s["end"]))
    nested, crossing, ex = 0, 0, []
    for i, a in enumerate(spans):
        for b in spans[i + 1:]:
            if b["start"] >= a["end"]:
                break
            if a["start"] < b["start"] < a["end"] < b["end"]:
                crossing += 1
                ex.append((a, b))
            else:
                nested += 1
    return nested, crossing, ex


def check_doc(d, allow=frozenset(), allowed=None):
    errs = []
    for s in d["spans"]:
        if s["label"] not in UNIFIED:
            errs.append(f"{d['doc_id']}: bad label {s['label']}")
        if not (0 <= s["start"] < s["end"] <= len(d["text"])):
            errs.append(f"{d['doc_id']}: bad offsets {s['start']}-{s['end']}")
    for a, b in pair_stats(d)[2]:
        if is_allowlisted(d, a, b, allow):
            if allowed is not None:
                allowed.append((d["doc_id"], a, b))
            continue
        errs.append(f"{d['doc_id']}: crossing {a['start']}-{a['end']} {a['label']} / {b['start']}-{b['end']} {b['label']}")
    return errs


def check_splits(by_split):
    seen, errs = {}, []
    for sp, docs in by_split.items():
        for d in docs:
            if d["doc_id"] in seen and seen[d["doc_id"]] != sp:
                errs.append(f"doc_id {d['doc_id']} in {seen[d['doc_id']]} and {sp}")
            seen[d["doc_id"]] = sp
    return errs


def load(directory="data/processed"):
    return {s: [json.loads(l) for l in open(pathlib.Path(directory) / f"{s}.jsonl")] for s in SPLITS}


def main():
    by_split = load()
    errs = check_splits(by_split)
    allow, allowed = load_allowlist(), []
    rows = []
    for sp, docs in by_split.items():
        for d in docs:
            n0 = len(allowed)
            errs += check_doc(d, allow, allowed)
            nested, crossing, _ = pair_stats(d)
            n_allow = len(allowed) - n0
            row = {"source": d["source"], "split": sp, "docs": 1, "spans": len(d["spans"]),
                   "ignored": len(d.get("ignored_spans", [])), "multi_annot": int(bool(d.get("other_annotations"))),
                   "nested_pairs": nested, "crossing_pairs": crossing - n_allow, "allowlisted": n_allow}
            for s in d["spans"]:
                row[s["label"]] = row.get(s["label"], 0) + 1
            rows.append(row)
    df = pd.DataFrame(rows).fillna(0).groupby(["source", "split"]).sum().astype(int)
    print(df.T.to_string())
    print(f"\nfailures: {len(errs)}")
    for e in errs[:20]:
        print(" ", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
