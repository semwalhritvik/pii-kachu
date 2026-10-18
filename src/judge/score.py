"""Score bake-off results and baselines. Usage: python -m src.judge.score [--split dev|test|both]

Invalid JSON / out-of-enum verdicts count as wrong. Offending-span F1 is micro-pooled over cases with gold
verdict != OK; a gold span the judge cannot see (detected=false) counts as found when missed_text names it.
"""
import argparse
import json
import random
import statistics
from math import comb
from collections import defaultdict

from src.judge.baselines import DETECTOR_TO_LABEL, always_ok, majority_class, rule_blind, rule_oracle
from src.judge.derive import ROOT, load_purposes
from src.judge.hybrid import hybrid_verdict, labels_oracle, predicted_labels, predicted_missed
from src.judge.prompt_labels import LABELS
from src.judge.prompt import VERDICTS
from src.judge.splits import load_cases, load_splits

OUT = ROOT / "results/bakeoff"
CLASSES = ["OK", "OVER", "UNDER", "SHOULD_PSEUDONYMIZE"]


def load_results(split_ids, labels=False):
    """model -> {case_id: last row} restricted to split_ids; merges <model>_dev/_test/_all files.
    labels=True reads only <model>-labels files (model name without the suffix); False skips them."""
    res = defaultdict(dict)
    for p in sorted(OUT.glob("*_*.jsonl"), key=lambda p: p.stat().st_mtime):  # later files/lines win
        model = p.stem.rsplit("_", 1)[0]
        if model.endswith("-labels") != labels:
            continue
        model = model.removesuffix("-labels")
        for r in map(json.loads, p.read_text().splitlines()):
            if r["case_id"] in split_ids:
                res[model][r["case_id"]] = r
    return res


def prediction(row):
    """Normalise a result row to {"verdict" (None if invalid), "offending_span_ids", "missed_text"}."""
    p = row.get("parsed") if row else None
    if not (row and row["valid_json"] and isinstance(p, dict) and p.get("verdict") in VERDICTS):
        return {"verdict": None, "offending_span_ids": [], "missed_text": []}
    ids, miss = p.get("offending_span_ids"), p.get("missed_text")
    return {"verdict": p["verdict"], "offending_span_ids": [i for i in ids if isinstance(i, str)] if isinstance(ids, list) else [],
            "missed_text": [m for m in miss if isinstance(m, str)] if isinstance(miss, list) else []}


def predicted_span_set(case, pred):
    out = set(pred["offending_span_ids"])
    for s in case["spans"]:
        if s.get("detected", True) is False and any(
                m.strip() and (m.strip().casefold() in s["text"].casefold() or s["text"].casefold() in m.strip().casefold())
                for m in pred["missed_text"]):
            out.add(s["id"])
    return out


def macro_f1(golds, preds):
    f1s = []
    for k in CLASSES:
        tp = sum(g == k and p == k for g, p in zip(golds, preds))
        fp = sum(g != k and p == k for g, p in zip(golds, preds))
        fn = sum(g == k and p != k for g, p in zip(golds, preds))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def bootstrap_ci(golds, ps, n=1000, seed=42):
    """95% percentile bootstrap CIs over cases for accuracy and macro-F1."""
    rng, accs, f1s = random.Random(seed), [], []
    for _ in range(n):
        idx = [rng.randrange(len(golds)) for _ in golds]
        g, p = [golds[i] for i in idx], [ps[i] for i in idx]
        accs.append(sum(a == b for a, b in zip(g, p)) / len(g))
        f1s.append(macro_f1(g, p))
    q = lambda xs: (sorted(xs)[int(0.025 * n)], sorted(xs)[int(0.975 * n) - 1])
    return q(accs), q(f1s)


def mcnemar_exact(a_ok, b_ok):
    """Two-sided exact McNemar p-value for paired correctness vectors; returns (p, b_only, a_only)."""
    b = sum(x and not y for x, y in zip(a_ok, b_ok))
    c = sum(y and not x for x, y in zip(a_ok, b_ok))
    n = b + c
    if n == 0:
        return 1.0, b, c
    p = 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, p), b, c


def score(cases, preds, rows=None):
    """cases: list; preds: {case_id: pred}; rows: optional {case_id: result row} for timing."""
    golds = [c["gold_verdict"] for c in cases]
    ps = [preds[c["case_id"]]["verdict"] if c["case_id"] in preds else None for c in cases]
    ok = [g == p for g, p in zip(golds, ps)]
    conf = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for g, p in zip(golds, ps):
        if p:
            conf[g][p] += 1
    tp = fp = fn = 0
    for c in cases:
        if c["gold_verdict"] == "OK":
            continue
        gold = set(c["offending_span_ids"])
        pr = predicted_span_set(c, preds[c["case_id"]]) if c["case_id"] in preds else set()
        tp, fp, fn = tp + len(gold & pr), fp + len(pr - gold), fn + len(gold - pr)
    acc_by = lambda key: {k: (sum(o for o, c in zip(ok, cases) if c[key] == k) / sum(c[key] == k for c in cases),
                              sum(c[key] == k for c in cases)) for k in sorted({c[key] for c in cases})}
    pairs = defaultdict(list)
    for o, c in zip(ok, cases):
        if c.get("pair_id") is not None:
            pairs[c["pair_id"]].append(o)
    ff = [bool(predicted_span_set(c, preds[c["case_id"]])) for c in cases if c["gold_verdict"] == "OK" and c["case_id"] in preds]
    acc_ci, f1_ci = bootstrap_ci(golds, ps)
    r = {"n": len(cases), "ok": ok, "acc_ci": acc_ci, "f1_ci": f1_ci, "n_pairs": len(pairs),
         "false_flag": sum(ff) / len(ff) if ff else float("nan"), "n_gold_ok": sum(c["gold_verdict"] == "OK" for c in cases),
         "done": sum(c["case_id"] in preds for c in cases), "acc": sum(ok) / len(cases),
         "macro_f1": macro_f1(golds, ps), "conf": conf,
         "acc_valid": (lambda v: sum(o for o, p in zip(ok, ps) if p) / v if v else float("nan"))(sum(bool(p) for p in ps)),
         "n_valid": sum(bool(p) for p in ps),
         "n_trunc": sum(x.get("error") == "truncated" for x in (rows or {}).values()),
         "n_timeout": sum(x.get("error") == "timeout" for x in (rows or {}).values()),
         "valid": sum(bool(p) for p in ps) / len(cases) if rows is not None else 1.0,
         "span_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
         "by_purpose": acc_by("purpose"), "by_family": acc_by("family"), "by_hard": acc_by("hard_case_type"),
         "pair": sum(all(v) for v in pairs.values()) / len(pairs) if pairs else float("nan"),
         "lat": None, "tps": None}
    if rows:
        lat = [x["latency_s"] for x in rows.values() if x["latency_s"] is not None]
        tps = [x["tokens_per_s"] for x in rows.values() if x["tokens_per_s"]]
        r["lat"], r["tps"] = (statistics.median(lat) if lat else None), (statistics.median(tps) if tps else None)
    return r


def fmt(x, pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.1f}" if pct else f"{x:.2f}"


def cell(v):
    """'acc% (n=N)'; n is always shown, and a missing cell is an explicit n=0."""
    return f"{v[0] * 100:.1f} (n={v[1]})" if v else "- (n=0)"


def table(results, key, title):
    keys = sorted({k for r in results.values() for k in r[key]})
    out = [f"## {title}", "", "| model | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    out += [f"| {n} | " + " | ".join(cell(r[key].get(k)) for k in keys) + " |" for n, r in results.items()]
    return out + [""]


def write(split, systems, results, suffix):
    cols = ["model", "done", "acc [95% CI] (invalid=wrong)", "acc valid-only (n)", "macro-F1 [95% CI]", "valid% (trunc/timeout)", "span-F1", "false-flag%", "pair-cons (pairs)",
            "McNemar p vs rule_blind (+/-)", "med lat s", "med tok/s"]
    lines = [f"# Bake-off summary ({split} split)", "",
             "Truncated/timeout cases have no valid answer and count as wrong in acc/macro-F1; valid-only accuracy excludes them. "
             "McNemar: exact two-sided, paired on case ids vs rule_blind; (+/-) = cases only the model / only rule_blind got right. "
             "false-flag% = gold-OK cases where any offending id is predicted.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, r in results.items():
        mc = "-"
        if name != "rule_blind" and "rule_blind" in results:
            p, b, c = mcnemar_exact(r["ok"], results["rule_blind"]["ok"])
            mc = f"{p:.3f} ({b}/{c})"
        lines.append(f"| {name} | {r['done']}/{r['n']} | {fmt(r['acc'], 1)} [{fmt(r['acc_ci'][0], 1)}, {fmt(r['acc_ci'][1], 1)}] | "
                     f"{fmt(r['acc_valid'], 1)} ({r['n_valid']}) | "
                     f"{r['macro_f1']:.3f} [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}] | "
                     f"{fmt(r['valid'], 1)} ({r['n_trunc']}/{r['n_timeout']}) | {r['span_f1']:.3f} | {fmt(r['false_flag'], 1)} | {fmt(r['pair'], 1)} ({r['n_pairs']}) | {mc} | "
                     f"{fmt(r['lat'])} | {fmt(r['tps'])} |")
    (OUT / f"summary{suffix}.md").write_text("\n".join(lines) + "\n")
    d = [f"# Detail ({split} split)", ""]
    for key, title in (("by_purpose", "Accuracy by purpose"), ("by_family", "Accuracy by family"), ("by_hard", "Accuracy by hard_case_type")):
        d += table(results, key, title)
    d.append("## Confusion matrices (rows gold, cols predicted; invalid excluded)")
    for n, r in results.items():
        d += ["", f"### {n}", "", "| gold \\ pred | " + " | ".join(CLASSES) + " |", "|" + "---|" * 5]
        d += [f"| {g} | " + " | ".join(str(r["conf"][g][p]) for p in CLASSES) + " |" for g in CLASSES]
    (OUT / f"summary_detail{suffix}.md").write_text("\n".join(d) + "\n")


def run_split(split):
    sp = load_splits()
    ids = set(sp[split])
    cases = [c for c in load_cases() if c["case_id"] in ids]
    dev_cases = [c for c in load_cases() if c["case_id"] in set(sp["dev"])]
    baselines = {"rule_oracle": rule_oracle, "rule_blind": rule_blind, "always_OK": always_ok}
    maj = majority_class(dev_cases)
    if any(maj(c)["verdict"] != "OK" for c in cases[:1]):  # keep majority only if it differs from always_OK
        baselines["majority(dev)"] = maj
    results = {}
    for name, fn in baselines.items():
        results[name] = score(cases, {c["case_id"]: fn(c) for c in cases})
    for model, rows in sorted(load_results(ids).items()):
        results[model] = score(cases, {k: prediction(v) for k, v in rows.items()}, rows)
    write(split, None, results, "" if split == "test" else "_dev")
    return results


def run_pooled():
    """Dev+test pooled by-family accuracy (separate file; models' rows are merged across both splits)."""
    sp = load_splits()
    ids = set(sp["dev"]) | set(sp["test"])
    cases = [c for c in load_cases() if c["case_id"] in ids]
    dev_cases = [c for c in cases if c["case_id"] in set(sp["dev"])]
    maj = majority_class(dev_cases)
    baselines = {"rule_oracle": rule_oracle, "rule_blind": rule_blind, "always_OK": always_ok}
    if any(maj(c)["verdict"] != "OK" for c in cases[:1]):
        baselines["majority(dev)"] = maj
    results = {n: score(cases, {c["case_id"]: fn(c) for c in cases}) for n, fn in baselines.items()}
    for model, rows in sorted(load_results(ids).items()):
        results[model] = score(cases, {k: prediction(v) for k, v in rows.items()}, rows)
    lines = [f"# Pooled dev+test (n={len(cases)})", ""] + table(results, "by_family", "Accuracy by family (pooled)")
    (OUT / "summary_family_pooled.md").write_text("\n".join(lines))
    return results


def _match(a, b):
    a, b = a.strip().casefold(), b.strip().casefold()
    return bool(a) and bool(b) and (a in b or b in a)


def label_stats(cases, rows):
    """Span-label accuracy (overall/by gold_label/by family), confusion, missed-span P/R over cases with a row.
    A span with no valid predicted label counts as wrong (column '-' in the confusion matrix)."""
    tot, hit = defaultdict(int), defaultdict(int)
    conf = defaultdict(lambda: defaultdict(int))
    mt = mp = mg = mh = 0  # missed entries, matching entries, gold undetected spans, recalled spans
    for c in cases:
        row = rows[c["case_id"]]
        parsed = row["parsed"] if row["valid_json"] and not row.get("error") else None
        pl, ents = predicted_labels(parsed), predicted_missed(c, parsed)
        for s in c["spans"]:
            if s.get("detected", True):
                p = pl.get(s["id"])
                conf[s["gold_label"]][p or "-"] += 1
                for key in ("all", f"label:{s['gold_label']}", f"family:{c['family']}"):
                    tot[key] += 1
                    hit[key] += p == s["gold_label"]
        gold_missed = [s for s in c["spans"] if not s.get("detected", True)]
        mt += len(ents)
        mp += sum(any(_match(t, s["text"]) for s in gold_missed) for t, _ in ents)
        mg += len(gold_missed)
        mh += sum(any(_match(t, s["text"]) for t, _ in ents) for s in gold_missed)
    return {"tot": tot, "hit": hit, "conf": conf, "missed": (mp, mt, mh, mg)}


def pr(a, b):
    return f"{a / b * 100:.1f} ({a}/{b})" if b else "- (0)"


def run_labels(split):
    """Labels-mode report for one split (dev and test are never mixed)."""
    ids = set(load_splits()[split])
    all_cases = [c for c in load_cases() if c["case_id"] in ids]
    purposes = load_purposes()
    e2e = {n: score(all_cases, {k: prediction(v) for k, v in rows.items()}, rows)
           for n, rows in sorted(load_results(ids).items())}
    best = max(e2e, key=lambda n: e2e[n]["acc"], default=None)
    e2e_preds = {n: {k: prediction(v) for k, v in rows.items()} for n, rows in load_results(ids).items()}
    out = [f"# Labels-mode summary ({split} split)", "",
           "Rows per model are restricted to the cases that model has a labels row for; rule_blind and labels_oracle are "
           "computed on the same cases. McNemar: exact two-sided, paired (+/-) = cases only hybrid / only the other got right. "
           "Span-label accuracy counts a missing or invalid label as wrong. Missed P/R: entries vs gold undetected spans "
           "(substring match). false-flag% = gold-OK cases with any predicted offending span.", ""]
    for model, rows in sorted(load_results(ids, labels=True).items()):
        cases = [c for c in all_cases if c["case_id"] in rows]
        if not cases:
            continue
        bad = {"verdict": None, "offending_span_ids": [], "missed_text": []}  # invalid/truncated/timeout = wrong
        hyb = {c["case_id"]: hybrid_verdict(c, rows[c["case_id"]]["parsed"], purposes) if rows[c["case_id"]]["valid_json"] else bad
               for c in cases}
        res = {"rule_blind": score(cases, {c["case_id"]: rule_blind(c) for c in cases}),
               "labels_oracle": score(cases, {c["case_id"]: labels_oracle(c, purposes) for c in cases}),
               f"hybrid({model})": score(cases, hyb, {c["case_id"]: rows[c["case_id"]] for c in cases})}
        h = res[f"hybrid({model})"]
        cols = ["system", "acc [95% CI]", "macro-F1 [95% CI]", "pair-cons (pairs)", "false-flag%", "McNemar p vs hybrid (+/-)"]
        out += [f"## {model} (n={len(cases)})", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for n, r in res.items():
            mc = "-" if r is h else "%.3f (%d/%d)" % mcnemar_exact(h["ok"], r["ok"])
            out.append(f"| {n} | {fmt(r['acc'], 1)} [{fmt(r['acc_ci'][0], 1)}, {fmt(r['acc_ci'][1], 1)}] | {r['macro_f1']:.3f} "
                       f"[{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}] | {fmt(r['pair'], 1)} ({r['n_pairs']}) | "
                       f"{fmt(r['false_flag'], 1)} | {mc} |")
        out.append("")
        if best:
            sub = [i for i, c in enumerate(cases) if c["case_id"] in e2e_preds[best]]
            eo = [e2e_preds[best][cases[i]["case_id"]]["verdict"] == cases[i]["gold_verdict"] for i in sub]
            p, b, c_ = mcnemar_exact([h["ok"][i] for i in sub], eo)
            out += [f"McNemar hybrid vs best e2e ({best}, acc {e2e[best]['acc'] * 100:.1f} on the full split), "
                    f"{len(sub)} shared cases: p={p:.3f} (hybrid-only {b} / e2e-only {c_})", ""]
        out += [f"Valid JSON {h['n_valid']}/{len(cases)} (truncated {h['n_trunc']}, timeout {h['n_timeout']}). Hybrid accuracy: "
                f"{fmt(h['acc'], 1)} counting those as wrong, {fmt(h['acc_valid'], 1)} over valid answers only (n={h['n_valid']}).", ""]
        st = label_stats(cases, rows)
        mp, mt, mh, mg = st["missed"]
        out += [f"Span-label accuracy overall: {pr(st['hit']['all'], st['tot']['all'])}; "
                f"missed-span precision {pr(mp, mt)}, recall {pr(mh, mg)}", "",
                "| group | acc% (n) |", "|---|---|"]
        for pref in ("label", "family"):
            keys = sorted(k for k in st["tot"] if k.startswith(pref + ":"))
            out += [f"| {k} | {pr(st['hit'][k], st['tot'][k])} |" for k in keys]
        cl = LABELS + ["-"]
        out += ["", "Label confusion (rows gold, cols predicted; '-' = missing/invalid)", "",
                "| gold \\ pred | " + " | ".join(cl) + " |", "|" + "---|" * (len(cl) + 1)]
        for g in LABELS:
            if g in st["conf"]:
                out.append(f"| {g} | " + " | ".join(str(st["conf"][g].get(p, 0)) for p in cl) + " |")
        out.append("")
    (OUT / f"summary_labels{'' if split == 'test' else '_dev'}.md").write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "both"], default="both")
    a = ap.parse_args()
    for split in (["test", "dev"] if a.split == "both" else [a.split]):
        for n, r in run_split(split).items():
            if n in ("rule_oracle", "rule_blind", "always_OK", "majority(dev)"):
                print(f"{split} {n}: acc={r['acc']:.3f} macroF1={r['macro_f1']:.3f} spanF1={r['span_f1']:.3f} pair={fmt(r['pair'])}")
    for split in (["test", "dev"] if a.split == "both" else [a.split]):
        run_labels(split)
    if a.split == "both":
        run_pooled()


if __name__ == "__main__":
    main()
