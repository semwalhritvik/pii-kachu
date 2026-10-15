"""Dev/test split of pilot cases: stratified search over text groups (shared pair_id or identical original), seed 42."""
import json
import random
from collections import Counter, defaultdict

import yaml

from src.judge.derive import ROOT, derive_verdict, load_purposes

PILOT = ROOT / "pilot/pilot_cases.jsonl"
SPLITS = ROOT / "pilot/splits.json"


FAMILIES = ROOT / "configs/families.yaml"
FLAGS = ("quasi", "detected=false", "mixed-fault", "pair member")
TRIALS, LO, HI, MIN_PURPOSE = 20000, 0.44, 0.56, 4


def family_map():
    return {t: f for f, ts in yaml.safe_load(FAMILIES.read_text()).items() for t in ts}


def load_cases(path=PILOT):
    """Cases from the jsonl plus a load-time `family` field (not stored); raises on an unmapped hard_case_type."""
    fm = family_map()
    with open(path) as f:
        cases = [json.loads(l) for l in f if l.strip()]
    unmapped = sorted({c["hard_case_type"] for c in cases} - set(fm))
    if unmapped:
        raise KeyError(f"hard_case_type not in configs/families.yaml: {unmapped}")
    for c in cases:
        c["family"] = fm[c["hard_case_type"]]
    return cases


def groups(cases):
    """Union-find over cases sharing pair_id or an identical original text."""
    parent = list(range(len(cases)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, c in enumerate(cases):
        for key in (("original", c["original"]), ("pair", c.get("pair_id"))):
            if key[1] is None:
                continue
            j = first.setdefault(key, i)
            parent[find(i)] = find(j)
    out = defaultdict(list)
    for i in range(len(cases)):
        out[find(i)].append(i)
    return list(out.values())


def flags(case, purposes):
    """Composition flags: quasi, detected=false, mixed-fault (>1 distinct fault type), pair member."""
    ft = set(derive_verdict(case, purposes)["fault_type"].values())
    return {"quasi": bool(case.get("quasi_groups")),
            "detected=false": any(s.get("detected", True) is False for s in case["spans"]),
            "mixed-fault": len(ft) > 1, "pair member": case.get("pair_id") is not None}


def features(case, fl):
    """Stratification keys a case contributes one count to."""
    p = case["purpose"]
    return ([("purpose", p), ("verdict", case["gold_verdict"]), ("purpose x ok", (p, case["gold_verdict"] == "OK")),
             ("family", case["family"])] + [("flag", f) for f in FLAGS if fl[f]])


def make_splits(cases, seed=42):
    """Best of TRIALS random group assignments (each side LO..HI of cases, >= MIN_PURPOSE per purpose per side),
    minimising summed |dev - test| over features(). Returns {"dev": [case_id], "test": [case_id]}."""
    rng, purposes = random.Random(seed), load_purposes()
    gs = groups(cases)
    fl = [flags(c, purposes) for c in cases]
    gfeat = [Counter(k for i in g for k in features(cases[i], fl[i])) for g in gs]
    n, all_p = len(cases), sorted({c["purpose"] for c in cases})
    total = Counter()
    for g in gfeat:
        total.update(g)
    best, best_cost = None, None
    for _ in range(TRIALS):
        pick = [rng.random() < 0.5 for _ in gs]
        dev_n = sum(len(g) for g, d in zip(gs, pick) if d)
        if not LO * n <= dev_n <= HI * n:
            continue
        dev = Counter()
        for g, d in zip(gfeat, pick):
            if d:
                dev.update(g)
        if any(dev[("purpose", p)] < MIN_PURPOSE or total[("purpose", p)] - dev[("purpose", p)] < MIN_PURPOSE for p in all_p):
            continue
        cost = sum(abs(2 * dev[k] - total[k]) for k in total)  # |dev - test| = |2 dev - total|
        if best_cost is None or cost < best_cost:
            best, best_cost = pick, cost
    if best is None:
        raise RuntimeError("no assignment satisfied the size/purpose constraints")
    side = {"dev": [], "test": []}
    for g, d in zip(gs, best):
        side["dev" if d else "test"] += [cases[i]["case_id"] for i in g]
    order = {c["case_id"]: i for i, c in enumerate(cases)}
    return {k: sorted(v, key=order.get) for k, v in side.items()}


def load_splits():
    return json.loads(SPLITS.read_text())


def flags(case, purposes):
    """Composition flags: quasi, detected=false, mixed-fault (>1 distinct fault type), pair member."""
    ft = set(derive_verdict(case, purposes)["fault_type"].values())
    return {"quasi": bool(case.get("quasi_groups")),
            "detected=false": any(s.get("detected", True) is False for s in case["spans"]),
            "mixed-fault": len(ft) > 1, "pair member": case.get("pair_id") is not None}


def composition(cases, sp):
    purposes, by_id = load_purposes(), {c["case_id"]: c for c in cases}
    for k, ids in sp.items():
        if k == "seed":
            continue
        cs = [by_id[i] for i in ids]
        print(f"[{k}] n={len(cs)}")
        for key in ("purpose", "hard_case_type"):
            print(f"  {key}:", dict(sorted(Counter(str(c[key]) for c in cs).items())))
        fl = Counter(f for c in cs for f, v in flags(c, purposes).items() if v)
        print("  flags:", {f: fl[f] for f in ("quasi", "detected=false", "mixed-fault", "pair member")})


def composition(cases, sp):
    """Print one table per dimension with dev | test counts side by side."""
    purposes, by_id = load_purposes(), {c["case_id"]: c for c in cases}
    side = {k: [by_id[i] for i in sp[k]] for k in ("dev", "test")}
    fl = {k: [flags(c, purposes) for c in cs] for k, cs in side.items()}
    dims = {
        "purpose": lambda c, f: [c["purpose"]],
        "verdict": lambda c, f: [c["gold_verdict"]],
        "purpose x verdict": lambda c, f: [f"{c['purpose'][:2]} {c['gold_verdict']}"],
        "family": lambda c, f: [c["family"]],
        "flags": lambda c, f: [k for k in FLAGS if f[k]],
    }
    print(f"{'':<34}{'dev':>5}{'test':>6}")
    print(f"{'cases':<34}{len(side['dev']):>5}{len(side['test']):>6}")
    for name, fn in dims.items():
        cnt = {k: Counter(v for c, f in zip(side[k], fl[k]) for v in fn(c, f)) for k in side}
        print(f"-- {name}")
        for key in sorted(set(cnt["dev"]) | set(cnt["test"])):
            print(f"  {key:<32}{cnt['dev'][key]:>5}{cnt['test'][key]:>6}")


def main():
    cases = load_cases()
    fv = Counter((c["family"], c["gold_verdict"]) for c in cases)
    print("-- all cases: family x verdict")
    for f in sorted({c["family"] for c in cases}):
        print(f"  {f:<28}" + " ".join(f"{v[:5]}={fv[f, v]}" for v in ("OK", "OVER", "UNDER", "SHOULD_PSEUDONYMIZE")))
    sp = make_splits(cases)
    SPLITS.parent.mkdir(exist_ok=True)
    SPLITS.write_text(json.dumps({"seed": 42, **sp}, indent=1))
    composition(cases, sp)


if __name__ == "__main__":
    main()
