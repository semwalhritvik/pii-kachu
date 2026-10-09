import hashlib
import json
import pathlib
import random

from . import crapii, tab

RANK = {"test": 0, "dev": 1, "train": 2}
OUT = pathlib.Path("data/processed")


def main():
    docs = list(tab.load())
    cr = list(crapii.load())
    random.Random(42).shuffle(cr)
    n = len(cr)
    for i, d in enumerate(cr):
        d["split"] = "train" if i < 0.8 * n else "dev" if i < 0.9 * n else "test"
    docs += cr
    best = {}
    for d in docs:
        h = hashlib.sha1(d["text"].encode()).hexdigest()
        if h not in best or RANK[d["split"]] < RANK[best[h]["split"]]:
            best[h] = d
    keep = {id(d) for d in best.values()}
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    files = {s: open(OUT / f"{s}.jsonl", "w") for s in RANK}
    for d in docs:
        if id(d) in keep:
            files[d["split"]].write(json.dumps(d, ensure_ascii=False) + "\n")
            counts[d["split"]] = counts.get(d["split"], 0) + 1
    for f in files.values():
        f.close()
    print(f"dropped {len(docs) - len(keep)} duplicates; written {counts}")


if __name__ == "__main__":
    main()
