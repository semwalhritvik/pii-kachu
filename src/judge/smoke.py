"""Smoke test: run the first 3 pilot cases against one model. Usage: python -m src.judge.smoke <name-or-tag>"""
import json
import sys
from pathlib import Path

import yaml

from src.judge.derive import ROOT, load_purposes
from src.judge.ollama_client import chat
from src.judge.prompt import build_messages, output_schema

PILOT = ROOT / "pilot/pilot_cases.jsonl"


def resolve(name):
    models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    return next((m["tag"] for m in models if name in (m["name"], m["tag"])), name)


def main(name, n=3):
    tag, purposes = resolve(name), load_purposes()
    with open(PILOT) as f:
        cases = [json.loads(next(f)) for _ in range(n)]
    for c in cases:
        r = chat(tag, build_messages(c, purposes), output_schema())
        p = r["parsed"] or {}
        tps = r["eval_count"] / (r["eval_duration_ns"] / 1e9) if r["eval_count"] and r["eval_duration_ns"] else float("nan")
        print(f"{c['case_id']} verdict={p.get('verdict')} offending={p.get('offending_span_ids')} "
              f"latency={r['latency_s']:.1f}s tok/s={tps:.1f} valid={r['valid_json']} err={r['error']}")


if __name__ == "__main__":
    main(sys.argv[1])
