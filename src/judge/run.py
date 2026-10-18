"""Run judge models over pilot cases. Usage: python -m src.judge.run --model NAME [NAME...|all] --split dev|test|all [--with-reason] [--mode e2e|labels]"""
import argparse
import hashlib
import json
import subprocess
import urllib.request

import yaml

from src.judge.derive import ROOT, load_purposes
from src.judge.ollama_client import REQUEST_TIMEOUT, URL, chat
from src.judge.prompt import build_messages, output_schema
from src.judge.prompt_labels import build_messages_labels, output_schema_labels, system_labels
from src.judge.splits import load_cases, load_splits

OUT = ROOT / "results/bakeoff"


def models():
    return {m["name"]: m["tag"] for m in yaml.safe_load((ROOT / "configs/models.yaml").read_text())}


def unload(tag):
    body = json.dumps({"model": tag, "keep_alive": 0, "messages": []}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=60).read()
    except OSError as e:
        print(f"unload {tag} failed: {e}")


def git_state():
    """(short HEAD hash, dirty) where dirty = uncommitted/untracked changes outside results/."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--", ".", ":!results"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def model_digest(tag):
    """Digest of the local model from /api/tags (None if unavailable)."""
    try:
        with urllib.request.urlopen(URL.rsplit("/", 1)[0] + "/tags", timeout=30) as r:
            tags = json.loads(r.read())["models"]
    except (OSError, ValueError, KeyError):
        return None
    want = {tag, tag if ":" in tag else f"{tag}:latest"}
    return next((m["digest"] for m in tags if m.get("name") in want or m.get("model") in want), None)


def prompt_hash(purposes, with_reason, mode="e2e"):
    """sha1 of the rendered system prompt + output schema (policy/case text is per-case and excluded)."""
    if mode == "labels":
        return hashlib.sha1((system_labels() + json.dumps(output_schema_labels(), sort_keys=True)).encode()).hexdigest()
    system = build_messages({"purpose": next(iter(purposes)), "original": "", "redacted": "", "spans": []},
                            purposes, with_reason)[0]["content"]
    return hashlib.sha1((system + json.dumps(output_schema(with_reason), sort_keys=True)).encode()).hexdigest()


SKIP_ERRORS = ("timeout", "truncated")


def num_predict_for(mode, with_reason):
    return 600 if mode == "labels" else 400 if with_reason else 300


def done_ids(path, phash, digest, retry_errors=False):
    """case_ids answered under the same prompt_hash and model_digest. Timeout/truncated rows count as done
    unless retry_errors; other errored rows are rerun."""
    if not path.exists():
        return set()
    rows = map(json.loads, path.read_text().splitlines())
    return {r["case_id"] for r in rows if r.get("prompt_hash") == phash and r.get("model_digest") == digest
            and (not r["error"] or (r["error"] in SKIP_ERRORS and not retry_errors))}


def run_model(name, tag, cases, split, with_reason, mode="e2e", retry_errors=False):
    purposes = load_purposes()
    path = OUT / f"{name}{'-labels' if mode == 'labels' else '-reason' if with_reason else ''}_{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_state()
    phash, digest = prompt_hash(purposes, with_reason, mode), model_digest(tag)
    skip = done_ids(path, phash, digest, retry_errors)
    npred = num_predict_for(mode, with_reason)
    todo = [c for c in cases if c["case_id"] not in skip]
    print(f"{name}: {len(todo)} to run, {len(skip)} done -> {path.name}")
    schema = (output_schema_labels() if mode == "labels" else output_schema(with_reason))
    for i, c in enumerate(todo, 1):
        msgs = build_messages_labels(c) if mode == "labels" else build_messages(c, purposes, with_reason)
        r = chat(tag, msgs, schema, num_predict=npred)
        tps = r["eval_count"] / (r["eval_duration_ns"] / 1e9) if r["eval_count"] and r["eval_duration_ns"] else None
        row = {"case_id": c["case_id"], "raw": r["raw"], "parsed": r["parsed"], "valid_json": r["valid_json"],
               "latency_s": r["latency_s"], "tokens_per_s": tps, "error": r["error"], "eval_count": r["eval_count"],
               "done_reason": r["done_reason"], "truncated": r["truncated"], "num_predict": npred,
               "request_timeout": REQUEST_TIMEOUT,
               "prompt_hash": phash, "model_digest": digest, "git_commit": commit, "git_dirty": dirty}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {c['case_id']} {r['latency_s']:.1f}s eval_count={r['eval_count']} valid_json={r['valid_json']} "
              f"done_reason={r['done_reason']}" + (f" error={r['error']}" if r["error"] else ""), flush=True)
    unload(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--split", choices=["dev", "test", "all"], required=True)
    ap.add_argument("--with-reason", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="also rerun rows that ended in timeout/truncated")
    ap.add_argument("--mode", choices=["e2e", "labels"], default="e2e")
    a = ap.parse_args()
    if a.mode == "labels" and a.with_reason:
        ap.error("--with-reason applies to e2e mode only")
    reg = models()
    names = list(reg) if a.model == ["all"] else a.model
    cases = load_cases()
    if a.split != "all":
        keep = set(load_splits()[a.split])
        cases = [c for c in cases if c["case_id"] in keep]
    for n in names:
        run_model(n, reg.get(n, n), cases, a.split, a.with_reason, a.mode, a.retry_errors)


if __name__ == "__main__":
    main()
