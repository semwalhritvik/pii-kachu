"""Tests for the helper cells of llm_p5/qwen_p5_eval.ipynb (no Ollama, no model: a fake local server stands in)."""
import http.server
import json
import math
import socket
import threading
import time
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "llm_p5" / "qwen_p5_eval.ipynb"


@pytest.fixture(scope="module")
def nb():
    with open(NB, encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


@pytest.fixture(scope="module")
def ns(nb):
    g = {"__name__": "nb_helpers"}
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and "helpers" in c.metadata.get("tags", []):
            exec(compile(c.source, f"cell{i}", "exec"), g)
    return g


# ---------------- windows / matching / merge ----------------
def test_windows_cover_overlap_and_boundaries(ns):
    text = " ".join(f"Sentence number {i} says something." for i in range(120))
    wins = ns["make_windows"](text, 300, 60)
    assert wins[0][0] == 0 and wins[-1][1] == len(text)
    for (s, e), (s2, e2) in zip(wins, wins[1:]):
        assert e - s <= 300 and s < s2 < e            # progress and overlap
        assert text[s2 - 1].isspace()                 # next window starts at a word
    assert all(text[e - 1] in ". " for _, e in wins[:-1])   # cut after a sentence end
    assert ns["make_windows"]("short", 300, 60) == [(0, 5)] and ns["make_windows"]("", 300, 60) == []


def test_match_strings_all_occurrences_word_boundary_offsets(ns):
    full = "x" * 50 + "Ann met Annabel. Then Ann left."
    win = full[50:]
    items = [{"text": "Ann", "label": "PERSON_PRIVATE", "decision": "REDACT"},
             {"text": "Annabel", "label": "PERSON_PRIVATE", "decision": "KEEP"},
             {"text": "Ghost", "label": "ORG", "decision": "KEEP"}]
    preds, unmatched = ns["match_strings"](win, items, offset=50)
    spans = [(p["start"], p["end"], p["text"]) for p in preds]
    assert spans == [(50, 53, "Ann"), (58, 65, "Annabel"), (72, 75, "Ann")]   # not "Ann" inside Annabel
    assert all(full[s:e] == t for s, e, t in spans)
    assert unmatched == 1


def test_longest_first_no_overlap(ns):
    preds, _ = ns["match_strings"]("John Smith arrived", [
        {"text": "John", "label": "PERSON_PRIVATE", "decision": "KEEP"},
        {"text": "John Smith", "label": "PERSON_PRIVATE", "decision": "REDACT"}])
    assert [(p["text"], p["decision"]) for p in preds] == [("John Smith", "REDACT")]


def test_merge_preds_redact_wins(ns):
    a = {"start": 1, "end": 4, "text": "Ann", "label": "X", "decision": "KEEP"}
    b = {**a, "decision": "REDACT"}
    assert ns["merge_preds"]([a, b])[0]["decision"] == "REDACT"
    assert ns["merge_preds"]([b, a])[0]["decision"] == "REDACT"
    assert len(ns["merge_preds"]([a, b, {**a, "start": 9, "end": 12}])) == 2


def _fake(replies):
    it = iter(replies)
    return lambda messages, schema: next(it)


def _ok(spans):
    return {"valid_json": True, "truncated": False, "parsed": {"spans": spans}, "latency_s": 0.1,
            "tokens_per_s": 5.0, "eval_count": 3, "error": None}


def test_detect_doc_overlap_union_and_redact_wins(ns):
    text = "Alice is here. " * 40
    n_win = len(ns["make_windows"](text, 200, 60))
    assert n_win > 2
    replies = [_ok([{"text": "Alice", "label": "PERSON_PRIVATE", "decision": "REDACT" if i == 0 else "KEEP"},
                    {"text": "Zed", "label": "ORG", "decision": "KEEP"}]) for i in range(n_win)]
    row = ns["detect_doc"]({"doc_id": "d", "source": "crapii", "split": "dev", "text": text}, _fake(replies), 200, 60)
    keys = [(p["start"], p["end"]) for p in row["preds"]]
    assert len(keys) == len(set(keys)) == 40                       # one pred per occurrence, no duplicates
    assert row["n_windows"] == n_win and row["n_unmatched"] == n_win
    first_win_end = ns["make_windows"](text, 200, 60)[0][1]
    assert all(p["decision"] == "REDACT" for p in row["preds"] if p["end"] <= first_win_end)
    # an occurrence in the overlap of windows 0 and 1 is REDACT (from window 0) although window 1 said KEEP
    s1, e1 = ns["make_windows"](text, 200, 60)[1]
    assert any(p["start"] >= s1 and p["end"] <= first_win_end and p["decision"] == "REDACT" for p in row["preds"])


def test_detect_doc_truncated_and_invalid_windows_yield_nothing(ns):
    text = "Alice is here. " * 40
    n_win = len(ns["make_windows"](text, 200, 60))
    bad = [{"valid_json": True, "truncated": True, "parsed": {"spans": []}, "error": "truncated"},
           {"valid_json": False, "truncated": False, "parsed": None, "error": None}]
    halves = [{"valid_json": True, "truncated": True, "parsed": None, "error": "truncated"}] * 2   # recovery of window 0 fails too
    replies = bad[:1] + halves + bad[1:] + [_ok([{"text": "Alice", "label": "PERSON_PRIVATE", "decision": "REDACT"}])] * (n_win - 2)
    row = ns["detect_doc"]({"doc_id": "d", "text": text}, _fake(replies), 200, 60)
    assert (row["n_truncated"], row["n_invalid"]) == (1, 1)
    assert row["errors"] == {"truncated": 3}
    first_two_end = ns["make_windows"](text, 200, 60)[1][1]
    s2 = ns["make_windows"](text, 200, 60)[2][0]
    assert all(p["start"] >= s2 or p["end"] > first_two_end for p in row["preds"])


# ---------------- gold adapters ----------------
TAB_DOC = {
    "doc_id": "t", "source": "tab", "text": "Ann met Bob in Paris on 5 May 1990 about 40 units.",
    "spans": [
        {"start": 0, "end": 3, "label": "PERSON_PRIVATE", "attrs": {"entity_type": "PERSON", "identifier_type": "DIRECT"}},
        {"start": 8, "end": 11, "label": "PERSON_PUBLIC", "attrs": {"entity_type": "PERSON", "identifier_type": "NO_MASK"}},
        {"start": 15, "end": 20, "label": "LOCATION_OTHER", "attrs": {"entity_type": "LOC", "identifier_type": "QUASI"}}],
    "ignored_spans": [{"start": 41, "end": 49, "label": "QUANTITY", "attrs": {"identifier_type": "QUASI"}}],
    "other_annotations": {"a2": [
        {"start": 24, "end": 36, "label": "DATE_OTHER", "attrs": {"entity_type": "DATETIME", "identifier_type": "DIRECT"}},
        {"start": 41, "end": 49, "label": "QUANTITY", "attrs": {"identifier_type": "QUASI"}}]},
}


def _dec(gold):
    return {(g["start"], g["end"]): g["decision"] for g in gold}


def test_gold_tab_scopes_and_annotators(ns):
    dq = _dec(ns["gold_tab"](TAB_DOC, "direct+quasi", "first"))
    assert dq[(0, 3)] == "redact" and dq[(8, 11)] == "keep" and dq[(15, 20)] == "redact" and dq[(41, 49)] == "ignore"
    d = _dec(ns["gold_tab"](TAB_DOC, "direct", "first"))
    assert d[(15, 20)] == "ignore" and d[(0, 3)] == "redact" and d[(8, 11)] == "keep"
    assert (24, 36) not in dq and _dec(ns["gold_tab"](TAB_DOC, "direct", "union"))[(24, 36)] == "redact"


def test_gold_crapii_and_synthetic(ns):
    doc = {"source": "crapii", "text": "x", "spans": [{"start": 0, "end": 1, "label": "EMAIL", "attrs": {}}]}
    assert _dec(ns["gold_for"](doc)) == {(0, 1): "redact"}
    case = {"source": "synthetic", "spans": [{"text": "a", "start": 0, "end": 1, "gold_label": "ORG", "expected_decision": "KEEP"},
                                             {"text": "b", "start": 2, "end": 3, "gold_label": "EMAIL", "expected_decision": "REDACT"}]}
    assert _dec(ns["gold_for"](case)) == {(0, 1): "keep", (2, 3): "redact"}
    assert ns["expected_decision"]("ORG") == "KEEP" and ns["expected_decision"]("DOB") == "REDACT"


# ---------------- metrics ----------------
TEXT = "Ann met Bob in Paris. Call 555-1234 now."


def _sp(sub, label, decision):
    s = TEXT.index(sub)
    return {"start": s, "end": s + len(sub), "label": label, "decision": decision}


def _pred(sub, decision="REDACT"):
    p = _sp(sub, "X", decision)
    return {**p, "text": sub}


def test_metrics_hand_example(ns):
    gold = [_sp("Ann", "PERSON_PRIVATE", "redact"), _sp("Bob", "PERSON_PUBLIC", "keep"),
            _sp("Paris", "LOCATION_OTHER", "keep"), _sp("555-1234", "PHONE", "redact")]
    preds = [_pred("Ann"), _pred("Paris"), _pred("555"), _pred("Bob", "KEEP")]
    c1 = ns["doc_counts"](TEXT, gold, preds)
    c2 = ns["doc_counts"]("hello world", [], [{"start": 6, "end": 11, "text": "world", "decision": "REDACT"}])
    m1 = ns["metrics_from_sums"](c1)
    P, R = 2 / 3, 1 / 2
    assert m1["span_recall"] == pytest.approx(R) and m1["span_precision"] == pytest.approx(P)
    assert m1["f2"] == pytest.approx(5 * P * R / (4 * P + R))
    assert m1["char_leakage_pct"] == pytest.approx(100 * 5 / 11)          # "-1234" uncovered of 11 gold chars
    assert m1["char_over_redaction_pct"] == pytest.approx(100 * 5 / 11)   # "Paris" of 11 flagged chars
    nonws = sum(not ch.isspace() for ch in TEXT)
    assert m1["non_pii_flag_rate"] == pytest.approx(5 / (nonws - 11))
    assert m1["doc_leakage"] == 1.0 and math.isnan(m1["doc_false_alarm"])
    assert m1["keep_false_redaction_rate"] == pytest.approx(0.5)
    assert m1["keep_false|LOCATION_OTHER"] == 1.0 and m1["keep_false|PERSON_PUBLIC"] == 0.0
    assert m1["recall|PERSON_PRIVATE"] == 1.0 and m1["recall|PHONE"] == 0.0
    both = {k: c1.get(k, 0) + c2.get(k, 0) for k in set(c1) | set(c2)}
    mb = ns["metrics_from_sums"](both)
    assert mb["doc_false_alarm"] == 1.0 and mb["doc_leakage"] == 1.0


def test_metrics_ignore_regions_excluded(ns):
    gold = [_sp("Ann", "PERSON_PRIVATE", "redact"), _sp("Paris", "LOCATION_OTHER", "ignore")]
    c = ns["doc_counts"](TEXT, gold, [_pred("Ann"), _pred("Paris")])
    assert c["pred_n"] == 1 and c["flag_nonpii"] == 0                     # the Paris prediction is dropped entirely
    nonws = sum(not ch.isspace() for ch in TEXT)
    assert c["nonpii_chars"] == nonws - 3 - 5


def test_bootstrap_ci_seeded_and_brackets(ns):
    import pandas as pd
    rows = [ns["doc_counts"](TEXT, [_sp("Ann", "PERSON_PRIVATE", "redact")], [_pred("Ann")] if i % 2 else [])
            for i in range(20)]
    df = pd.DataFrame(rows).fillna(0)
    a, b = ns["bootstrap_ci"](df, 200, 42), ns["bootstrap_ci"](df, 200, 42)
    assert repr(a) == repr(b)
    lo, hi = a["span_recall"]
    assert lo <= 0.5 <= hi and 0 <= lo and hi <= 1


def test_score_table_per_source_never_pooled(ns):
    docs = {"a": {"doc_id": "a", "source": "crapii", "split": "dev", "text": TEXT,
                  "spans": [{**_sp("Ann", "PERSON_PRIVATE", "redact"), "attrs": {}}]},
            "b": {**TAB_DOC, "split": "dev"}}
    rows = [{"doc_id": "a", "preds": [_pred("Ann")]}, {"doc_id": "b", "preds": []}]
    summ, lab = ns["score_table"](rows, docs, n_boot=20)
    assert sorted(summ.source) == ["crapii", "tab"] and set(summ.n_docs) == {1}
    assert (lab.n > 0).all()
    miss, fals = ns["worst_errors"](rows, docs, k=10)
    assert set(miss.source) == {"tab"} and miss.context.str.len().max() <= 120


# ---------------- client guardrails (fake streaming server) ----------------
class _Server:
    def __init__(self, handler):
        self.stop = threading.Event()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.end_headers()
                try:
                    handler(self, outer.stop)
                except OSError:
                    pass

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/api/chat"

    def close(self):
        self.stop.set()
        self.srv.shutdown()
        self.srv.server_close()


def _line(h, obj):
    h.wfile.write(json.dumps(obj).encode() + b"\n")
    h.wfile.flush()


def test_client_deadline_on_server_that_never_ends(ns):
    def never(h, stop):
        _line(h, {"message": {"content": "{"}, "done": False})
        stop.wait(30)

    s = _Server(never)
    try:
        t0 = time.perf_counter()
        r = ns["chat"]("m", [], {}, s.url, timeout=1.5, num_predict=900)
        assert time.perf_counter() - t0 < 6
        assert r["error"] == "timeout" and not r["valid_json"] and r["parsed"] is None
    finally:
        s.close()


def test_client_deadline_when_server_sends_nothing(ns):
    s = _Server(lambda h, stop: stop.wait(30))
    try:
        t0 = time.perf_counter()
        r = ns["chat"]("m", [], {}, s.url, timeout=1.5)
        assert time.perf_counter() - t0 < 6 and r["error"] == "timeout"
    finally:
        s.close()


def test_client_token_cap_marks_truncated_without_retry(ns):
    def endless(h, stop):
        while not stop.is_set():
            _line(h, {"message": {"content": "x"}, "done": False})

    s = _Server(endless)
    try:
        r = ns["chat"]("m", [], {}, s.url, timeout=20, num_predict=20)
        assert r["truncated"] and r["done_reason"] == "length" and r["error"] == "truncated" and not r["valid_json"]
    finally:
        s.close()


def test_client_valid_reply_and_invalid_json_not_retried(ns):
    calls = []

    def good(h, stop):
        calls.append(1)
        _line(h, {"message": {"content": '{"spans": []}'}, "done": False})
        _line(h, {"message": {"content": ""}, "done": True, "done_reason": "stop", "eval_count": 7,
                  "eval_duration": 1_000_000_000, "load_duration": 2_000_000_000})

    s = _Server(good)
    try:
        r = ns["chat"]("m", [], {}, s.url)
        assert r["valid_json"] and r["parsed"] == {"spans": []} and r["tokens_per_s"] == pytest.approx(7)
        assert r["load_duration_ns"] == 2_000_000_000 and not r["truncated"]
    finally:
        s.close()

    def junk(h, stop):
        calls.append(1)
        _line(h, {"message": {"content": "not json"}, "done": False})
        _line(h, {"message": {"content": ""}, "done": True, "done_reason": "stop"})

    calls.clear()
    s = _Server(junk)
    try:
        r = ns["chat"]("m", [], {}, s.url)
        assert not r["valid_json"] and r["raw"] == "not json" and len(calls) == 1
    finally:
        s.close()


def test_client_refused_connection_retried_once(ns, monkeypatch):
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    n = []
    real = ns["_stream"]
    monkeypatch.setitem(ns, "_stream", lambda *a, **k: (n.append(1), real(*a, **k))[1])
    r = ns["chat"]("m", [], {}, f"http://127.0.0.1:{port}/api/chat", timeout=5)
    assert len(n) == 2 and "ConnectionRefused" in r["error"] and not r["valid_json"]


# ---------------- system-check logic ----------------
def test_decision_agreement_and_set_agreement(ns):
    gold = [{"text": "Ann", "start": 0, "end": 3, "expected_decision": "REDACT"},
            {"text": "Paris", "start": 10, "end": 15, "expected_decision": "KEEP"},
            {"text": "Bob", "start": 20, "end": 23, "expected_decision": "REDACT"}]
    preds = [{"start": 0, "end": 3, "text": "Ann", "decision": "REDACT"},
             {"start": 10, "end": 15, "text": "Paris", "decision": "REDACT"}]
    a, t, bad = ns["decision_agreement"](gold, preds)
    assert (a, t) == (1, 3) and {b["text"] for b in bad} == {"Paris", "Bob"}
    assert ns["set_agreement"]([["a", "REDACT"], ["b", "KEEP"]], [["a", "REDACT"], ["b", "REDACT"]]) == pytest.approx(100 / 3)
    assert ns["set_agreement"]([], []) == 100.0
    assert ns["pred_set"](preds) == [("Ann", "REDACT"), ("Paris", "REDACT")]


def test_latency_stats_excludes_first_call(ns):
    w = lambda lat, tps=10.0, valid=True, trunc=False: {"latency_s": lat, "tokens_per_s": tps, "valid": valid,
                                                        "truncated": trunc, "load_s": 4.0}
    rows = [{"windows": [w(30.0, 1.0)]}, {"windows": [w(2.0), w(4.0, valid=False)]}, {"windows": [w(6.0, trunc=True, valid=False)]}]
    st = ns["latency_stats"](rows)
    assert st["median_latency_s"] == 4.0 and st["first_call_latency_s"] == 30.0 and st["model_load_s"] == 4.0
    assert st["valid_json_rate"] == pytest.approx(1 / 3) and st["n_truncated"] == 1 and st["median_tokens_per_s"] == 10.0


def test_sampling_stratified_and_seeded(ns):
    docs = [{"doc_id": f"d{i:03d}", "spans": [1] if i % 5 == 0 else []} for i in range(200)]
    s = ns["sample_docs"](docs, 20, stratified=True, seed=42)
    assert len(s) == 20 and sum(bool(d["spans"]) for d in s) == 10
    assert s == ns["sample_docs"](docs, 20, stratified=True, seed=42)
    few = [{"doc_id": f"d{i}", "spans": [1] if i < 2 else []} for i in range(50)]
    assert len(ns["sample_docs"](few, 10, stratified=True)) == 10     # tops up from the other group


# ---------------- prompts ----------------
def test_review_prompt_has_no_gold_fields(ns):
    cands = [{"text": "Ann", "label": "PERSON_PRIVATE", "score": 0.91, "gold_label": "SECRET_GOLD",
              "expected_decision": "SECRET_EXPECTED", "decision": "SECRET_DECISION"}]
    msgs = ns["build_review_messages"]("Ann met Bob.", cands)
    blob = json.dumps(msgs)
    assert "SECRET" not in blob and "gold" not in blob.lower() and "expected" not in blob.lower()
    assert "1 | Ann | PERSON_PRIVATE | 0.91" in msgs[1]["content"]
    assert "gold" not in json.dumps(ns["build_messages"]("t")).lower()
    assert ns["prompt_hash"]("review") != ns["prompt_hash"]("standalone")


# ---------------- notebook hygiene ----------------
def test_notebook_valid_compiles_no_outputs(nb):
    nbformat.validate(nb)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            compile(c.source, f"cell{i}", "exec")
            assert c.outputs == [] and c.execution_count is None
    assert any("helpers" in c.metadata.get("tags", []) for c in nb.cells)


# ---------------- needs data/ (skipped when absent) ----------------
def test_real_first_doc_roundtrip(ns):
    path = ROOT / "data" / "processed" / "dev.jsonl"
    if not path.exists():
        pytest.skip("data/processed/dev.jsonl not available")
    with open(path, encoding="utf-8") as f:
        doc = json.loads(f.readline())
    wins = ns["make_windows"](doc["text"], 1200, 100)
    assert wins[-1][1] == len(doc["text"])
    c = ns["doc_counts"](doc["text"], ns["gold_for"](doc), [])
    assert c["gold_uncov"] == c["gold_chars"]


# ---------------- dev+train protocol: lock, prefix sampling, budget, prompt guard, comparison ----------------
V0_HASH = "2b43662c93c3"            # prompt_hash of the committed v0 prompt (before COPY_EXACT_LINE)
COPY_LINE = "- Copy each string exactly as it appears in the text, including capitalization and punctuation.\n"


def test_test_split_is_locked(ns):
    f = ns["check_run_splits"]
    with pytest.raises(RuntimeError, match="locked"):
        f(["dev", "test"])
    with pytest.raises(RuntimeError, match="locked"):
        f(["test"], allow_test=False)
    assert f(["train", "dev"]) == ["dev", "train"]                  # dev always before train
    assert f(["dev", "test"], allow_test=True) == ["dev", "test"]
    with pytest.raises(ValueError):
        f(["dev", "val"])


def test_prefix_property_stratified_deterministic(ns):
    docs = [{"doc_id": f"d{i:03d}", "source": "crapii", "spans": [1] if i % 4 == 0 else []} for i in range(300)]
    ids = lambda n: [d["doc_id"] for d in ns["sample_docs"](docs, n, True, 42)]
    a, b, c = ids(40), ids(100), ids(40)
    assert a == c and b[:40] == a and len(set(b)) == 100            # deterministic prefix, no repeats
    for n in (40, 100):
        s = ns["sample_docs"](docs, n, True, 42)
        assert sum(bool(d["spans"]) for d in s) == n // 2           # stratified 1:1 at every size
    # stopping early on the full order is still stratified
    assert sum(bool(d["spans"]) for d in ns["sample_order"](docs, True, 42)[:20]) == 10
    assert ids(40) != [d["doc_id"] for d in ns["sample_docs"](docs, 40, True, 7)]      # seed matters
    tab = [{"doc_id": f"t{i}", "source": "tab", "spans": [1]} for i in range(30)]
    assert ns["sample_docs"](tab, 10, False)[:5] == ns["sample_docs"](tab, 5, False)


def test_interleave_sources_proportional(ns):
    crapii = [{"doc_id": f"c{i}", "source": "crapii", "spans": [1] if i % 2 else []} for i in range(150)]
    tab = [{"doc_id": f"t{i}", "source": "tab", "spans": [1]} for i in range(20)]
    order = ns["interleave_sources"]({"crapii": crapii, "tab": tab})
    assert len(order) == 170 and len({d["doc_id"] for d in order}) == 170
    head = order[:34]                                              # an early stop keeps both sources, ~proportional
    assert 2 <= sum(d["source"] == "tab" for d in head) <= 6
    sel = ns["select_split"](crapii + tab, {"crapii": 10, "tab": 4}, 42)
    assert len(sel) == 14 and {d["source"] for d in sel} == {"crapii", "tab"}


def _meta(**kw):
    m = {"prompt_hash": "h", "model_digest": "dg", "model_tag": "m", "prompt_version": "v0.1", "git_commit": "c",
         "git_dirty": False, "machine": {"chip": "Chip", "ram_gb": 16, "os": "OS", "ollama_version": "0.1"},
         "prompt_modified": False}
    m.update(kw)
    return m


def test_budget_stop_resumable_and_rows_stamped(ns, tmp_path):
    docs = [{"doc_id": f"d{i}", "source": "crapii", "split": "dev", "text": "Alice."} for i in range(6)]
    calls, t = [], [0.0]

    def chat(m, s):
        calls.append(1)
        t[0] += 10
        return _ok([])

    path = tmp_path / "rows.jsonl"
    rows = ns["run_detection"](docs, path, chat, _meta(), max_seconds=25, clock=lambda: t[0], log=lambda *_: None)
    assert len(rows) == 3 and len(calls) == 3                       # starts at t=0,10,20; stops at 30
    saved = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 3
    for r in saved:
        assert r["prompt_version"] == "v0.1" and r["prompt_modified"] is False
        assert set(r["machine"]) == {"chip", "ram_gb", "os", "ollama_version"}
        assert r["sample_fingerprint"] == ns["sample_fingerprint"](docs) and r["prompt_hash"] == "h"
    rows2 = ns["run_detection"](docs, path, chat, _meta(), log=lambda *_: None)   # resume
    assert len(rows2) == 6 and len(calls) == 6                      # only the 3 remaining docs were run
    assert len(path.read_text(encoding="utf-8").splitlines()) == 6
    ns["run_detection"](docs, path, chat, _meta(prompt_hash="other"), max_seconds=0, log=lambda *_: None)
    assert len(calls) == 6                                          # different prompt hash: rows not reused, budget 0 runs nothing


def test_prompt_hash_guard(ns, tmp_path):
    actual = ns["prompt_hash"]("standalone")
    logs = []
    assert ns["enforce_prompt_hash"](actual, actual) is False
    with pytest.raises(RuntimeError, match="ALLOW_PROMPT_CHANGE"):
        ns["enforce_prompt_hash"](actual, "deadbeef0000")
    mod = ns["enforce_prompt_hash"](actual, "deadbeef0000", True, log=logs.append)
    assert mod is True and any("WARNING" in l for l in logs)
    docs = [{"doc_id": "a", "source": "crapii", "text": "x."}]
    ns["run_detection"](docs, tmp_path / "r.jsonl", lambda m, s: _ok([]), _meta(prompt_modified=mod), log=lambda *_: None)
    assert json.loads((tmp_path / "r.jsonl").read_text(encoding="utf-8"))["prompt_modified"] is True


def test_config_hash_matches_prompt(nb, ns):
    src = next(c.source for c in nb.cells if c.cell_type == "code" and "EXPECTED_PROMPT_HASH =" in c.source)
    import re
    assert re.search(r'EXPECTED_PROMPT_HASH = "([0-9a-f]{12})"', src).group(1) == ns["prompt_hash"]("standalone")
    assert 'PROMPT_VERSION = "v0.1"' in src and "ALLOW_TEST = False" in src and 'RUN_SPLITS = ["dev", "train"]' in src


def test_prompt_changed_only_by_copy_line(ns):
    system = ns["SYSTEM_STANDALONE"]
    assert system.count(COPY_LINE) == 1
    import hashlib
    v0 = hashlib.sha1((system.replace(COPY_LINE, "") + json.dumps(ns["SCHEMA_STANDALONE"], sort_keys=True)).encode()).hexdigest()[:12]
    assert v0 == V0_HASH


def _row(i, preds, digest="dg", ph="h", lat=1.0, chip="A"):
    return {"doc_id": i, "source": "crapii", "split": "dev", "preds": preds, "model_digest": digest, "prompt_hash": ph,
            "prompt_version": "v0.1", "window": 1200, "overlap": 100, "latency_s": lat, "n_truncated": 0, "n_unmatched": 1,
            "machine": {"chip": chip, "ram_gb": 16},
            "windows": [{"latency_s": lat, "tokens_per_s": 10.0, "valid": True, "truncated": False, "load_s": 0}] * 2}


def _p(text, dec="REDACT", label="PERSON_PRIVATE", s=0):
    return {"start": s, "end": s + len(text), "text": text, "label": label, "decision": dec}


def test_machine_comparison_on_fake_rows(ns):
    a = [_row("d1", [_p("Ann")]), _row("d2", [_p("Bob"), _p("Paris", "KEEP", "LOCATION_OTHER", 10)]), _row("d3", [])]
    b = [_row("d1", [_p("Ann")], chip="B"), _row("d2", [_p("Bob")], chip="B"), _row("d4", [_p("Zed")], chip="B")]
    cmp = ns["check_comparable"](a, b)
    assert cmp["valid"] and cmp["common"] == ["d1", "d2"]
    assert any("doc coverage differs" in w for w in cmp["warnings"])
    summ, per = ns["run_agreement"](a, b)
    assert summ["n_common"] == 2 and summ["share_identical"] == 0.5
    assert per.set_index("doc_id").loc["d2", "jaccard_all"] == pytest.approx(0.5)
    assert summ["mean_jaccard_redact"] == 1.0 and summ["mean_jaccard_all"] == pytest.approx(0.75)
    bad = ns["check_comparable"](a, [_row("d1", [], digest="other", ph="h2")])
    assert not bad["valid"] and any("digest" in w for w in bad["warnings"]) and any("prompt_hash differs" in w for w in bad["warnings"])
    spd = ns["speed_stats"](a, gpu=1.0)
    assert spd["chip"] == "A" and spd["n_windows"] == 6 and spd["valid_rate"] == 1.0 and spd["n_unmatched"] == 3
    assert spd["median_s_per_window"] == 1.0 and spd["gpu_share"] == 1.0


def test_metrics_diff_columns(ns):
    import pandas as pd
    base = {"source": "crapii", "split": "dev", "scope": "n/a", "n_docs": 3}
    sa = pd.DataFrame([{**base, **{k: 0.8 for k in ns["DIFF_METRICS"]}}])
    sb = pd.DataFrame([{**base, **{k: 0.5 for k in ns["DIFF_METRICS"]}}])
    d = ns["metrics_diff"](sa, sb)
    assert d["span_recall_diff"].iloc[0] == pytest.approx(0.3) and d["n_docs"].iloc[0] == 3


def test_error_counts_have_no_text(ns):
    docs = {"a": {"doc_id": "a", "source": "crapii", "split": "train", "text": TEXT,
                  "spans": [{**_sp("Ann", "PERSON_PRIVATE", "redact"), "attrs": {}}]}}
    ec = ns["error_counts"]([{"doc_id": "a", "preds": []}], docs)
    assert list(ec.columns) == ["kind", "source", "split", "label", "count"]
    assert ec.iloc[0].to_dict() == {"kind": "miss", "source": "crapii", "split": "train", "label": "PERSON_PRIVATE", "count": 1}


def test_fingerprint_hashes_test_without_parsing(ns, tmp_path):
    (tmp_path / "dev.jsonl").write_text('{"source": "tab"}\n', encoding="utf-8", newline="\n")
    (tmp_path / "train.jsonl").write_text('{"source": "crapii"}\n{"source": "crapii"}\n', encoding="utf-8", newline="\n")
    (tmp_path / "test.jsonl").write_text("this is not json at all\n", encoding="utf-8", newline="\n")
    fp = ns["data_fingerprint"](str(tmp_path))
    assert fp["train.jsonl"]["docs_per_source"] == {"crapii": 2} and fp["test.jsonl"]["docs_per_source"] is None
    assert fp["test.jsonl"]["rows"] == 1 and len(fp["test.jsonl"]["sha256"]) == 64


# ---------------- v0.1 pre-run fixes: frozen prompt, UTF-8 I/O, text fingerprint, hardware, timing ----------------
import ast
import getpass
import hashlib
import re

FROZEN_HASH = "06f049e45980"


def test_prompt_version_and_hash_unchanged(nb, ns):
    src = next(c.source for c in nb.cells if c.cell_type == "code" and "EXPECTED_PROMPT_HASH =" in c.source)
    assert 'PROMPT_VERSION = "v0.1"' in src and f'EXPECTED_PROMPT_HASH = "{FROZEN_HASH}"' in src
    assert ns["prompt_hash"]("standalone") == FROZEN_HASH


def _io_violations(src, name):
    """Text-mode file I/O calls without encoding="utf-8" (and newline="\\n" / lineterminator="\\n" when writing)."""
    bad = []
    const = lambda v: v.value if isinstance(v, ast.Constant) else None
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        kw = {k.arg: const(k.value) for k in node.keywords}
        where = f"{name}:{node.lineno} {fname}()"
        if fname == "open":
            is_method = isinstance(f, ast.Attribute) and not (isinstance(f.value, ast.Name) and f.value.id in ("gzip", "io"))
            idx = 0 if is_method else 1
            mode = const(node.args[idx]) if len(node.args) > idx else kw.get("mode", "r")
            if "b" in mode and "t" not in mode:
                continue
            if kw.get("encoding") != "utf-8":
                bad.append(where + " lacks encoding='utf-8'")
            if any(c in mode for c in "wax") and kw.get("newline") != "\n":
                bad.append(where + " lacks newline='\\n'")
        elif fname == "read_text" and kw.get("encoding") != "utf-8":
            bad.append(where + " lacks encoding='utf-8'")
        elif fname == "write_text" and (kw.get("encoding") != "utf-8" or kw.get("newline") != "\n"):
            bad.append(where + " needs encoding='utf-8', newline='\\n'")
        elif fname == "to_csv" and (kw.get("encoding") != "utf-8" or kw.get("lineterminator") != "\n"):
            bad.append(where + " needs encoding='utf-8', lineterminator='\\n'")
    return bad


def test_all_file_io_is_explicit_utf8(nb):
    bad = []
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            bad += _io_violations(c.source, f"notebook cell {i}")
    for p in sorted(Path(__file__).parent.glob("*.py")):
        bad += _io_violations(p.read_text(encoding="utf-8"), p.name)
    assert bad == []


def test_io_checker_catches_violations():
    assert _io_violations("open('a')\nPath('b').read_text()\nPath('c').write_text('x')\ndf.to_csv('d')", "t") != []
    assert len(_io_violations("open('a')\nPath('b').read_text()\nPath('c').write_text('x')\ndf.to_csv('d')", "t")) == 4
    assert _io_violations("open('a', 'rb')\nopen('a', encoding='utf-8')", "t") == []
    assert _io_violations("open('a', 'w', encoding='utf-8')", "t") != []     # missing newline


def test_utf8_guard_warns_naming_locale_encoding(ns):
    logs = []
    msg = ns["utf8_guard"](log=logs.append, utf8_mode=0, encoding="cp1252")
    text = "\n".join(logs)
    assert "WARNING" in text and "cp1252" in text and "getpreferredencoding" in text and "explicitly UTF-8" in text
    assert msg and "PYTHONUTF8" in msg
    quiet = []
    assert ns["utf8_guard"](log=quiet.append, utf8_mode=1) is None and quiet == []


# non-ASCII fixture: S-cedilla, e-acute, curly quotes
TEXTS = ["Ayşe said “hello” at the café; it’s Şanlıurfa.", "plain ascii doc", "Émile Zola"]


def _write_jsonl(path, docs, ensure_ascii=False):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=ensure_ascii) + "\n")


def _fixture_dir(tmp_path, n=3, ensure_ascii=False):
    docs = [{"doc_id": f"d{i}", "source": "crapii", "text": TEXTS[i % 3] + f" #{i}", "spans": []} for i in range(n)]
    for sp in ("dev", "train", "test"):
        _write_jsonl(tmp_path / f"{sp}.jsonl", docs, ensure_ascii)
    return docs


def test_text_sha1_nonascii_roundtrip_and_first_200(ns, tmp_path):
    docs = _fixture_dir(tmp_path)
    with open(tmp_path / "dev.jsonl", encoding="utf-8") as f:                      # write/read round trip
        back = [json.loads(line) for line in f]
    assert [d["text"] for d in back] == [d["text"] for d in docs] and "Ş" in back[0]["text"] and "é" in back[0]["text"]
    fp = ns["data_fingerprint"](str(tmp_path))
    want = hashlib.sha1("\n".join(d["text"] for d in docs).encode("utf-8")).hexdigest()
    assert fp["dev.jsonl"]["text_sha1"] == fp["train.jsonl"]["text_sha1"] == want
    assert fp["test.jsonl"]["text_sha1"] is None                                    # test is never parsed
    _fixture_dir(tmp_path, ensure_ascii=True)                                       # \uXXXX escapes decode to the same text
    assert ns["data_fingerprint"](str(tmp_path))["dev.jsonl"]["text_sha1"] == want
    assert ns["data_fingerprint"](str(tmp_path))["dev.jsonl"]["sha256"] != fp["dev.jsonl"]["sha256"]   # file bytes differ, text does not
    # only the first 200 docs in file order count
    docs = _fixture_dir(tmp_path, n=250)
    base = ns["data_fingerprint"](str(tmp_path))["dev.jsonl"]["text_sha1"]
    docs[220]["text"] = "changed after doc 200"
    _write_jsonl(tmp_path / "dev.jsonl", docs)
    assert ns["data_fingerprint"](str(tmp_path))["dev.jsonl"]["text_sha1"] == base
    docs[5]["text"] = "changed within the first 200"
    _write_jsonl(tmp_path / "dev.jsonl", docs)
    assert ns["data_fingerprint"](str(tmp_path))["dev.jsonl"]["text_sha1"] != base


def test_compare_fingerprint_warns_on_text_mismatch(ns, tmp_path):
    _fixture_dir(tmp_path)
    fp = ns["data_fingerprint"](str(tmp_path))
    ref = json.loads(json.dumps(fp))
    assert ns["compare_fingerprint"](fp, ref) == []
    ref["dev.jsonl"]["text_sha1"] = "0" * 40
    warns = ns["compare_fingerprint"](fp, ref)
    assert len(warns) == 1 and "dev.jsonl" in warns[0] and "text_sha1" in warns[0]
    old_ref = {k: {"sha256": v["sha256"], "rows": v["rows"]} for k, v in fp.items()}   # reference from before text_sha1 existed
    assert ns["compare_fingerprint"](fp, old_ref) == []
    ref["train.jsonl"]["sha256"] = "x"
    assert any("train.jsonl differs" in w for w in ns["compare_fingerprint"](fp, ref))


def test_real_dev_text_sha1_matches_independent_read(ns):
    path = ROOT / "data" / "processed" / "dev.jsonl"
    if not path.exists():
        pytest.skip("data/processed/dev.jsonl not available")
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() and len(texts) < 200:
                texts.append(json.loads(line)["text"])
    fp = ns["data_fingerprint"](str(path.parent), splits=("dev",))
    assert fp["dev.jsonl"]["text_sha1"] == hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()


# ---------------- hardware ----------------
def _fake_sh(table, log=None):
    def sh(cmd):
        if log is not None:
            log.append(cmd)
        joined = " ".join(cmd)
        return next((v for k, v in table.items() if k in joined), "")
    return sh


NV = "NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB, 551.61"


def test_parse_nvidia_smi(ns):
    assert ns["parse_nvidia_smi"](NV) == {"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "vram_gb": 8, "driver": "551.61"}
    assert ns["parse_nvidia_smi"]("A, 24576 MiB, 535.1\nB, 8192 MiB, 535.1")["vram_gb"] == 24    # first GPU
    assert ns["parse_nvidia_smi"]("") is None and ns["parse_nvidia_smi"](None) is None and ns["parse_nvidia_smi"]("garbage") is None
    assert ns["parse_nvidia_smi"]("X, [N/A], 1.0")["vram_gb"] is None


def test_hw_info_windows_uses_cim_not_wmic(ns):
    cmds = []
    sh = _fake_sh({"Win32_Processor": "13th Gen Intel(R) Core(TM) i7-13700H", "Win32_ComputerSystem": "34089230336",
                   "nvidia-smi": NV}, cmds)
    hw = ns["hw_info"](sh=sh, system="Windows")
    assert hw["chip"] == "13th Gen Intel(R) Core(TM) i7-13700H" and hw["ram_gb"] == 32
    assert hw["gpu"] == {"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "vram_gb": 8, "driver": "551.61"}
    flat = [" ".join(c) for c in cmds]
    assert not any("wmic" in c.lower() for c in flat)
    assert sum("Get-CimInstance" in c and c.lower().startswith("powershell") for c in flat) == 2
    assert any("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader" in c for c in flat)


def test_hw_info_linux_proc_files_and_no_gpu(ns):
    files = {"/proc/cpuinfo": "processor\t: 0\nmodel name\t: AMD Ryzen 7 5800X 8-Core Processor\n",
             "/proc/meminfo": "MemTotal:       16384000 kB\nMemFree: 1 kB\n"}
    hw = ns["hw_info"](sh=_fake_sh({}), system="Linux", reader=files.__getitem__)
    assert hw["chip"] == "AMD Ryzen 7 5800X 8-Core Processor" and hw["ram_gb"] == 16
    assert hw["gpu"] == {"name": None, "vram_gb": None, "driver": None}


def test_hw_info_darwin(ns):
    hw = ns["hw_info"](sh=_fake_sh({"brand_string": "Apple M1 Pro", "hw.memsize": str(16 * 2**30)}), system="Darwin")
    assert hw["chip"] == "Apple M1 Pro" and hw["ram_gb"] == 16 and hw["gpu"]["name"] is None


def test_hw_info_override_wins_and_keeps_driver(ns):
    sh = _fake_sh({"Win32_Processor": "cpu", "Win32_ComputerSystem": str(8 * 2**30), "nvidia-smi": NV})
    ov = {"cpu": "Ryzen X", "ram_gb": 64, "gpu": "RTX 5090", "vram_gb": 32, "os": "Windows 11"}
    hw = ns["hw_info"](ov, sh=sh, system="Windows")
    assert (hw["chip"], hw["ram_gb"], hw["os"]) == ("Ryzen X", 64, "Windows 11")
    assert hw["gpu"] == {"name": "RTX 5090", "vram_gb": 32, "driver": "551.61"}
    none_ov = {"cpu": None, "ram_gb": None, "gpu": None, "vram_gb": None, "os": None}
    assert ns["hw_info"](none_ov, sh=sh, system="Windows")["ram_gb"] == 8


def test_hw_info_real_machine_has_no_identifiers(ns):
    hw = ns["hw_info"]()
    assert set(hw) == {"os", "chip", "ram_gb", "gpu", "python"} and set(hw["gpu"]) == {"name", "vram_gb", "driver"}
    blob = json.dumps(hw)
    for secret in (socket.gethostname(), getpass.getuser(), str(Path.home())):
        assert len(secret) < 4 or secret not in blob


def test_machine_tag_label_and_sanitised_default(ns):
    hw = {"chip": "Intel(R) Core(TM) i7-13700H @ 2.40GHz", "ram_gb": 32}
    tag = ns["machine_tag"](hw)
    assert tag == "intel_r_core_tm_i7_13700h_2_40ghz_32gb" and re.fullmatch(r"[a-z0-9_]+", tag)
    assert ns["machine_tag"](hw, "Win RTX4070!") == "win_rtx4070" and ns["machine_tag"]({"chip": "Apple M1 Pro", "ram_gb": 16}) == "apple_m1_pro_16gb"
    assert ns["machine_tag"]({"chip": "x", "ram_gb": None}) == "x"
    stamp = ns["machine_stamp"]({**hw, "gpu": {"name": "G"}, "label": "L"}, "0.1")
    assert stamp["gpu"] == "G" and stamp["label"] == "L" and "host" not in json.dumps(stamp).lower()


# ---------------- timing fields ----------------
def test_client_records_timing_fields_from_final_object(ns):
    def good(h, stop):
        _line(h, {"message": {"content": '{"spans": []}'}, "done": False})
        _line(h, {"message": {"content": ""}, "done": True, "done_reason": "stop", "eval_count": 7, "eval_duration": 1_000_000_000,
                  "load_duration": 2_000_000_000, "prompt_eval_duration": 500_000_000, "total_duration": 4_000_000_000,
                  "prompt_eval_count": 120})

    s = _Server(good)
    try:
        r = ns["chat"]("m", [], {}, s.url)
    finally:
        s.close()
    assert (r["prompt_eval_duration_ns"], r["total_duration_ns"], r["prompt_eval_count"]) == (500_000_000, 4_000_000_000, 120)


def _timed(lat=3.0, gen=1.0, pe=0.5, load=0.2, total=2.0, n=100):
    return {"valid_json": True, "truncated": False, "parsed": {"spans": []}, "latency_s": lat, "tokens_per_s": 5.0, "eval_count": 3,
            "eval_duration_ns": gen * 1e9, "prompt_eval_duration_ns": pe * 1e9, "load_duration_ns": load * 1e9,
            "total_duration_ns": total * 1e9, "prompt_eval_count": n, "error": None}


def test_detect_doc_window_and_row_timing(ns):
    text = "Alice is here. " * 40
    nw = len(ns["make_windows"](text, 200, 60))
    row = ns["detect_doc"]({"doc_id": "a", "text": text}, lambda m, s: _timed(), window=200, overlap=60)
    w = row["windows"][0]
    assert (w["prompt_eval_s"], w["load_s"], w["total_s"], w["prompt_eval_count"], w["gen_s"]) == (0.5, 0.2, 2.0, 100, 1.0)
    assert row["prompt_eval_s"] == pytest.approx(0.5 * nw) and row["load_s"] == pytest.approx(0.2 * nw)
    assert row["total_s"] == pytest.approx(2.0 * nw) and row["prompt_eval_count"] == 100 * nw
    old = ns["detect_doc"]({"doc_id": "a", "text": text}, lambda m, s: _ok([]), window=200, overlap=60)   # server sent none
    assert old["prompt_eval_s"] is None and old["windows"][0]["prompt_eval_s"] is None and old["total_s"] is None


def test_latency_stats_timing_medians_and_overhead(ns):
    w = lambda lat, gen, pe: {"latency_s": lat, "tokens_per_s": 5.0, "valid": True, "truncated": False, "load_s": 0.0,
                              "gen_s": gen, "prompt_eval_s": pe, "total_s": lat - 0.5, "prompt_eval_count": 100}
    rows = [{"windows": [w(30.0, 20.0, 9.0), w(4.0, 2.0, 1.0), w(6.0, 3.0, 2.0), w(8.0, 5.0, 2.0)]}]
    st = ns["latency_stats"](rows)                       # first call excluded
    assert st["median_gen_s"] == 3.0 and st["median_prompt_eval_s"] == 2.0 and st["median_total_s"] == 5.5
    assert st["median_overhead_s"] == 1.0 and st["median_prompt_eval_count"] == 100 and st["median_latency_s"] == 6.0
    json.loads(json.dumps(st))                           # goes into the system-check JSON


def test_timing_absent_in_reference_or_old_rows_still_works(ns):
    old = [_row("d1", [_p("Ann")]), _row("d2", [])]      # windows without any of the new fields
    st = ns["latency_stats"](old)
    for k in ("median_gen_s", "median_prompt_eval_s", "median_overhead_s"):
        assert math.isnan(st[k])
    spd = ns["speed_stats"](old)
    assert math.isnan(spd["median_overhead_s"]) and spd["median_s_per_window"] == 1.0
    new = [{**_row("d3", []), "windows": [{"latency_s": 3.0, "tokens_per_s": 9.0, "valid": True, "truncated": False, "load_s": 0.0,
                                           "gen_s": 1.0, "prompt_eval_s": 0.5}] * 3}]
    spd = ns["speed_stats"](new)
    assert spd["median_gen_s"] == 1.0 and spd["median_prompt_eval_s"] == 0.5 and spd["median_overhead_s"] == 1.5
    import pandas as pd
    rs = {"median_latency_s": 2.0}                       # a reference stats block from before the timing fields
    tbl = pd.DataFrame({"this": [3.0, 1.0], "reference": [rs.get("median_latency_s"), rs.get("median_gen_s")]}).astype(float)
    assert math.isnan(tbl.loc[1, "reference"])


# ---------------- repetition guard, recovery, tolerant matching ----------------
def _obj(t, label="ORG", dec="KEEP"):
    return json.dumps({"text": t, "label": label, "decision": dec})


def _send(h, s):
    _line(h, {"message": {"content": s}, "done": False})


def test_repeating_stream_aborts_early_and_salvages(ns):
    def looping(h, stop):
        _send(h, '{"spans": [')
        _send(h, _obj("Acme") + ",")
        _send(h, _obj("Bob", "PERSON_PRIVATE", "REDACT") + ",")
        while not stop.is_set():
            _send(h, _obj("Acme") + ",")

    s = _Server(looping)
    try:
        r = ns["chat"]("m", [], {}, s.url, timeout=20, num_predict=900)
        assert r["loop"] and r["cause"] == "loop" and r["error"] == "loop" and not r["truncated"] and not r["valid_json"]
        assert r["eval_count"] < 30                                   # far below the 900 cap
        assert [o["text"] for o in r["objects"]] == ["Acme", "Bob"]   # unique objects kept
    finally:
        s.close()


def test_repeat_total_threshold_not_consecutive(ns):
    def spaced(h, stop):
        _send(h, '{"spans": [' + _obj("A") + ",")
        for i in range(10):                                           # repeat, unique, repeat, unique, ...
            _send(h, _obj("A") + ",")
            _send(h, _obj(f"u{i}") + ",")
        _send(h, "]}")
        _line(h, {"message": {"content": ""}, "done": True, "done_reason": "stop"})

    s = _Server(spaced)
    try:
        r = ns["chat"]("m", [], {}, s.url, timeout=20)
        assert r["loop"] and r["repeats"] == 5 and len(r["objects"]) == 5   # 5 in total, never 3 in a row
    finally:
        s.close()


def test_dense_stream_hits_cap_labelled_dense(ns):
    def dense(h, stop):
        _send(h, '{"spans": [')
        i = 0
        while not stop.is_set():
            _send(h, _obj(f"item{i}") + ",")
            i += 1

    s = _Server(dense)
    try:
        r = ns["chat"]("m", [], {}, s.url, timeout=20, num_predict=40)
        assert r["truncated"] and r["cause"] == "dense" and not r["loop"] and r["repeats"] == 0
        assert len(r["objects"]) >= 30 and len({o["text"] for o in r["objects"]}) == len(r["objects"])
    finally:
        s.close()


def test_normal_stream_cause_none_and_escapes_parse(ns):
    def good(h, stop):
        _send(h, '{"spans": [{"text": "say \\"hi\\" {x}", "label": "ORG", "decision": "KEEP"}')
        _send(h, "]}")
        _line(h, {"message": {"content": ""}, "done": True, "done_reason": "stop"})

    s = _Server(good)
    try:
        r = ns["chat"]("m", [], {}, s.url)
        assert r["valid_json"] and r["cause"] == "none" and not r["loop"] and r["repeats"] == 0
        assert r["objects"][0]["text"] == 'say "hi" {x}' and r["parsed"]["spans"][0]["text"] == 'say "hi" {x}'
    finally:
        s.close()


TXT = "Alice Smith lives here. " * 3 + "Then Bob Jones left town. " * 3


def _trunc(objs, cause="dense", loop=False):
    return {"valid_json": False, "truncated": not loop, "loop": loop, "cause": cause, "parsed": None,
            "objects": objs, "raw": "partial" * 500, "error": "loop" if loop else "truncated", "eval_count": 900}


def _o(t, dec="REDACT", label="PERSON_PRIVATE"):
    return {"text": t, "label": label, "decision": dec}


def _wtext(messages):
    return messages[-1]["content"][len("TEXT:\n"):]


def test_truncated_window_salvages_then_two_halves_offsets_map_back(ns):
    calls = []

    def fn(messages, schema):
        wt = _wtext(messages)
        calls.append(wt)
        if len(calls) == 1:
            return _trunc([_o("Alice Smith"), _o("Alice Smith"), _o("Ghost")])
        return _ok([_o("Bob Jones")] if "Bob" in wt else [_o("Alice Smith")])

    row = ns["detect_doc"]({"doc_id": "d", "text": TXT}, fn, 5000, 100)
    assert len(calls) == 3 and calls[1] + calls[2] == TXT and calls[1].endswith(" ")   # sentence boundary near the middle
    assert abs(len(calls[1]) - len(TXT) / 2) < 30
    assert all(TXT[p["start"]:p["end"]] == p["text"] for p in row["preds"])
    texts = [p["text"] for p in row["preds"]]
    assert texts.count("Alice Smith") == 3 and texts.count("Bob Jones") == 3 and len(row["preds"]) == 6   # union, no duplicates
    w0, h1, h2 = row["windows"]
    assert (w0["cause"], w0["n_spans_salvaged"], w0["split"], w0["failed_after_split"]) == ("dense", 2, True, False)
    assert (h1["half"], h2["half"]) == (1, 2) and not h1["truncated"]
    assert w0["unmatched"] == ["Ghost"] and len(w0["raw_partial"]) == 2000
    assert row["n_truncated"] == 1 and row["n_split"] == 1 and row["n_failed_after_split"] == 0 and row["n_salvaged"] == 2


def test_loop_abort_salvages_and_splits(ns):
    calls = []

    def fn(messages, schema):
        calls.append(1)
        return _trunc([_o("Alice Smith")], "loop", loop=True) if len(calls) == 1 else _ok([])

    row = ns["detect_doc"]({"doc_id": "d", "text": TXT}, fn, 5000, 100)
    w0 = row["windows"][0]
    assert len(calls) == 3 and w0["cause"] == "loop" and not w0["truncated"] and w0["split"] and row["n_loop"] == 1
    assert row["n_truncated"] == 0 and row["n_invalid"] == 0 and len(row["preds"]) == 3   # salvaged Alice x3


def test_stream_failing_again_after_split_terminates_with_flag(ns):
    calls = []

    def fn(messages, schema):
        calls.append(_wtext(messages))
        return _trunc([_o("Alice Smith")] if "Alice" in calls[-1] else [_o("Bob Jones")])

    row = ns["detect_doc"]({"doc_id": "d", "text": TXT}, fn, 5000, 100)
    assert len(calls) == 3                                    # window + two halves, no further splitting
    w0, h1, h2 = row["windows"]
    assert w0["failed_after_split"] and w0["split"] and h1["cause"] == h2["cause"] == "dense"
    assert h1["n_spans_salvaged"] == 1 and row["n_failed_after_split"] == 1 and row["n_salvaged"] == 3
    assert {p["text"] for p in row["preds"]} == {"Alice Smith", "Bob Jones"}
    rc = ns["recovery_counts"]([{**row, "source": "crapii"}])["crapii"]
    assert (rc["windows"], rc["dense"], rc["failed"], rc["recovered"]) == (1, 1, 1, 0)


def test_tolerant_matching_newline_quotes_and_offsets(ns):
    full = "xx " + "Mary\n   Jane  went to O’Neil’s “place” today."
    win, off = full[3:], 3
    items = [{"text": "Mary Jane", "label": "PERSON_PRIVATE", "decision": "REDACT"},
             {"text": "O'Neil's", "label": "ORG", "decision": "KEEP"},
             {"text": '"place"', "label": "LOCATION_OTHER", "decision": "KEEP"},
             {"text": "Nobody", "label": "ORG", "decision": "KEEP"}]
    preds, unm = ns["match_strings_ex"](win, items, off)
    got = {p["text"]: (p["start"], p["end"]) for p in preds}
    assert full[slice(*got["Mary\n   Jane"])] == "Mary\n   Jane"
    assert full[slice(*got["O’Neil’s"])] == "O’Neil’s" and full[slice(*got["“place”"])] == "“place”"
    assert unm == ["Nobody"] and ns["match_strings"](win, items, off)[1] == 1
    assert ns["match_strings"]("Annabel Ann", [{"text": "Ann", "label": "X", "decision": "KEEP"}])[0][0]["start"] == 8   # still word-bounded


def test_normal_window_unchanged_and_pipeline_version(ns, tmp_path):
    text = "Alice is here. " * 4
    replies = [_ok([_o("Alice"), _o("Zed", "KEEP", "ORG")])]
    row = ns["detect_doc"]({"doc_id": "d", "text": text}, _fake(replies), 5000, 100)
    assert row["n_windows"] == 1 and len(row["windows"]) == 1 and row["n_split"] == 0 and row["n_loop"] == row["n_dense"] == 0
    w = row["windows"][0]
    assert (w["cause"], w["split"], w["n_spans_salvaged"], w["unmatched"]) == ("none", False, 0, ["Zed"]) and "raw_partial" not in w
    assert row["n_unmatched"] == 1 and len(row["preds"]) == 4
    meta = {"prompt_hash": "h", "model_digest": "d"}
    out = tmp_path / "r.jsonl"
    ns["run_detection"]([{"doc_id": "d", "text": text, "source": "x"}], out, _fake(replies), meta, log=lambda *a: None)
    rows = ns["load_jsonl"](out)
    assert rows[0]["pipeline_version"] == "p1"
    old = {**rows[0], "pipeline_version": None}
    out.write_text(json.dumps(old) + "\n", encoding="utf-8", newline="\n")
    assert ns["load_rows"](out, meta, "standalone", 1200, 100) == {}      # old rows are never reused


# ---------------- miss taxonomy ----------------
TAX_TEXT = ("Ann met Bob. Dr. Smith and Mrs. Jones Lee arrived. Zed called. Pat Roe left. "
            "Cy and Di went. Cy again. Eve Fox here.")


def _tg(sub, label="PERSON_PRIVATE", nth=0):
    i = -1
    for _ in range(nth + 1):
        i = TAX_TEXT.index(sub, i + 1)
    return {"start": i, "end": i + len(sub), "label": label, "attrs": {}}


def _tp(sub, decision="REDACT", nth=0):
    g = _tg(sub, nth=nth)
    return {"start": g["start"], "end": g["end"], "text": sub, "decision": decision}


def _tax(ns, spans, preds):
    doc = {"doc_id": "d", "source": "crapii", "split": "dev", "text": TAX_TEXT, "spans": spans}
    return ns["classify_gold_redact"]([{"doc_id": "d", "preds": preds}], {"d": doc})


def test_taxonomy_each_category_and_first_match_wins(ns):
    spans = [_tg("Ann"), _tg("Bob"), _tg("Dr. Smith"), _tg("Jones Lee"), _tg("Zed"), _tg("Pat Roe", "EMAIL"), _tg("Eve Fox")]
    preds = [_tp("Ann"), _tp("Bob", "KEEP"), _tp("Smith"), _tp("Lee"), _tp("Eve"), _tp("Fox", "KEEP")]
    recs = {r["text"]: r for r in _tax(ns, spans, preds)}
    assert recs["Ann"]["category"] == "covered"
    assert recs["Bob"]["category"] == "predicted_keep"
    r = recs["Dr. Smith"]
    assert (r["category"], r["shape"], r["title"]) == ("partial_redact", "narrower", "gold_title_uncovered")
    r = recs["Jones Lee"]
    assert (r["category"], r["shape"], r["title"]) == ("partial_redact", "narrower", "")
    assert recs["Zed"]["category"] == "never_returned" and recs["Pat Roe"]["category"] == "never_returned"
    assert recs["Eve Fox"]["category"] == "predicted_keep"        # REDACT partial + KEEP overlap: first match wins
    assert all(len(r["context"]) <= 120 for r in recs.values())


def test_taxonomy_wider_shape_and_pred_title(ns):
    s = TAX_TEXT.index("Mrs. Jones")
    wide = {"start": s, "end": s + len("Mrs. Jones L"), "decision": "REDACT"}      # extends past the gold span, still partial
    r = _tax(ns, [_tg("Jones Lee")], [wide])[0]
    assert (r["category"], r["shape"], r["title"]) == ("partial_redact", "wider", "pred_title_extra")


def test_taxonomy_tab_scope_direct_ignores_quasi(ns):
    spans = [{**_tg("Ann"), "attrs": {"identifier_type": "DIRECT"}}, {**_tg("Bob"), "attrs": {"identifier_type": "QUASI"}}]
    doc = {"doc_id": "d", "source": "tab", "split": "dev", "text": TAX_TEXT, "spans": spans}
    recs = ns["classify_gold_redact"]([{"doc_id": "d", "preds": []}], {"d": doc}, "direct")
    assert [r["text"] for r in recs] == ["Ann"] and recs[0]["category"] == "never_returned"


def test_taxonomy_tables_counts_shares_and_person_recall(ns):
    spans = [_tg("Ann"), _tg("Bob"), _tg("Dr. Smith"), _tg("Zed", "EMAIL"), _tg("Cy", nth=0), _tg("Cy", nth=1), _tg("Pat Roe")]
    preds = [_tp("Ann"), _tp("Bob", "KEEP"), _tp("Smith"), _tp("Cy", nth=0), _tp("Pat Roe")]
    recs = _tax(ns, spans, preds)
    tt = ns["taxonomy_table"](recs).set_index("label")
    assert tt.loc["ALL", "n"] == 7 and tt.loc["ALL", "covered"] == 3          # Ann, first Cy, Pat Roe
    assert tt.loc["PERSON_PRIVATE", "n"] == 6 and tt.loc["EMAIL", "never_returned"] == 1
    assert tt.loc["ALL", "predicted_keep"] == 1 and tt.loc["ALL", "partial_redact"] == 1 and tt.loc["ALL", "never_returned"] == 2
    assert tt.loc["ALL", "never_returned_share"] == pytest.approx(2 / 7)
    pr = ns["person_recall_table"](recs).set_index(["tokens", "mentions"])
    assert pr.loc[("single", "several"), "n"] == 2 and pr.loc[("single", "several"), "recall"] == pytest.approx(0.5)
    assert pr.loc[("multi", "one"), "n"] == 2 and pr.loc[("multi", "one"), "covered"] == 1      # Dr. Smith partial, Pat Roe ok
    assert pr.loc[("all", "all"), "n"] == 6
    ex = ns["taxonomy_examples"](recs, k=1)
    assert set(ex.category) <= set(ns["MISS_CATEGORIES"]) and ex.groupby("category").size().max() == 1
    assert len(ns["partial_shape_table"](recs)) == 1
