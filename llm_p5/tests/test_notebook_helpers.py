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
    return nbformat.read(NB, as_version=4)


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
    replies = bad + [_ok([{"text": "Alice", "label": "PERSON_PRIVATE", "decision": "REDACT"}])] * (n_win - 2)
    row = ns["detect_doc"]({"doc_id": "d", "text": text}, _fake(replies), 200, 60)
    assert (row["n_truncated"], row["n_invalid"]) == (1, 1)
    assert row["errors"] == {"truncated": 1}
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
    with open(path) as f:
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
    saved = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(saved) == 3
    for r in saved:
        assert r["prompt_version"] == "v0.1" and r["prompt_modified"] is False
        assert set(r["machine"]) == {"chip", "ram_gb", "os", "ollama_version"}
        assert r["sample_fingerprint"] == ns["sample_fingerprint"](docs) and r["prompt_hash"] == "h"
    rows2 = ns["run_detection"](docs, path, chat, _meta(), log=lambda *_: None)   # resume
    assert len(rows2) == 6 and len(calls) == 6                      # only the 3 remaining docs were run
    assert len(path.read_text().splitlines()) == 6
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
    assert json.loads((tmp_path / "r.jsonl").read_text())["prompt_modified"] is True


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
    (tmp_path / "dev.jsonl").write_text('{"source": "tab"}\n')
    (tmp_path / "train.jsonl").write_text('{"source": "crapii"}\n{"source": "crapii"}\n')
    (tmp_path / "test.jsonl").write_text("this is not json at all\n")
    fp = ns["data_fingerprint"](str(tmp_path))
    assert fp["train.jsonl"]["docs_per_source"] == {"crapii": 2} and fp["test.jsonl"]["docs_per_source"] is None
    assert fp["test.jsonl"]["rows"] == 1 and len(fp["test.jsonl"]["sha256"]) == 64
