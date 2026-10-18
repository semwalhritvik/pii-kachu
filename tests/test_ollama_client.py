import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.judge.ollama_client import chat


def make_server(handler_fn):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            try:
                handler_fn(self)
            except OSError:
                pass

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/api/chat"


def line(w, obj):
    w.wfile.write(json.dumps(obj).encode() + b"\n")
    w.wfile.flush()


@pytest.fixture
def serve():
    servers = []

    def start(fn):
        srv, url = make_server(fn)
        servers.append(srv)
        return url
    yield start
    for s in servers:
        s.shutdown()


def endless(w):
    while True:
        line(w, {"message": {"content": "x"}, "done": False})
        time.sleep(0.005)


def stall(w):
    time.sleep(10)


def test_endless_stream_stops_at_token_cap(serve):
    out = chat("m", [], {}, num_predict=20, timeout=30, url=serve(endless))
    assert out["error"] == "truncated" and out["truncated"] and not out["valid_json"]
    assert out["done_reason"] == "length" and out["latency_s"] < 10


def test_endless_stream_stops_at_deadline(serve):
    out = chat("m", [], {}, num_predict=10**9, timeout=1, url=serve(endless))
    assert out["error"] == "timeout" and not out["valid_json"] and 0.9 < out["latency_s"] < 5


def test_stalled_server_stops_at_deadline(serve):
    out = chat("m", [], {}, timeout=1, url=serve(stall))
    assert out["error"] == "timeout" and out["latency_s"] < 5


def test_server_reported_length_is_truncated_even_if_json_parses(serve):
    def fn(w):
        line(w, {"message": {"content": "{}"}, "done": True, "done_reason": "length", "eval_count": 5})
    out = chat("m", [], {}, url=serve(fn))
    assert out["error"] == "truncated" and not out["valid_json"] and out["eval_count"] == 5


def test_normal_stream_and_invalid_json_not_retried(serve):
    calls = []

    def fn(w):
        calls.append(1)
        line(w, {"message": {"content": '{"a":'}, "done": False})
        line(w, {"message": {"content": " 1}"}, "done": True, "done_reason": "stop", "eval_count": 2})
    out = chat("m", [], {}, url=serve(fn))
    assert out["parsed"] == {"a": 1} and out["valid_json"] and out["error"] is None and out["done_reason"] == "stop"

    def bad(w):
        calls.append(1)
        line(w, {"message": {"content": "not json"}, "done": True, "done_reason": "stop"})
    calls.clear()
    out = chat("m", [], {}, url=serve(bad))
    assert not out["valid_json"] and len(calls) == 1


def test_refused_connection_returns_error_row():
    out = chat("m", [], {}, timeout=5, url="http://127.0.0.1:9/api/chat")
    assert out["error"] and not out["valid_json"]
