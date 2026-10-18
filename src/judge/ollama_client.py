"""Minimal Ollama /api/chat client (stdlib only): streams, caps tokens, enforces a wall-clock deadline."""
import http.client
import json
import socket
import threading
import time
import urllib.parse

URL = "http://localhost:11434/api/chat"
REQUEST_TIMEOUT = 120  # seconds, wall clock per request (connect + generation)


def _stream(body, url, timeout, t0, num_predict, out):
    """Stream one request into `out`; raises OSError subclasses on connection failures."""
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    state = {"timed_out": False}
    sock = []  # http.client drops conn.sock for close-delimited responses, so keep our own reference

    def kill():  # watchdog: abort the connection at the deadline, even if the server sends nothing
        state["timed_out"] = True
        try:
            sock[0].shutdown(socket.SHUT_RDWR)
        except (OSError, IndexError):
            pass
        conn.close()

    parts, chunks = [], 0
    try:
        conn.connect()
        sock.append(conn.sock)
        timer = threading.Timer(max(timeout - (time.perf_counter() - t0), 0.01), kill)
        timer.daemon = True
        timer.start()
        try:
            conn.request("POST", u.path or "/", json.dumps(body).encode(), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                out["error"] = f"HTTP {resp.status}: {resp.reason}"
                return
            while True:
                line = resp.readline()
                if not line:
                    if not state["timed_out"] and out["done_reason"] is None:
                        out["error"] = "incomplete stream"
                    break
                obj = json.loads(line)
                if obj.get("error"):
                    out["error"] = f"server: {obj['error']}"
                    break
                parts.append((obj.get("message") or {}).get("content") or "")
                chunks += 1
                if obj.get("done"):
                    out["done_reason"] = obj.get("done_reason") or "stop"
                    out["eval_count"], out["eval_duration_ns"] = obj.get("eval_count"), obj.get("eval_duration")
                    break
                if chunks > num_predict:  # a chunk is >= 1 token: the server ignored the cap
                    out["done_reason"] = "length"
                    break
        except (OSError, ValueError, http.client.HTTPException) as e:
            state["timed_out"] |= isinstance(e, TimeoutError)  # socket timeout racing the watchdog
            if not state["timed_out"]:
                out["error"] = f"{type(e).__name__}: {e}"
        finally:
            timer.cancel()
    finally:
        conn.close()
    out["raw"] = "".join(parts) or None
    if out["eval_count"] is None:
        out["eval_count"] = chunks
    if state["timed_out"]:
        out["error"] = "timeout"
    elif out["done_reason"] == "length":
        out["truncated"], out["error"] = True, "truncated"


def chat(model, messages, schema, num_ctx=4096, seed=42, timeout=REQUEST_TIMEOUT, num_predict=300, url=URL):
    """Return {raw, parsed, valid_json, latency_s, eval_count, eval_duration_ns, error, done_reason, truncated}.
    No retry on timeout, truncation or bad JSON; one retry only on a refused connection."""
    body = {
        "model": model, "messages": messages, "format": schema, "stream": True, "think": False,
        "keep_alive": "5m", "options": {"temperature": 0, "num_ctx": num_ctx, "seed": seed, "num_predict": num_predict},
    }
    out = {"raw": None, "parsed": None, "valid_json": False, "latency_s": None, "eval_count": None,
           "eval_duration_ns": None, "error": None, "done_reason": None, "truncated": False}
    t0 = time.perf_counter()
    for attempt in (0, 1):
        try:
            _stream(body, url, timeout, t0, num_predict, out)
            break
        except ConnectionRefusedError as e:
            out["error"] = f"{type(e).__name__}: {e}"
        except (OSError, http.client.HTTPException) as e:  # connect timeout, DNS, reset
            out["error"] = "timeout" if isinstance(e, TimeoutError) else f"{type(e).__name__}: {e}"
            break
    out["latency_s"] = time.perf_counter() - t0
    if out["error"] is None and out["raw"] is not None:
        try:
            parsed = json.loads(out["raw"])
            if isinstance(parsed, dict):
                out["parsed"], out["valid_json"] = parsed, True
        except ValueError:
            pass
    return out
