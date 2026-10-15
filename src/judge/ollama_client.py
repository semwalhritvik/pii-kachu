"""Minimal Ollama /api/chat client (stdlib only)."""
import json
import time
import urllib.error
import urllib.request

URL = "http://localhost:11434/api/chat"


def _post(body, timeout):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(model, messages, schema, num_ctx=4096, seed=42, timeout=300):
    body = {
        "model": model, "messages": messages, "format": schema, "stream": False, "think": False,
        "keep_alive": "5m", "options": {"temperature": 0, "num_ctx": num_ctx, "seed": seed},
    }
    out = {"raw": None, "parsed": None, "valid_json": False, "latency_s": None,
           "eval_count": None, "eval_duration_ns": None, "error": None}
    t0, resp = time.perf_counter(), None
    for attempt in (0, 1):
        try:
            resp = _post(body, timeout)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError):
                out["error"] = f"HTTP {e.code}: {e.reason}"
                break
            out["error"] = f"{type(e).__name__}: {e}"
        except (OSError, ValueError) as e:
            out["error"] = f"{type(e).__name__}: {e}"
            break
    out["latency_s"] = time.perf_counter() - t0
    if resp is None:
        return out
    out["error"] = None
    out["raw"] = (resp.get("message") or {}).get("content")
    out["eval_count"] = resp.get("eval_count")
    out["eval_duration_ns"] = resp.get("eval_duration")
    try:
        parsed = json.loads(out["raw"])
        if isinstance(parsed, dict):
            out["parsed"], out["valid_json"] = parsed, True
    except (TypeError, ValueError):
        pass
    return out
