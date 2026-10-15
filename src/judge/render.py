"""Render the redacted text of a case from its span actions."""
import re

DETECTOR = {
    "PERSON_PRIVATE": "PERSON", "PERSON_PUBLIC": "PERSON", "DOB": "DATE", "DATE_OTHER": "DATE",
    "LOCATION_OTHER": "LOCATION", "STREET_ADDRESS": "LOCATION", "ID_NUM": "ID", "URL_PERSONAL": "URL",
    "OTHER_PII": "USERNAME", "EMAIL": "EMAIL", "PHONE": "PHONE", "ORG": "ORG", "DEMOGRAPHIC": "OTHER",
}


def locate(case):
    """Set start/end on every span: each takes the first free occurrence of its text (longest spans claim first)."""
    text, taken = case["original"], []
    for s in sorted(case["spans"], key=lambda s: -len(s["text"])):
        for m in re.finditer(re.escape(s["text"]), text):
            if all(m.end() <= a or m.start() >= b for a, b in taken):
                s["start"], s["end"] = m.start(), m.end()
                taken.append((m.start(), m.end()))
                break
        else:
            raise ValueError(f"{case['case_id']}/{s['id']}: no free occurrence of {s['text']!r}")


def _entity(s):
    return s.get("entity") or s["text"].casefold()


def replacements(case):
    """span id -> replacement string; entity numbering follows text order."""
    spans = sorted(case["spans"], key=lambda s: s["start"])
    red, pse, out = {}, {}, {}
    for s in spans:
        a, key = s["action"], (s["detector_label"], _entity(s))
        if a == "keep" or s.get("detected", True) is False:
            out[s["id"]] = s["text"]
        elif a == "redact":
            n = red.setdefault(key, 1 + sum(k[0] == key[0] for k in red))
            out[s["id"]] = f"<{s['detector_label']}_{n}>"
        elif a == "pseudonymize":
            n = pse.setdefault(_entity(s), len(pse) + 1)
            out[s["id"]] = f"P-{n:04d}"
        else:
            out[s["id"]] = s["replacement"]
    return out


def render(case):
    rep, text = replacements(case), case["original"]
    for s in sorted(case["spans"], key=lambda s: -s["start"]):
        text = text[: s["start"]] + rep[s["id"]] + text[s["end"]:]
    return text
