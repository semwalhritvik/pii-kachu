import json

import pytest

from src.judge.derive import ROOT

CASES = ROOT / "pilot/pilot_cases.jsonl"
SPLITS = ROOT / "pilot/splits.json"


def require(path):
    if not path.exists():
        pytest.fail(f"missing {path.relative_to(ROOT)}: pilot files are untracked, restore them (see README 'LLM judge experiments')")


def test_pilot_files_exist():
    require(CASES)
    require(SPLITS)


def test_split_ids_exist_in_pilot_file():
    require(CASES)
    require(SPLITS)
    ids = {json.loads(l)["case_id"] for l in CASES.read_text().splitlines() if l.strip()}
    sp = json.loads(SPLITS.read_text())
    missing = [i for k in ("dev", "test") for i in sp[k] if i not in ids]
    assert not missing, f"case_ids in splits.json but not in pilot_cases.jsonl: {missing}"
    assert not set(sp["dev"]) & set(sp["test"]), "dev and test overlap"
