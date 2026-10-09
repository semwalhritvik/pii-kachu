import pytest

from src.ingest import validate as v
from src.ingest.labels import map_label


def doc(spans, doc_id="a", text="hello world"):
    return {"doc_id": doc_id, "source": "x", "text": text, "spans": spans, "split": "train"}


def sp(s, e, label="EMAIL"):
    return {"start": s, "end": e, "label": label, "attrs": {}}


def test_ok():
    assert v.check_doc(doc([sp(0, 5), sp(6, 11)])) == []


def test_nested_passes():
    d = doc([sp(0, 11), sp(0, 5), sp(0, 5), sp(6, 11)])
    assert v.check_doc(d) == []
    assert v.pair_stats(d)[:2] == (4, 0)


def test_crossing_fails():
    d = doc([sp(0, 6), sp(5, 11)])
    assert any("crossing" in e for e in v.check_doc(d))


def test_bad_label():
    assert any("bad label" in e for e in v.check_doc(doc([sp(0, 5, "NOPE")])))


def test_bad_offsets():
    assert any("offsets" in e for e in v.check_doc(doc([sp(0, 50)])))


def test_split_leak():
    assert v.check_splits({"train": [doc([])], "dev": [doc([])]})


def test_unmapped_raises():
    with pytest.raises(KeyError):
        map_label("crapii", "WHATEVER")
    with pytest.raises(KeyError):
        map_label("tab", "PERSON", "BOGUS")


def test_processed_data():
    import os
    if not os.path.exists("data/processed/test.jsonl"):
        pytest.skip("no processed data")
    by = v.load()
    assert v.check_splits(by) == []


def test_allowlisted_crossing_passes():
    d = doc([sp(0, 6), sp(5, 11)])
    allow = {("x", "a", ((0, 6), (5, 11)))}
    allowed = []
    assert v.check_doc(d, allow, allowed) == []
    assert len(allowed) == 1


def test_unlisted_crossing_fails():
    d = doc([sp(0, 6), sp(5, 11)])
    allow = {("x", "a", ((0, 7), (5, 11)))}
    assert any("crossing" in e for e in v.check_doc(d, allow))
