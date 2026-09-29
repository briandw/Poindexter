import hashlib
from pathlib import Path

import pytest

from poindexter import datasets as D
from poindexter.contract import read_records

FIXTURES = Path(__file__).parent / "fixtures"
SQUAD = FIXTURES / "squad_30.json"
HOTPOT = FIXTURES / "hotpot_30.json"


def sentences(text):
    return [s for _, _, s in D.sentence_split(text)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello world. This is a test.", ["Hello world.", "This is a test."]),
        ("Mr. Smith went home. He slept.", ["Mr. Smith went home.", "He slept."]),
        ("Dr. Jones and Mrs. Jones arrived.", ["Dr. Jones and Mrs. Jones arrived."]),
        ("The U.S. Army fought. It won.", ["The U.S. Army fought.", "It won."]),
        ("Fruits, e.g. Apples, are sweet.", ["Fruits, e.g. Apples, are sweet."]),
        ("Cities, i.e. Paris, grew.", ["Cities, i.e. Paris, grew."]),
        ("It grew 3.5 percent. Then it fell.", ["It grew 3.5 percent.", "Then it fell."]),
        ("He wore No. 5 for years. Fans cheered.", ["He wore No. 5 for years.", "Fans cheered."]),
        ("Is it? Yes! It is.", ["Is it?", "Yes!", "It is."]),
        ('He said "Stop." Then he left.', ['He said "Stop."', "Then he left."]),
        ("It ended in 1945. 1946 was calm.", ["It ended in 1945.", "1946 was calm."]),
        ("John F. Kennedy won. He served.", ["John F. Kennedy won.", "He served."]),
        ("Pay 5 vs. 6 dollars. ok. Next one.", ["Pay 5 vs. 6 dollars. ok.", "Next one."]),
        ("It opened in Oct. 2013 in Beijing.", ["It opened in Oct. 2013 in Beijing."]),
        ("He left in 1830.[n 6] Here he lived.", ["He left in 1830.[n 6]", "Here he lived."]),
        ("Short (see below). Next.", ["Short (see below).", "Next."]),
        ("  Padded.  Twice.  ", ["Padded.", "Twice."]),
        ("", []),
    ],
)
def test_sentence_split(text, expected):
    assert sentences(text) == expected
    for start, end, s in D.sentence_split(text):
        assert text[start:end] == s


def squad():
    return D.squad_records(D.load_squad("dev", SQUAD), "dev")


def test_squad_records_basic_shape():
    records = squad()
    assert len(records) == 30
    assert sum(r.meta["unanswerable"] for r in records) == 10
    for r in records:
        assert r.id.startswith("squad-")
        assert r.meta["dataset"] == "squad" and r.meta["split"] == "dev"
        assert [u.id for u in r.units] == [f"u{i}" for i in range(1, len(r.units) + 1)]


def test_squad_gold_is_sentences_overlapping_answers():
    for r in squad():
        if r.meta["unanswerable"]:
            assert r.gold == [] and r.meta["dataset_answers"] == []
            continue
        spans = r.meta["sentence_spans"]
        starts = r.meta["answer_starts"]
        assert r.gold, r.id
        for start in starts:
            assert any(spans[int(g[1:]) - 1][0] <= start < spans[int(g[1:]) - 1][1] for g in r.gold)
        # every gold sentence holds at least one answer text
        gold_text = " ".join(u.text for u in r.units if u.id in r.gold)
        assert any(a in gold_text for a in r.meta["dataset_answers"])


def test_squad_gold_known_record():
    r = next(r for r in squad() if r.id == "squad-56dde0ba66d3e219004dad75")
    assert r.gold == ["u2"]
    assert r.meta["dataset_answers"] == ["911"]
    assert "began in 911" in r.units[1].text


def test_squad_stratified_sample_is_deterministic():
    records = squad()
    a = D.sample_squad(records, 10, seed=3)
    b = D.sample_squad(records, 10, seed=3)
    assert [r.id for r in a] == [r.id for r in b]
    assert sum(r.meta["unanswerable"] for r in a) == 3
    assert len({r.id for r in a}) == 10
    c = D.sample_squad(records, 10, seed=4)
    assert [r.id for r in a] != [r.id for r in c]


def test_squad_sample_too_large_fails():
    with pytest.raises(ValueError):
        D.sample_squad(squad(), 30, seed=0)  # needs 9 unanswerable, 21 answerable; has 20


def test_load_squad_rejects_unknown_split():
    with pytest.raises(ValueError):
        D.load_squad("test", SQUAD)


def hotpot():
    return D.hotpot_records(D.load_hotpotqa(HOTPOT))


def test_hotpot_gold_matches_supporting_facts():
    raw = D.load_hotpotqa(HOTPOT)
    records = hotpot()
    assert len(records) == 30
    for q, r in zip(raw, records, strict=True):
        assert r.id == f"hotpot-{q['_id']}"
        assert len(r.units) == len(q["context"])
        support = {title for title, _ in q["supporting_facts"]}
        gold_titles = {
            title for (title, _), u in zip(q["context"], r.units, strict=True) if u.id in r.gold
        }
        assert gold_titles == support
        assert r.meta["type"] == q["type"] in ("bridge", "comparison")
        assert r.meta["dataset_answers"] == [q["answer"]]
        assert r.meta["dataset"] == "hotpotqa" and r.meta["split"] == "dev"


def test_hotpot_known_record():
    r = hotpot()[0]
    assert r.gold == ["u2", "u5"]  # Scott Derrickson, Ed Wood
    assert r.units[1].text.startswith("Scott Derrickson: ")


def test_hotpot_sample_is_deterministic():
    a = D.sample_hotpotqa(hotpot(), 5, seed=1)
    assert [r.id for r in a] == [r.id for r in D.sample_hotpotqa(hotpot(), 5, seed=1)]


def test_build_records_round_trip(tmp_path):
    out = tmp_path / "r.jsonl"
    built = D.build_records("squad", out, n=10, seed=0, source=SQUAD)
    assert read_records(out) == built
    built = D.build_records("hotpotqa", out, n=5, seed=0, source=HOTPOT)
    assert read_records(out) == built
    built = D.build_records("squad-int", out, source=SQUAD)
    assert len(built) == 6 and read_records(out) == built
    with pytest.raises(ValueError):
        D.build_records("nope", out, n=1, source=SQUAD)


def test_data_dir_env_and_cached_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv(D.DATA_ENV, str(tmp_path))
    assert D.data_dir() == tmp_path
    sha = hashlib.sha256(SQUAD.read_bytes()).hexdigest()
    monkeypatch.setitem(D.SOURCES, "squad-dev", (("http://invalid.test/",), "dev.json", sha))
    (tmp_path / "dev.json").write_bytes(SQUAD.read_bytes())
    assert D.fetch("squad-dev") == tmp_path / "dev.json"  # no network: cached and verified
    assert len(D.squad_records(D.load_squad("dev"), "dev")) == 30


def test_fetch_rejects_corrupt_cached_file(tmp_path, monkeypatch):
    monkeypatch.setenv(D.DATA_ENV, str(tmp_path))
    (tmp_path / "dev-v2.0.json").write_text("{}")
    with pytest.raises(ValueError, match="sha256"):
        D.fetch("squad-dev")
