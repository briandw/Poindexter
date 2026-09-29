import random
from pathlib import Path

import pytest

from poindexter import datasets as D
from poindexter.contract import Answer, Record, Unit, read_records

SQUAD = Path(__file__).parent / "fixtures" / "squad_30.json"

CANDIDATES = [
    "squad-56dde0ba66d3e219004dad75",  # 911
    "squad-56de148dcffd8e1900b4b5bd",  # 1082
    "squad-56de1563cffd8e1900b4b5c2",  # 1185, two-sentence paragraph
    "squad-56de1645cffd8e1900b4b5d1",  # 1041
    "squad-56de16ca4396321400ee25c7",  # 1066
    "squad-56de179dcffd8e1900b4b5da",  # 1169
]


def candidates():
    return D.integer_candidates("dev", SQUAD)


def record(units, gold, answer, rid="squad-x"):
    return Record(
        id=rid,
        question="q?",
        units=[Unit(f"u{i}", t) for i, t in enumerate(units, 1)],
        gold=gold,
        meta={"dataset": "squad", "split": "dev", "dataset_answers": [answer]},
    )


def test_integer_candidates_on_fixture():
    assert [r.id for r in candidates()] == CANDIDATES


def test_integer_filter_counts_drop_ambiguous():
    records = D.squad_records(D.load_squad("dev", SQUAD), "dev")
    kept, counts = D.integer_filter_counts(records)
    assert [r.id for r in kept] == CANDIDATES
    assert counts == {
        "all": 30,
        "answerable": 20,
        "first_answer_integer": 8,
        "multi_digit": 8,
        "answers_agree": 7,  # drops 1018/1064: annotators gave different years
        "first_answer_in_one_sentence": 7,
        "one_gold_sentence": 7,
        "value_unique_in_units": 6,  # drops 1191: occurs twice in the paragraph
        "occurrence_is_answer_span": 6,
        "swappable": 6,
    }


def squad_json(context, answers, qid="q1"):
    """A one-question SQuAD 2.0 file; answers are (text, occurrence index in context)."""
    spans = []
    for text, nth in answers:
        start = -1
        for _ in range(nth + 1):
            start = context.index(text, start + 1)
        spans.append({"text": text, "answer_start": start})
    qa = {"id": qid, "question": "q?", "answers": spans, "is_impossible": False}
    return {
        "version": "v2.0",
        "data": [{"title": "T", "paragraphs": [{"context": context, "qas": [qa]}]}],
    }


def filter_step(context, answers):
    records = D.squad_records(squad_json(context, answers), "dev")
    kept, counts = D.integer_filter_counts(records)
    return next(step for step in reversed(D.FILTER_STEPS) if counts[step])


@pytest.mark.parametrize(
    "context, answers, last_step",
    [
        ("It cost 1000 dollars. Later it cost 1,000 more.", [("1000", 0)], "one_gold_sentence"),
        ("There were 13 ships. Thirteen sank.", [("13", 0)], "one_gold_sentence"),
        ("There were 13 ships. Twelve sank.", [("13", 0)], "swappable"),
        ("A triangle has 3 sides. Squares have 4.", [("3", 0)], "first_answer_integer"),
        ("It cost 1000 or 1002 dollars. Later.", [("1000", 0), ("1002", 0)], "multi_digit"),
        (
            "It cost 1000 dollars. Later.",
            [("1000", 0), ("1000 dollars", 0)],
            "swappable",
        ),
        (
            "It had 10 rooms. Plans: 11, 12, thirteen, fourteen.",  # 10 has no free value in 11..14
            [("10", 0)],
            "occurrence_is_answer_span",
        ),
    ],
)
def test_filter_numeric_aliases_digits_and_agreement(context, answers, last_step):
    assert filter_step(context, answers) == last_step


def test_number_mentions():
    text = "1,000 or 1000, Three and 3.5, the 5th of 1990s, 8-pin, one."
    assert [v for _, _, v in D.number_mentions(text)] == [1000, 1000, 3, 8, 1]


def draws(original, avoid="", n=400):
    return [D.swapped_integer(original, random.Random(seed), avoid) for seed in range(n)]


@pytest.mark.parametrize(
    "year, lo, hi",
    [
        ("1066", 1041, 1091),
        ("1853", 1828, 1878),
        ("2008", 1983, 2016),  # clamped: no year past 2016
        ("2014", 1989, 2016),
        ("2016", 1991, 2015),  # only earlier years are valid
        ("2050", 2025, 2049),  # a year already past 2016 may stay there, but not rise
        ("1000", 1001, 1025),  # only later years are valid
        ("2100", 2075, 2099),
    ],
)
def test_year_window_and_clamp(year, lo, hi):
    values = [int(s) for s in draws(year)]
    assert all(lo <= x <= hi and x != int(year) for x in values)
    assert all(1 <= abs(x - int(year)) <= 25 for x in values)
    assert min(values) == lo and max(values) == hi  # the whole window is reachable


def test_year_picks_a_side_uniformly_when_both_are_valid():
    later = sum(int(s) > 2014 for s in draws("2014"))  # 2 later values vs 25 earlier
    assert 150 < later < 250


@pytest.mark.parametrize(
    "n, lo, hi",
    [
        ("10", 11, 14),  # window 6..14, two-digit part only
        ("13", 10, 18),
        ("34", 21, 47),
        ("99", 60, 99),
        ("911", 547, 999),  # window 547..1275, three-digit part only
        ("3600", 2160, 5040),  # four digits but not a year
        ("50000", 30000, 70000),
    ],
)
def test_count_window(n, lo, hi):
    values = [int(s) for s in draws(n, n=60)]
    assert all(lo <= x <= hi and x != int(n) for x in values)
    assert all(len(str(x)) == len(n) for x in values)


@pytest.mark.parametrize(
    "n, expected",
    [
        ("8000", {5000, 6000, 7000, 9000}),  # window 4800..11200, four digits
        ("400", {300, 500}),  # window 240..560
        ("50", {30, 40, 60, 70}),
        ("3600", set(range(2200, 5001, 100)) - {3600}),  # window 2160..5040
    ],
)
def test_round_numbers_stay_round(n, expected):
    assert {int(s) for s in draws(n, n=600)} == expected


def test_round_number_falls_back_when_no_round_value_is_free():
    values = {int(s) for s in draws("400", avoid="300 500", n=50)}
    assert values and all(240 <= x <= 560 and x not in (300, 400, 500) for x in values)
    assert any(x % 100 for x in values)


@pytest.mark.parametrize("n", ["1", "5", "9"])
def test_one_digit(n):
    values = {int(s) for s in draws(n)}
    assert values == set(range(1, 10)) - {int(n)}


def test_no_value_outside_the_window():
    # Window for 10 is 11..14 (two digits); all taken, so there is no plausible swap.
    with pytest.raises(ValueError):
        D.swapped_integer("10", random.Random(0), avoid="11 12 13 14")


def test_year_avoids_values_in_context():
    taken = " ".join(str(y) for y in range(1041, 1092) if y != 1070)
    assert set(draws("1066", avoid=taken, n=20)) == {"1070"}


def test_swapped_integer_is_deterministic_and_rejects_leading_zeros():
    assert draws("1853", n=50) == draws("1853", n=50)
    assert len(set(draws("1853", n=50))) > 1
    with pytest.raises(ValueError):
        D.swapped_integer("007", random.Random(0))


def test_swap_changes_only_gold_sentence():
    for r in candidates():
        s = D.swap_record(r, seed=0)
        cf = s.meta["counterfactual"]
        assert s.id == f"{r.id}~swap"
        assert s.gold == r.gold
        assert [u.id for u in s.units] == [u.id for u in r.units]
        assert cf["original"] == r.meta["dataset_answers"][0]
        assert cf["swapped"] != cf["original"] and len(cf["swapped"]) == len(cf["original"])
        assert cf["source_id"] == r.id
        assert s.meta["dataset_answers"] == [cf["swapped"]]
        for old, new in zip(r.units, s.units, strict=True):
            if old.id in r.gold:
                assert new.text == old.text.replace(cf["original"], cf["swapped"])
                assert new.text != old.text
            else:
                assert new == old
        assert r.meta["dataset_answers"][0] == cf["original"]  # source not mutated


def test_swap_is_deterministic_per_record_and_seed():
    rs = candidates()
    assert [D.swap_record(r, 0) for r in rs] == [D.swap_record(r, 0) for r in rs]
    by_seed = {
        tuple(D.swap_record(r, seed).meta["counterfactual"]["swapped"] for r in rs)
        for seed in range(5)
    }
    assert len(by_seed) > 1


def test_swap_rejects_ambiguous_records():
    with pytest.raises(ValueError):  # integer also in a non-gold sentence
        D.swap_record(record(["Born 1950.", "Died 1950."], ["u1"], "1950"), 0)
    with pytest.raises(ValueError):  # not an integer
        D.swap_record(record(["Born in May."], ["u1"], "May"), 0)
    with pytest.raises(ValueError):  # two gold units
        D.swap_record(record(["Born 1950.", "Died."], ["u1", "u2"], "1950"), 0)


def test_swap_whole_token_only():
    r = record(["It cost 19 dollars in 1919.", "Later."], ["u1"], "19")
    s = D.swap_record(r, 0)
    assert s.units[0].text.endswith("in 1919.")
    assert s.units[0].text == f"It cost {s.meta['counterfactual']['swapped']} dollars in 1919."


def test_redundant_copy_placement_and_ids():
    for r in candidates():
        d = D.redundant_record(r, seed=0)
        ids = [u.id for u in d.units]
        info = d.meta["redundant"]
        assert d.id == f"{r.id}~dup"
        assert d.gold == r.gold
        assert info == {"copy_of": r.gold[0], "copy_id": f"u{len(r.units) + 1}", "source_id": r.id}
        assert len(d.units) == len(r.units) + 1
        assert [u for u in d.units if u.id != info["copy_id"]] == r.units
        orig, copy = ids.index(info["copy_of"]), ids.index(info["copy_id"])
        assert abs(orig - copy) > 1
        assert d.units[copy].text == d.units[orig].text
        assert d == D.redundant_record(r, seed=0)


def test_redundant_two_units_goes_to_the_end():
    d = D.redundant_record(record(["Gold 1185.", "Other."], ["u1"], "1185"), 0)
    assert [u.id for u in d.units] == ["u1", "u2", "u3"]
    assert d.units[2].text == "Gold 1185."
    d = D.redundant_record(record(["Other.", "Gold 1185."], ["u2"], "1185"), 0)
    assert [u.id for u in d.units] == ["u3", "u1", "u2"]


def test_redundant_single_unit_fails():
    with pytest.raises(ValueError):
        D.redundant_record(record(["Only 1185."], ["u1"], "1185"), 0)


def test_write_swapped_and_redundant(tmp_path):
    src = tmp_path / "int.jsonl"
    D.build_records("squad-int", src, source=SQUAD)
    swapped = D.write_swapped(src, tmp_path / "swap.jsonl", seed=0)
    assert read_records(tmp_path / "swap.jsonl") == swapped
    assert [r.id for r in swapped] == [f"{i}~swap" for i in CANDIDATES]
    dup = D.write_redundant(src, tmp_path / "dup.jsonl", seed=0)
    assert read_records(tmp_path / "dup.jsonl") == dup
    assert len(dup) == len(CANDIDATES)


def test_swap_avoids_values_already_in_context():
    # Every other 1..9 digit is present, so the only legal swap is 9.
    units = ["Room 1 2 3 4.", "Room 5 6 7 8."]
    for seed in range(20):
        s = D.swap_record(record(units, ["u1"], "1"), seed)
        assert s.meta["counterfactual"]["swapped"] == "9"


def test_swap_never_draws_a_value_mentioned_in_another_form():
    # Window for 12 is 10..16; 10, 11, 13, 14, 15 are taken as words or with separators.
    units = ["It had 12 rooms.", "Ten, eleven, thirteen, 14 and fifteen were planned."]
    values = {
        D.swap_record(record(units, ["u1"], "12"), seed).meta["counterfactual"]["swapped"]
        for seed in range(40)
    }
    assert values == {"16"}


def test_swap_avoids_number_words_and_separators():
    assert "3" not in set(draws("5", avoid="three", n=200))
    assert "1050" not in set(draws("1066", avoid="1,050", n=400))


def test_swap_rejects_value_written_in_another_form_elsewhere():
    with pytest.raises(ValueError):
        D.swap_record(record(["It had 1000 rooms.", "Or 1,000."], ["u1"], "1000"), 0)
    with pytest.raises(ValueError):
        D.swap_record(record(["It had 13 rooms.", "Thirteen burned."], ["u1"], "13"), 0)


def test_swap_and_redundant_drop_supplied_answer():
    r = record(["Born 1950.", "Died."], ["u1"], "1950")
    r = Record(r.id, r.question, r.units, Answer("1950", ["u1"], False), r.gold, r.meta)
    assert D.swap_record(r, 0).answer is None
    assert D.redundant_record(r, 0).answer is None
