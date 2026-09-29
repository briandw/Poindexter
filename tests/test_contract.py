import json

import pytest

from poindexter.contract import (
    ABSTAIN,
    OTHER,
    SAME,
    UNSTABLE,
    Answer,
    ProbeRun,
    Record,
    Rejection,
    Result,
    Unit,
    canonical,
    majority,
    normalize,
    outcome_of,
    span_in_cites,
    validate,
)

IDS = {"u1", "u2", "u3"}


def obj(**kw):
    base = {"answer": "90 days", "cites": ["u2"], "abstain": False}
    base.update(kw)
    return json.dumps(base)


def code(raw):
    r = validate(raw, IDS)
    assert isinstance(r, Rejection), r
    return r.code


def test_valid_bare_and_fenced():
    assert validate(obj(), IDS) == Answer("90 days", ["u2"], False)
    assert validate("```json\n" + obj() + "\n```", IDS) == Answer("90 days", ["u2"], False)
    assert validate("  " + obj() + "\n", IDS) == Answer("90 days", ["u2"], False)


def test_valid_abstain():
    assert validate(obj(answer=None, cites=[], abstain=True), IDS) == Answer(None, [], True)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "Here you go: " + obj(),
        obj() + "\n" + obj(),
        "```\n" + obj() + "\n```",
        "```json\n" + obj() + "\n```\nthanks",
        "[1, 2]",
    ],
)
def test_invalid_json(raw):
    assert code(raw) == "INVALID_JSON"


def test_bad_keys():
    assert code(json.dumps({"answer": "x", "cites": ["u1"]})) == "BAD_KEYS"
    assert code(json.dumps({"answer": "x", "cites": ["u1"], "abstain": False, "why": ""})) == (
        "BAD_KEYS"
    )


def test_bad_abstain():
    assert code(obj(abstain=True)) == "BAD_ABSTAIN"
    assert code(obj(answer=None, abstain=True)) == "BAD_ABSTAIN"
    assert code(obj(abstain="false")) == "BAD_ABSTAIN"


def test_bad_answer():
    assert code(obj(answer=None)) == "BAD_ANSWER"
    assert code(obj(answer="  ")) == "BAD_ANSWER"
    assert code(obj(answer="x" * 201)) == "BAD_ANSWER"
    assert code(obj(answer=90)) == "BAD_ANSWER"


def test_bad_cites():
    assert code(obj(cites=[])) == "BAD_CITES"
    assert code(obj(cites=["u2", "u2"])) == "BAD_CITES"
    assert code(obj(cites=["u9"])) == "BAD_CITES"
    assert code(obj(cites="u2")) == "BAD_CITES"


def test_normalize_squad_examples():
    assert normalize("The Normans") == "normans"
    assert normalize("  10th and 11th   centuries.") == "10th and 11th centuries"
    assert normalize("a Denmark, Iceland and Norway!") == "denmark iceland and norway"
    assert normalize("An apple") == "apple"


def test_span_in_cites():
    units = [Unit("u1", "It rained."), Unit("u2", "The warranty lasts 90 days.")]
    assert span_in_cites(Answer("90 Days", ["u2"], False), units)
    assert not span_in_cites(Answer("90 days", ["u1"], False), units)
    assert not span_in_cites(Answer(None, [], True), units)


def test_outcome_and_majority():
    a = Answer("The Normans", ["u1"], False)
    assert outcome_of(Answer("normans", ["u2"], False), a) == SAME
    assert outcome_of(Answer(None, [], True), a) == ABSTAIN
    assert outcome_of(Answer("Vikings", ["u1"], False), a) == OTHER
    assert outcome_of(None, a) == OTHER
    assert majority([SAME, SAME, OTHER]) == SAME
    assert majority([SAME, OTHER]) == UNSTABLE
    assert majority([SAME, OTHER, ABSTAIN]) == UNSTABLE


def test_record_strict_keys_and_collisions():
    good = {"id": "r", "question": "q", "units": [{"id": "u1", "text": "t"}]}
    assert Record.from_json(good).to_json() == good
    with pytest.raises(ValueError):
        Record.from_json({**good, "extra": 1})
    with pytest.raises(ValueError):
        Record.from_json({**good, "units": [{"id": "u1", "text": "a"}, {"id": "u1", "text": "b"}]})
    with_meta = {**good, "gold": ["u1"], "meta": {"dataset": "squad"}}
    assert Record.from_json(with_meta).to_json() == with_meta


def test_result_round_trip():
    rec = Record("r", "q", [Unit("u1", "t")], gold=["u1"])
    a = Answer("t", ["u1"], False)
    run = ProbeRun(raw=["{}"], parsed=[a], outcomes=[SAME], majority=SAME)
    res = Result(
        record=rec, model="m", k=1, temperature=None, status="ok", A=a,
        probes={"O": run}, loo={"u1": run}, verdict="grounded", flags={}, alignment={},
    )
    assert Result.from_json(json.loads(json.dumps(res.to_json()))).to_json() == res.to_json()


def test_duplicate_keys_rejected():
    raw = '{"answer": "x", "answer": "y", "cites": ["u1"], "abstain": false}'
    assert code(raw) == "BAD_KEYS"


def test_abstain_rule_wins_over_type_checks():
    assert code(obj(answer=90, cites=[], abstain=True)) == "BAD_ABSTAIN"
    assert code(obj(answer=None, cites="u2", abstain=True)) == "BAD_ABSTAIN"


def test_supplied_answer_must_satisfy_contract():
    base = {"id": "r", "question": "q", "units": [{"id": "u1", "text": "t"}]}
    ok = {**base, "answer": {"text": "t", "cites": ["u1"], "abstain": False}}
    assert Record.from_json(ok).answer == Answer("t", ["u1"], False)
    bad = [
        {"text": "t", "cites": "u1", "abstain": False},
        {"text": "t", "cites": ["u1"], "abstain": "false"},
        {"text": "t", "cites": ["u9"], "abstain": False},
        {"text": "t", "cites": [], "abstain": True},
        {"text": "", "cites": ["u1"], "abstain": False},
    ]
    for answer in bad:
        with pytest.raises(ValueError):
            Record.from_json({**base, "answer": answer})
    with pytest.raises(ValueError):
        Record.from_json({**base, "answer": None})


@pytest.mark.parametrize(
    "a,b",
    [("two atoms", "2"), ("in 1791", "1791"), ("1,000 men", "1000"), ("Twenty", "20"),
     ("1791.", "the year 1791"), ("3.5", "3.5 km")],
)  # fmt: skip
def test_canonical_numbers_equal(a, b):
    assert canonical(a) == canonical(b)


def test_canonical_other_cases():
    assert canonical("3.5") != canonical("35")
    assert canonical("1791 and 1792") == normalize("1791 and 1792")
    assert canonical("The Normans") == "normans"
    assert canonical("10th and 11th centuries") == "10th and 11th centuries"
    assert canonical("1791") != canonical("1792")


def test_outcome_uses_canonical_and_span_stays_text():
    a = Answer("in 1791", ["u1"], False)
    assert outcome_of(Answer("1791", ["u1"], False), a) == SAME
    assert outcome_of(Answer("1792", ["u1"], False), a) == OTHER
    units = [Unit("u1", "It opened in 1791.")]
    assert not span_in_cites(Answer("seventeen ninety-one", ["u1"], False), units)
    assert not span_in_cites(Answer("two", ["u1"], False), [Unit("u1", "It had 2 wings.")])


@pytest.mark.parametrize(
    "a,b",
    [("one hundred", "1"), ("1 million", "1"), ("two dozen", "2"), ("3 thousand", "3"),
     ("four score", "4"), ("-1", "1"), ("\u22121", "1"), ("-1 degrees", "1 degree")],
)  # fmt: skip
def test_canonical_magnitude_and_minus_are_not_numbers(a, b):
    assert canonical(a) != canonical(b)
    assert canonical(a) == normalize(a)


def test_canonical_hyphen_between_words_is_not_minus():
    assert canonical("twenty-one") == normalize("twenty-one")  # two number words
    assert canonical("a 2-hour wait") == canonical("2")
