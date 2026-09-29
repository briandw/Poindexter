"""Probe C (counterfactual), its verdict, and the open agent's validator."""

import asyncio
import json
import random

import pytest
from conftest import ABSTAIN_JSON, ScriptedBackend, answer

from poindexter import datasets
from poindexter.contract import (
    ABSTAIN,
    FOLLOWS,
    OTHER,
    SAME,
    UNSTABLE,
    Answer,
    ProbeRun,
    Record,
    Result,
    Unit,
    counterfactual_outcome,
    split_values,
    validate,
    validate_open,
)
from poindexter.probes import answer_target, choose_replacement, counterfactual, occurrences
from poindexter.prompt import (
    AGENTS,
    CLOSED_BOOK_SYSTEM,
    NO_CONTEXT,
    OPEN_SYSTEM,
    SYSTEM,
    build_closed_book_prompt,
)
from poindexter.prompt import build_prompt as _build_prompt
from poindexter.runner import PROBE_SETS, counterfactual_plan, run_record, run_records
from poindexter.verdict import alignment, counterfactual_verdict, recompute


def go(coro):
    return asyncio.run(coro)


# --- the pure probe ---------------------------------------------------------------------

UNITS = [
    Unit("u1", "The bridge opened in 1850."),
    Unit("u2", "It was rebuilt in 1850s style, 1,850 feet long."),
    Unit("u3", "Its architect was born in 1850 too."),
]


def test_counterfactual_replaces_the_one_cited_occurrence():
    out = counterfactual(UNITS, ["u1", "u2"], "1850", "1862")
    assert [u.id for u in out] == ["u1", "u2", "u3"]
    assert out[0] == Unit("u1", "The bridge opened in 1862.")
    # "1850s" and "1,850" are not standalone "1850"; u3 is uncited and untouched.
    assert out[1:] == UNITS[1:]
    assert UNITS[0].text == "The bridge opened in 1850."  # input not mutated


def test_counterfactual_keeps_cited_order_and_matches_case_insensitively():
    units = [Unit("a", "Home of the Louvre."), Unit("b", "The capital is Paris.")]
    out = counterfactual(units, ["b", "a"], "paris", "Lyon")
    assert out == [units[0], Unit("b", "The capital is Lyon.")]


@pytest.mark.parametrize(
    "cites, text",
    [
        (["u1", "u3"], "1850"),  # once in each of two cited units
        (["u2"], "1850"),  # only inside "1850s" and "1,850"
        (["u2"], "850"),  # inside the thousands group
        (["u1"], "1851"),  # absent
    ],
)
def test_counterfactual_requires_exactly_one_occurrence(cites, text):
    with pytest.raises(ValueError):
        counterfactual(UNITS, cites, text, "1862")


def test_counterfactual_errors_on_twice_in_one_unit_and_empty_strings():
    units = [Unit("u1", "Paris, Paris.")]
    assert occurrences(units, ["u1"], "Paris") == 2
    with pytest.raises(ValueError):
        counterfactual(units, ["u1"], "Paris", "Lyon")
    with pytest.raises(ValueError):
        counterfactual(UNITS, ["u1"], "1850", " ")
    with pytest.raises(ValueError):
        counterfactual(UNITS, ["u1"], "", "1862")


def test_answer_target():
    assert answer_target(UNITS, ["u1"], "in 1850") == ("1850", 1)
    assert answer_target(UNITS, ["u1"], "the year 1850") == (None, 0)  # "year" not in u1
    assert answer_target(UNITS, ["u2"], "1850") == ("1,850", 1)  # the mention as written
    assert answer_target(UNITS, ["u1", "u3"], "1850") == (None, 2)
    assert answer_target(UNITS, ["u1"], "1851") == (None, 0)
    three = [Unit("u1", "It has three arches.")]
    assert answer_target(three, ["u1"], "3") == ("three", 1)
    assert answer_target(three, ["u1"], "three arches") == ("three", 1)
    assert answer_target([Unit("u1", "By Eiffel.")], ["u1"], "Eiffel") == ("Eiffel", 1)
    assert answer_target([Unit("u1", "By Eiffel.")], ["u1"], "Gustave Eiffel") == (None, 0)


# --- the replacement chooser -------------------------------------------------------------


def rec(units, meta=None, rid="r1", question="When did it open?"):
    return Record(rid, question, [Unit(f"u{i}", t) for i, t in enumerate(units, 1)], meta=meta)


def test_choose_replacement_integer_uses_v1_rules_in_its_own_namespace():
    r = rec(["The bridge opened in 1850.", "It is long."])
    got = choose_replacement(r, "1850", 7)
    context = "\n".join([r.question, *(u.text for u in r.units)])
    assert got == datasets.swapped_integer("1850", random.Random("7:r1:C"), avoid=context)
    assert got == choose_replacement(r, "in 1850", 7)  # deterministic, canonical input
    for seed in range(50):
        new = int(choose_replacement(r, "1850", seed))
        assert new != 1850 and 1825 <= new <= 1875  # year window, same digit count


def test_choose_replacement_integer_avoids_values_in_the_record():
    # 1851..1875 are all mentioned somewhere, in digits, with separators, or in the question.
    later = ", ".join(str(y) for y in range(1851, 1874))
    r = rec([f"Opened in 1850; see {later}.", "Also 1,874."], question="In 1875?")
    for seed in range(40):
        assert int(choose_replacement(r, "1850", seed)) < 1850


def test_choose_replacement_keeps_thousands_separators():
    r = rec(["It is 1,850 feet long."], rid="r2")
    got = choose_replacement(r, "1,850", 0)
    assert "," in got and got.replace(",", "") != "1850"
    assert 1000 <= int(got.replace(",", "")) <= 9999


def test_choose_replacement_entities():
    meta = {"alternatives": ["Paris", "the Paris", "Lyon", "Marseille", "Nice"]}
    r = rec(["The capital is Paris.", "Nice is on the coast."], meta)
    picks = {choose_replacement(r, "Paris", s) for s in range(40)}
    assert picks == {"Lyon", "Marseille"}  # not canonically Paris, not in any unit
    assert choose_replacement(r, "Paris", 3) == choose_replacement(r, "Paris", 3)


@pytest.mark.parametrize(
    "meta",
    [None, {}, {"alternatives": []}, {"alternatives": "Lyon"}, {"alternatives": ["Lyon", 3]},
     {"alternatives": ["paris", "Nice"]}],
)  # fmt: skip
def test_choose_replacement_none(meta):
    r = rec(["The capital is Paris.", "Nice is on the coast."], meta)
    assert choose_replacement(r, "Paris", 0) is None


def test_choose_replacement_number_words():
    r = rec(["It has three arches and 2 towers."])
    picks = {choose_replacement(r, "three", s) for s in range(60)}
    assert picks == {"one", "four", "five", "six", "seven", "eight", "nine"}  # not 2 or 3
    assert choose_replacement(r, "3", 0) == choose_replacement(r, "three", 0)
    cap = rec(["Three arches remain."])
    assert all(choose_replacement(cap, "Three", s)[0].isupper() for s in range(10))
    assert choose_replacement(rec(["TWO arches."]), "TWO", 0).isupper()
    twelve = rec(["It has twelve arches."])
    words = {"ten", "eleven", "thirteen", "fourteen", "fifteen", "sixteen"}
    assert {choose_replacement(twelve, "twelve", s) for s in range(60)} <= words
    # zero has no plausible swap, and a number never takes an entity alternative.
    assert choose_replacement(rec(["Zero arches."], {"alternatives": ["five"]}), "zero", 0) is None
    # Twenty-something: a value past twenty is written in digits.
    twenty = rec(["Twenty arches."], rid="r20")
    got = {choose_replacement(twenty, "twenty", s) for s in range(60)}
    assert got - {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
                  "Seventeen", "Eighteen", "Nineteen"} <= {str(n) for n in range(21, 29)}


def test_choose_replacement_excludes_heldout_values():
    # Every plausible one-digit swap but 7 is held out, in some numeric form.
    heldout = ["1", "two", "3", "4", "5 km", "six", "8", "9"]
    r = rec(["It has three arches."], {"heldout_values": heldout})
    assert {choose_replacement(r, "3", s) for s in range(30)} == {"seven"}
    r = rec(["It has three arches."], {"heldout_values": [*heldout, "7"]})
    assert choose_replacement(r, "3", 0) is None  # no_replacement
    year = rec(["Opened in 1850."], {"heldout_values": ["1,851", "1849"]})
    assert all(choose_replacement(year, "1850", s) not in {"1851", "1849"} for s in range(80))
    ents = {"alternatives": ["Lyon", "Nice", "Lille"], "heldout_values": ["lyon", "the Nice"]}
    r = rec(["The capital is Paris."], ents)
    assert {choose_replacement(r, "Paris", s) for s in range(20)} == {"Lille"}
    r = rec(["The capital is Paris."], {**ents, "heldout_values": ["Lyon", "Nice", "Lille"]})
    assert choose_replacement(r, "Paris", 0) is None
    info, _ = counterfactual_plan(r, Answer("Paris", ["u1"], False), 0)
    assert info["reason"] == "no_replacement"


def test_numeric_c_needs_the_answers_other_words_in_the_target_unit():
    units = [Unit("u1", "The bridge is 90 feet long."), Unit("u2", "Its warranty: 90 days.")]
    assert answer_target(units, ["u1"], "90 days") == (None, 0)
    assert answer_target(units, ["u1"], "90") == ("90", 1)  # a bare number is unchanged
    assert answer_target(units, ["u1"], "90 feet") == ("90", 1)
    assert answer_target(units, ["u2"], "The 90 Days") == ("90", 1)
    r = Record("r", "How long?", units)
    info, ctx = counterfactual_plan(r, Answer("90 days", ["u1"], False), 0)
    assert info["reason"] == "answer_not_in_cites" and ctx is None


def test_numeric_answer_never_falls_back_to_alternatives():
    alts = {"alternatives": ["Lyon"]}
    for units, text in [
        (["It weighs 3.5 tons."], "3.5"),  # decimal: no integer swap
        (["Zero arches."], "0"),  # no plausible swap
        (["It opened long ago."], "1850"),  # the record never mentions the value
    ]:
        r = rec(units, alts)
        assert choose_replacement(r, text, 0) is None
    r = rec(["It weighs 3.5 tons."], alts)
    info, _ = counterfactual_plan(r, Answer("3.5", ["u1"], False), 0)
    assert info["reason"] == "no_replacement"


@pytest.mark.parametrize(
    "text", ["A Paris-based firm.", "Near PARIS.", "(Paris)", "Paris's mayor.", "Paris, 1900."]
)
def test_entity_alternatives_excluded_at_token_boundaries(text):
    r = rec(["The capital is Lyon.", text], {"alternatives": ["Paris", "Nice"]})
    assert {choose_replacement(r, "Lyon", s) for s in range(20)} == {"Nice"}


def test_entity_alternatives_inside_a_word_still_allowed():
    r = rec(["The capital is Lyon.", "Parisians and Niceties."], {"alternatives": ["Paris"]})
    assert choose_replacement(r, "Lyon", 0) == "Paris"


def test_choose_replacement_entity_excludes_question():
    meta = {"alternatives": ["Lyon", "Nice"]}
    r = rec(["The capital is Paris."], meta, question="Is it Lyon or Paris?")
    assert {choose_replacement(r, "Paris", s) for s in range(20)} == {"Nice"}


# --- outcome and verdict -----------------------------------------------------------------


def test_counterfactual_outcome():
    a = Answer("1850", ["u1"], False)
    assert counterfactual_outcome(Answer("in 1862", ["u1"], False), a, "1862") == FOLLOWS
    assert counterfactual_outcome(Answer("1850.", [], False), a, "1862") == SAME
    assert counterfactual_outcome(Answer(None, [], True), a, "1862") == ABSTAIN
    assert counterfactual_outcome(Answer("1900", ["u1"], False), a, "1862") == OTHER
    assert counterfactual_outcome(None, a, "1862") == OTHER
    paris = Answer("Paris", ["u1"], False)
    assert counterfactual_outcome(Answer("lyon", ["u1"], False), paris, "Lyon") == FOLLOWS


def test_counterfactual_verdict():
    assert counterfactual_verdict(FOLLOWS) == "grounded"
    assert counterfactual_verdict(SAME) == "decorative"
    for m in (ABSTAIN, OTHER, UNSTABLE):
        assert counterfactual_verdict(m) == "unstable"


def test_plan_reasons():
    r = rec(["The bridge opened in 1850.", "It opened in 1850 again."])
    info, ctx = counterfactual_plan(r, Answer(None, [], True), 0)
    assert (info["applicable"], info["reason"], ctx) == (False, "abstain", None)
    assert counterfactual_plan(r, Answer("1850", [], False), 0)[0]["reason"] == "uncited"
    both = Answer("1850", ["u1", "u2"], False)
    assert counterfactual_plan(r, both, 0)[0]["reason"] == "answer_repeated_in_cites"
    assert counterfactual_plan(r, Answer("1851", ["u1"], False), 0)[0]["reason"] == (
        "answer_not_in_cites"
    )
    paris = rec(["The capital is Paris."])
    assert counterfactual_plan(paris, Answer("Paris", ["u1"], False), 0)[0] == {
        "applicable": False, "reason": "no_replacement", "replacement": None, "target": None,
        "seed": 0, "run": None, "verdict": None,
    }  # fmt: skip
    info, ctx = counterfactual_plan(r, Answer("1850", ["u1"], False), 0)
    assert info["applicable"] and info["reason"] is None and info["seed"] == 0
    assert info["target"] == {"unit": "u1", "start": 21, "end": 25, "text": "1850"}
    hague = rec(["The court sits in The Hague."], {"alternatives": ["Paris"]})
    long_, _ = counterfactual_plan(hague, Answer("The Hague", ["u1"], False), 0)
    short, hague_ctx = counterfactual_plan(hague, Answer("hague", ["u1"], False), 0)
    assert long_["target"] == {"unit": "u1", "start": 18, "end": 27, "text": "The Hague"}
    assert short["target"] == {"unit": "u1", "start": 22, "end": 27, "text": "Hague"}
    assert hague_ctx[0].text == "The court sits in The Paris."
    assert ctx[0].text == f"The bridge opened in {info['replacement']}." and ctx[1] == r.units[1]


# --- runner end to end -------------------------------------------------------------------

Q = "How long is the warranty?"
W_UNITS = [Unit("u1", "The store opened in 1998."), Unit("u2", "The warranty lasts 90 days.")]
W = Record("w1", Q, W_UNITS)
W_REPLACEMENT = choose_replacement(W, "90", 0)
W_C = [W_UNITS[0], Unit("u2", f"The warranty lasts {W_REPLACEMENT} days.")]
CB = build_closed_book_prompt(Q)[1]


def user(units, question=Q, agent="context"):
    return _build_prompt(units, question, agent)[1]


def warranty_backend(c_response, o_response=None, model="scripted"):
    return ScriptedBackend(model=model, rules=[
        (user(W_C), c_response),
        (user(W_UNITS), o_response or answer("90 days", "u2")),
        (user(W_UNITS[:1]), ABSTAIN_JSON),
        (user(W_UNITS[1:]), answer("90 days", "u2")),
        (CB, ABSTAIN_JSON),
    ])  # fmt: skip


def run_one(backend, record=W, k=3, probes="verdict+C", agent="context"):
    sem = asyncio.Semaphore(4)
    return go(run_record(record, backend, k, None, 0, None, sem, probes, agent))


def test_probe_sets():
    assert PROBE_SETS == ("all", "verdict", "verdict+C", "O")


def test_runner_c_grounded():
    assert W_REPLACEMENT is not None and W_REPLACEMENT != "90"
    b = warranty_backend(answer(f"{W_REPLACEMENT} days", "u2"))
    r = run_one(b)
    assert list(r.probes) == ["O", "N", "R_remove", "M"] and r.verdict == "grounded"
    cf = r.counterfactual
    assert (cf["applicable"], cf["reason"], cf["replacement"]) == (True, None, W_REPLACEMENT)
    assert cf["run"]["outcomes"] == [FOLLOWS] * 3 and cf["run"]["majority"] == FOLLOWS
    assert cf["verdict"] == "grounded"
    assert b.prompts.count(user(W_C)) == 3
    assert r.compliance["calls"] == 15  # O, N, R_remove, M, C at k=3


def test_runner_c_decorative_and_json_round_trip():
    r = run_one(warranty_backend(answer("90 days", "u2")))
    assert r.counterfactual["run"]["outcomes"] == [SAME] * 3
    assert r.counterfactual["verdict"] == "decorative"
    back = Result.from_json(json.loads(json.dumps(r.to_json())))
    assert back.counterfactual == r.counterfactual and back.to_json() == r.to_json()


def test_runner_c_unstable_and_rejections_counted():
    c = [answer(f"{W_REPLACEMENT} days", "u2"), ABSTAIN_JSON, answer("90 days", "u9")]
    b = warranty_backend(c)
    b.rules.insert(0, ("rejected: BAD_CITES", answer("90 days", "u9")))
    r = run_one(b)
    assert r.counterfactual["run"]["outcomes"] == [FOLLOWS, ABSTAIN, OTHER]
    assert r.counterfactual["verdict"] == "unstable"
    assert r.compliance["final_rejections"] == 1


def test_runner_c_not_applicable():
    b = warranty_backend(answer("x", "u2"), o_response=answer("90 days", "u1"))
    b.rules.insert(0, (user(W_UNITS[1:]), ABSTAIN_JSON))  # R_remove when u1 is cited
    r = run_one(b)
    assert r.counterfactual == {
        "applicable": False, "reason": "answer_not_in_cites", "replacement": None,
        "target": None, "seed": 0, "run": None, "verdict": None,
    }  # fmt: skip
    assert user(W_C) not in b.prompts and r.verdict is not None

    abstains = ScriptedBackend([("", ABSTAIN_JSON)], model="abstains")  # own cache keys
    r = run_one(abstains)
    assert r.A.abstain and r.counterfactual["reason"] == "abstain" and r.verdict is None
    b = warranty_backend(answer("x", "u2"), model="m3")
    assert run_one(b, probes="verdict").counterfactual is None


def test_runner_all_includes_c():
    records = [W, Record("w2", "What opened?", [Unit("u1", "A shop opened.")])]
    b = ScriptedBackend([
        (user(W_C), answer(f"{W_REPLACEMENT} days", "u2")),
        ("Question: How long", answer("90 days", "u2")),
        ("Question: What opened", ABSTAIN_JSON),
        ("", ABSTAIN_JSON),
    ])  # fmt: skip
    by_id = {r.record.id: r for r in go(run_records(records, b, 1, None, 0, probes="all"))}
    assert by_id["w1"].counterfactual["verdict"] == "grounded"
    assert list(by_id["w1"].loo) == ["u1", "u2"]


# --- the open agent ----------------------------------------------------------------------


def test_open_prompt_allows_uncited_answers():
    assert '"cites": []' in OPEN_SYSTEM.split("Rules:")[1].split("3.")[0]
    assert "own knowledge" in OPEN_SYSTEM and OPEN_SYSTEM != SYSTEM


CONFLICT = (
    '4. If the context gives conflicting answers, answer with each value separated by " / " '
    "and cite every unit that states one.\n"
)


def test_conflict_rule_in_v2_agents_only():
    for name in ("context", "open"):
        rules = AGENTS[name].split("Rules:\n")[1]
        assert CONFLICT in rules
        numbers = [line.split(".")[0] for line in rules.splitlines()]
        assert numbers == ["1", "2", "3", "4", "5", "6"]
    for text in (AGENTS["context_v1"], AGENTS["open_v1"], CLOSED_BOOK_SYSTEM):
        assert "conflicting" not in text and " / " not in text


def test_split_values():
    assert split_values("1527 / 1525") == ["#1527", "#1525"]
    assert split_values("in 1527") == ["#1527"]
    assert split_values("The Normans / Vikings") == ["normans", "vikings"]
    assert split_values("AC/DC") == ["acdc"]  # no spaced separator: one value
    assert split_values("1527 / ") == ["#1527"]
    assert split_values("two / 2") == ["#2", "#2"]


IDS = {"u1", "u2"}


def raw(answer_, cites, abstain=False):
    return json.dumps({"answer": answer_, "cites": cites, "abstain": abstain})


def test_validate_open_accepts_uncited_answer_only_for_open():
    assert validate_open(raw("90 days", []), IDS) == Answer("90 days", [], False)
    assert validate_open(raw("90 days", ["u2"]), IDS) == Answer("90 days", ["u2"], False)
    assert validate(raw("90 days", []), IDS).code == "BAD_CITES"


@pytest.mark.parametrize(
    "text, code",
    [
        ("not json", "INVALID_JSON"),
        ("[1]", "INVALID_JSON"),
        ('{"answer": "x", "cites": []}', "BAD_KEYS"),
        ('{"answer": "x", "cites": [], "abstain": false, "why": 1}', "BAD_KEYS"),
        ('{"answer": "x", "answer": "y", "cites": [], "abstain": false}', "BAD_KEYS"),
        (raw("x", ["u3"]), "BAD_CITES"),
        (raw("x", ["u1", "u1"]), "BAD_CITES"),
        (raw("x", "u1"), "BAD_CITES"),
        (raw("x" * 201, []), "BAD_ANSWER"),
        (raw("  ", []), "BAD_ANSWER"),
        (raw(None, []), "BAD_ANSWER"),
        (raw(None, ["u1"], True), "BAD_ABSTAIN"),
        (raw("x", [], True), "BAD_ABSTAIN"),
        (raw("x", [], "no"), "BAD_ABSTAIN"),
    ],
)
def test_validate_open_rejects_everything_else(text, code):
    got = validate_open(text, IDS)
    assert got.code == code and validate(text, IDS).code == code


def test_open_agent_uncited_answer_end_to_end():
    def ou(units):
        return user(units, agent="open")

    b = ScriptedBackend([
        (ou(W_UNITS), answer("90 days")),  # from memory: cites []
        (ou([]), answer("90 days")),  # M: no cited units, so no context at all
        (CB, answer("90 days")),
    ])  # fmt: skip
    r = run_one(b, agent="open")
    assert r.status == "ok" and r.A == Answer("90 days", [], False)
    assert NO_CONTEXT in ou([]) and b.prompts.count(ou([])) == 3
    assert r.probes["R_remove"].majority == SAME and r.verdict == "decorative"
    assert r.flags["span_in_cites"] is False
    assert r.counterfactual["reason"] == "uncited"
    assert all(s == OPEN_SYSTEM for s, p in zip(b.systems, b.prompts, strict=True) if p != CB)

    # The same responses break the context agent's contract.
    r = run_one(ScriptedBackend([("", answer("90 days"))]))
    assert r.status == "non_compliant" and r.compliance["code"] == "BAD_CITES"


def test_open_agent_c_uses_open_prompt_and_validator():
    b = ScriptedBackend([
        (user(W_C, agent="open"), answer(f"{W_REPLACEMENT} days")),  # follows, uncited
        (user(W_UNITS, agent="open"), answer("90 days", "u2")),
        (user(W_UNITS[:1], agent="open"), answer("90 days")),
        (user(W_UNITS[1:], agent="open"), answer("90 days", "u2")),
        (CB, answer("90 days")),
    ])  # fmt: skip
    r = run_one(b, agent="open")
    assert r.verdict == "decorative" and r.flags["parametric"]
    assert r.counterfactual["verdict"] == "grounded"
    assert r.compliance["final_rejections"] == 0


def test_unknown_agent():
    with pytest.raises(ValueError):
        run_one(ScriptedBackend([]), agent="bogus")


def test_alignment_with_no_cites():
    assert alignment([], {"u1": SAME, "u2": OTHER}) == {
        "precision": None, "recall": 0.0, "load_bearing": ["u2"],
    }  # fmt: skip


# --- recompute and old results -----------------------------------------------------------


def test_recompute_rescores_c_from_first_k():
    c = [answer(f"{W_REPLACEMENT} days", "u2"), answer("90 days", "u2"), answer("90 days", "u2")]
    r = run_one(warranty_backend(c))
    assert r.counterfactual["verdict"] == "decorative"
    r1 = recompute(r, 1)
    assert r1.counterfactual["run"]["outcomes"] == [FOLLOWS]
    assert r1.counterfactual["verdict"] == "grounded"
    assert r1.counterfactual["replacement"] == W_REPLACEMENT
    assert recompute(r, 3).counterfactual == r.counterfactual
    assert r.counterfactual["run"]["outcomes"] == [FOLLOWS, SAME, SAME]  # stored run intact
    assert r1.compliance["final_rejections"] == 0


def test_recompute_keeps_not_applicable_and_drops_c_when_a_changes():
    b = warranty_backend(answer("x", "u2"), o_response=answer("90 days", "u1"))
    b.rules.insert(0, (user(W_UNITS[1:]), ABSTAIN_JSON))
    r = run_one(b)
    assert recompute(r, 1).counterfactual == r.counterfactual

    o = [answer("30 days", "u1"), answer("90 days", "u2"), answer("90 days", "u2")]
    b = warranty_backend(answer(f"{W_REPLACEMENT} days", "u2"), o_response=o, model="m2")
    r = run_one(b)
    assert r.counterfactual["verdict"] == "grounded"
    r1 = recompute(r, 1)
    assert r1.status == "k_unavailable" and r1.counterfactual is None


def test_recompute_c_unavailable_when_the_edit_target_changes():
    units = [Unit("u1", "The court sits in The Hague."), Unit("u2", "It was founded in 1945.")]
    q = "Where does the court sit?"
    hague = Record("h1", q, units, meta={"alternatives": ["Paris"]})
    edited = [Unit("u1", "The court sits in Paris."), units[1]]
    b = ScriptedBackend([
        (user(edited, q), answer("Paris", "u1")),
        (user(units, q), answer("The Hague", "u1")),
        (user(units[1:], q), ABSTAIN_JSON),
        (user(units[:1], q), answer("The Hague", "u1")),
        ("", ABSTAIN_JSON),
    ])  # fmt: skip
    r = run_one(b, record=hague)
    assert r.counterfactual["verdict"] == "grounded"
    assert r.counterfactual["target"]["text"] == "The Hague"
    # A canonically equal A ("Hague") that edits a different span: the stored C samples
    # answered a different context, so C is unavailable at k.
    o = r.probes["O"]
    hague_a = Answer("Hague", ["u1"], False)
    r.probes["O"] = ProbeRun(o.raw, [hague_a] * 3, o.outcomes, o.majority)
    r1 = recompute(r, 1)
    assert r1.status == "ok" and r1.A == hague_a
    assert r1.counterfactual == {
        "applicable": False, "reason": "k_changed_target", "replacement": None,
        "target": None, "seed": 0, "run": None, "verdict": None,
    }  # fmt: skip
    assert r1.verdict == "grounded"  # the removal verdict is unaffected


def test_supplied_uncited_answer_only_for_the_open_agent():
    uncited = Answer("90 days", [], False)
    rec_ = Record.from_json({**W.to_json(), "answer": uncited.to_json()})
    assert rec_.answer == uncited
    with pytest.raises(ValueError, match="BAD_ANSWER"):  # other rules still apply
        Record.from_json({**W.to_json(), "answer": {"text": "", "cites": [], "abstain": False}})
    for agent in ("context", "context_v1", "open_v1"):
        with pytest.raises(ValueError, match="only the open agent"):
            run_one(ScriptedBackend([]), record=rec_, agent=agent)
    b = ScriptedBackend([
        (user(W_UNITS, agent="open"), answer("90 days")),
        (user([], agent="open"), answer("90 days")),
        (CB, answer("90 days")),
    ])  # fmt: skip
    r = run_one(b, record=rec_, agent="open")
    assert r.supplied and r.A == uncited and r.flags["reproduced"]
    assert r.counterfactual["reason"] == "uncited" and r.verdict == "decorative"


def test_v1_result_json_loads_without_counterfactual():
    a = Answer("t", ["u1"], False)
    run = ProbeRun(raw=["{}"], parsed=[a], outcomes=[SAME], majority=SAME)
    res = Result(
        record=Record("r", "q", [Unit("u1", "t")]), model="m", k=1, temperature=None,
        status="ok", A=a, probes={"O": run}, verdict="grounded", flags={}, alignment={},
        compliance={"calls": 1, "retries": 0, "final_rejections": 0, "code": None},
    )  # fmt: skip
    v1 = res.to_json()
    assert v1.pop("counterfactual") is None
    loaded = Result.from_json(json.loads(json.dumps(v1)))
    assert loaded.counterfactual is None and loaded.verdict == "grounded"
    assert recompute(loaded, 1).counterfactual is None
