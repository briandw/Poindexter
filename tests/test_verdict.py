import itertools
from dataclasses import replace

import pytest

from poindexter.contract import (
    ABSTAIN,
    OTHER,
    SAME,
    UNSTABLE,
    Answer,
    ProbeRun,
    Record,
    Result,
    Unit,
)
from poindexter.verdict import alignment, finish, flags, recompute, score, verdict

A = Answer("90 days", ["u2"], False)
UNITS = [Unit("u1", "It rained."), Unit("u2", "The warranty lasts 90 days.")]

SAME_A = Answer("90 Days", ["u2"], False)
OTHER_A = Answer("30 days", ["u1"], False)
ABST_A = Answer(None, [], True)
SAMPLE = {SAME: SAME_A, OTHER: OTHER_A, ABSTAIN: ABST_A}


def run(*outcomes):
    """A ProbeRun whose samples produce the given outcomes against A."""
    parsed = [SAMPLE[o] for o in outcomes]
    return score(["raw"] * len(parsed), parsed, A)


def expected_verdict(r, m):
    if UNSTABLE in (r, m):
        return UNSTABLE
    if r == SAME:
        return "decorative"
    return "grounded" if m == SAME else "incomplete"


@pytest.mark.parametrize(
    "r,m", list(itertools.product([SAME, ABSTAIN, OTHER, UNSTABLE], repeat=2))
)
def test_verdict_table(r, m):
    assert verdict({"R_remove": r, "M": m}) == expected_verdict(r, m)


def test_verdict_table_rows_explicitly():
    assert verdict({"R_remove": SAME, "M": OTHER}) == "decorative"
    assert verdict({"R_remove": ABSTAIN, "M": SAME}) == "grounded"
    assert verdict({"R_remove": OTHER, "M": SAME}) == "grounded"
    assert verdict({"R_remove": OTHER, "M": ABSTAIN}) == "incomplete"
    assert verdict({"R_remove": UNSTABLE, "M": SAME}) == UNSTABLE
    assert verdict({"R_remove": SAME, "M": UNSTABLE}) == UNSTABLE


def test_score_majority_and_k2_tie():
    assert run(SAME, SAME, OTHER).majority == SAME
    assert run(SAME, OTHER).majority == UNSTABLE
    assert run(SAME, OTHER, ABSTAIN).majority == UNSTABLE
    rejected = score(["x", "y"], [None, SAME_A], A)
    assert rejected.outcomes == [OTHER, SAME]
    assert rejected.rejections == 1


def base_probes(**over):
    probes = {
        "O": run(SAME, SAME, SAME),
        "N": run(ABSTAIN, ABSTAIN, ABSTAIN),
        "R_remove": run(ABSTAIN, ABSTAIN, ABSTAIN),
        "R_replace": run(ABSTAIN, ABSTAIN, ABSTAIN),
        "M": run(SAME, SAME, SAME),
        "S": run(SAME, SAME, SAME),
    }
    probes.update(over)
    return probes


def test_flags_all_false_baseline():
    f = flags(base_probes(), A, UNITS, supplied=False)
    assert f == {
        "parametric": False,
        "fabricates": False,
        "fabricates_on_replace": False,
        "position_sensitive": False,
        "span_in_cites": True,
        "reproduced": None,
    }


def test_each_flag():
    assert flags(base_probes(N=run(SAME, SAME, OTHER)), A, UNITS, False)["parametric"]
    assert flags(base_probes(R_remove=run(OTHER, OTHER, SAME)), A, UNITS, False)["fabricates"]
    assert not flags(base_probes(R_remove=run(ABSTAIN, ABSTAIN, OTHER)), A, UNITS, False)[
        "fabricates"
    ]
    f = flags(base_probes(R_replace=run(OTHER, OTHER, OTHER)), A, UNITS, False)
    assert f["fabricates_on_replace"]
    assert flags(base_probes(S=run(OTHER, OTHER, SAME)), A, UNITS, False)["position_sensitive"]
    moved = Answer("90 days", ["u1", "u2"], False)
    s_cites = score(["r"] * 3, [moved, moved, SAME_A], A)
    assert s_cites.majority == SAME
    assert flags(base_probes(S=s_cites), A, UNITS, False)["position_sensitive"]
    both = Answer("90 days", ["u1", "u2"], False)
    o_both = score(["r"] * 3, [both, both, SAME_A], A)
    s_both = score(["r"] * 3, [both, both, both], A)
    assert not flags(base_probes(O=o_both, S=s_both), A, UNITS, False)["position_sensitive"]
    assert flags(base_probes(O=o_both), A, UNITS, False)["position_sensitive"]
    # S compared with O, not A: a supplied A that O and S both disagree with is not it.
    o_other = score(["r"] * 3, [OTHER_A] * 3, A)
    s_other = score(["r"] * 3, [OTHER_A] * 3, A)
    assert not flags(base_probes(O=o_other, S=s_other), A, UNITS, True)["position_sensitive"]
    assert flags(base_probes(O=o_other), A, UNITS, True)["position_sensitive"]
    wrong_cite = Answer("90 days", ["u1"], False)
    assert not flags(base_probes(), wrong_cite, UNITS, False)["span_in_cites"]
    assert flags(base_probes(), A, UNITS, True)["reproduced"] is True
    assert flags(base_probes(O=run(OTHER, OTHER, SAME)), A, UNITS, True)["reproduced"] is False


def test_flags_without_optional_probes_are_none():
    probes = base_probes()
    del probes["R_replace"], probes["S"]
    f = flags(probes, A, UNITS, False)
    assert f["fabricates_on_replace"] is None
    assert f["position_sensitive"] is None


def test_alignment():
    al = alignment(["u1", "u2"], {"u1": SAME, "u2": ABSTAIN, "u3": OTHER})
    assert al == {"precision": 0.5, "recall": 0.5, "load_bearing": ["u2", "u3"]}
    al = alignment(["u1"], {"u1": SAME, "u2": SAME})
    assert al == {"precision": 0.0, "recall": None, "load_bearing": []}
    assert alignment(["u1"], {"u1": UNSTABLE})["load_bearing"] == ["u1"]


def make_result(o, **probes):
    record = Record("r", "How long is the warranty?", UNITS, gold=["u2"])
    res = Result(
        record=record, model="m", k=3, temperature=None, status="ok", A=A,
        probes=base_probes(O=o, **probes),
        loo={"u1": run(SAME, SAME, SAME), "u2": run(SAME, ABSTAIN, ABSTAIN)},
        compliance={"calls": 30, "retries": 2, "final_rejections": 0, "code": None},
    )  # fmt: skip
    return finish(res)


EXACT_A = Answer("90 days", ["u2"], False)


def test_recompute_from_first_k_samples():
    o = score(["raw"] * 3, [EXACT_A, SAME_A, SAME_A], A)
    full = make_result(o, R_remove=run(ABSTAIN, SAME, SAME), M=run(SAME, OTHER, OTHER))
    assert full.verdict == "decorative"
    assert full.alignment["load_bearing"] == ["u2"]
    one = recompute(full, 1)
    assert one.status == "ok" and one.k == 1 and one.verdict == "grounded"
    assert one.alignment["load_bearing"] == []
    assert all(len(p.raw) == 1 for p in one.probes.values())
    assert one.compliance == {"calls": None, "retries": None, "final_rejections": 0, "code": None}
    same = recompute(full, 3)
    assert (same.A, same.verdict, same.flags, same.alignment) == (
        full.A, full.verdict, full.flags, full.alignment
    )
    with pytest.raises(ValueError):
        recompute(full, 4)


def test_recompute_equal_by_canonical_text_is_available():
    two = Answer("two", ["u2"], False)
    a2 = Answer("2", ["u2"], False)
    o = score(["raw"] * 3, [two, a2, a2], a2)
    full = replace(make_result(o), A=a2)
    one = recompute(full, 1)
    assert one.status == "ok" and one.A == two and one.verdict is not None


def test_recompute_k_unavailable_when_a_changes():
    o = score(["raw"] * 3, [OTHER_A, SAME_A, SAME_A], A)
    full = make_result(o)
    one = recompute(full, 1)
    assert one.status == "k_unavailable" and one.A == OTHER_A
    assert list(one.probes) == ["O"] and one.loo == {}
    assert one.verdict is None and one.flags is None and one.alignment is None
    moved = Answer("90 days", ["u1"], False)
    full = make_result(score(["raw"] * 3, [moved, SAME_A, SAME_A], A))
    assert recompute(full, 1).status == "k_unavailable"  # same text, other cites


def test_recompute_supplied_answer_stays():
    o = score(["raw"] * 3, [OTHER_A, SAME_A, SAME_A], A)
    full = replace(make_result(o), supplied=True)
    one = recompute(full, 1)
    assert one.status == "ok" and one.A == A and one.flags["reproduced"] is False


def test_recompute_to_non_compliant_rederives_code():
    o = score(["not json", '{"answer": 1}', "raw"], [None, None, SAME_A], A)
    full = make_result(o)
    assert full.status == "ok"
    for k, status in [(1, "non_compliant"), (2, "non_compliant")]:
        got = recompute(full, k)
        assert got.status == status and got.A is None and list(got.probes) == ["O"]
        assert got.compliance["code"] == "INVALID_JSON" and got.verdict is None
    nc = Result(record=full.record, model="m", k=2, temperature=None, status="non_compliant",
                probes={"O": ProbeRun(["x", "y"], [None, None], [], UNSTABLE, 2)},
                compliance={})  # fmt: skip
    assert recompute(nc, 1).status == "non_compliant"
