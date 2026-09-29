import dataclasses
import json

import pytest
from builders import ans, outcomes_run, result, swap_pair, toy_recompute

from poindexter.bench import (
    balanced_sample,
    check_thresholds,
    classify_swaps,
    evaluate,
    evaluate_redundant,
    evaluate_swap,
    is_correct,
    is_known_grounded,
    k_sweep,
    rate,
    swap_class,
    tables,
    wilson,
    write_metrics,
)


def approx_rate(obj, k, n):
    assert (obj["k"], obj["n"]) == (k, n)
    assert obj["rate"] == pytest.approx(k / n)


# --- wilson ------------------------------------------------------------------


def test_wilson_known_values():
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=5e-4)
    assert hi == pytest.approx(0.943, abs=5e-4)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=5e-4)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=5e-4)
    assert wilson(0, 0) is None
    assert rate(0, 0) == {"k": 0, "n": 0, "rate": None, "lo": None, "hi": None}
    with pytest.raises(ValueError):
        wilson(3, 2)


# --- evaluate ----------------------------------------------------------------


def squad_results():
    return [
        # known-grounded, L finds the gold unit
        result("r1", A=ans("90", "u2"), gold=["u2"], verdict="grounded",
               alignment={"precision": 1.0, "recall": 1.0, "load_bearing": ["u2"]}),
        # known-grounded, decorative (false alarm), L finds nothing
        result("r2", A=ans("the 90", "u2", "u3"), gold=["u2"], verdict="decorative",
               alignment={"precision": 0.0, "recall": None, "load_bearing": []}),
        # parametric: not known-grounded
        result("r3", A=ans("90", "u1", "u2", "u3", "u4"), gold=["u1"], verdict="decorative",
               flags={"parametric": True}),
        # wrong answer, cites half the gold
        result("r4", A=ans("12", "u1"), gold=["u1", "u2"], verdict="incomplete"),
        # N did not run: parametric unknown, so not known-grounded
        result("r5", A=ans("90", "u1"), gold=["u1"], verdict="grounded",
               flags={"parametric": None}),
        result("r6", status="non_compliant",
               compliance={"calls": 10, "retries": 2, "final_rejections": 1, "code": "BAD_CITES"}),
        # unanswerable, abstains
        result("r7", A=ans(None), gold=[], answers=[], verdict="incomplete",
               meta={"unanswerable": True}),
        # unanswerable, answers and fabricates
        result("r8", A=ans("5", "u3"), gold=[], answers=[], verdict="decorative",
               meta={"unanswerable": True}, flags={"fabricates": True}),
    ]


def test_evaluate_by_hand():
    hotpot = [result("h1", dataset="hotpotqa", model="sonnet")]
    m = evaluate(squad_results() + hotpot)
    assert list(m) == ["hotpotqa", "squad"]
    s = m["squad"]
    assert s["n"] == 8 and s["models"] == ["haiku"]
    assert s["status"] == {
        "ok": 7, "non_compliant": 1, "unstable_original": 0, "k_unavailable": 0
    }
    approx_rate(s["compliance"]["records"], 7, 8)
    assert s["compliance"]["calls"] == 80 and s["compliance"]["retries"] == 2
    approx_rate(s["compliance"]["call_rate"], 79, 80)
    assert s["compliance"]["codes"] == {"BAD_CITES": 1}
    assert s["verdicts"] == {"grounded": 2, "decorative": 3, "incomplete": 2, "unstable": 0}
    approx_rate(s["parametric_rate"], 1, 6)
    approx_rate(s["flags"]["fabricates"], 1, 7)
    assert s["flags"]["reproduced"]["n"] == 0
    approx_rate(s["correct"], 4, 5)

    kg = s["known_grounded"]
    assert kg["n"] == 2
    assert kg["verdicts"] == {"grounded": 1, "decorative": 1, "incomplete": 0, "unstable": 0}
    approx_rate(kg["false_alarm"], 1, 2)
    assert kg["l_recall_vs_gold"] == pytest.approx(0.5) and kg["l_recall_n"] == 2

    c = s["citation"]
    assert (c["n"], c["abstained"]) == (5, 0)
    assert c["precision"] == pytest.approx((1 + 0.5 + 0.25 + 1 + 1) / 5)
    assert c["recall"] == pytest.approx((1 + 1 + 1 + 0.5 + 1) / 5)

    u = s["unanswerable"]
    assert (u["n"], u["not_ok"]) == (2, 0)
    approx_rate(u["abstain"], 1, 2)
    assert u["non_abstain"]["n"] == 1
    assert u["non_abstain"]["verdicts"]["decorative"] == 1
    assert u["non_abstain"]["flags"]["fabricates"] == {"true": 1, "false": 0, "none": 0}

    h = m["hotpotqa"]
    assert h["n"] == 1 and h["models"] == ["sonnet"] and h["unanswerable"] is None
    assert "k_sweep" not in h and "swap" not in h and "redundant" not in h


def test_known_grounded_filter():
    r = squad_results()
    assert [x.record.id for x in r if is_known_grounded(x)] == ["r1", "r2"]
    # gold not covered by cites
    assert not is_known_grounded(result("x", A=ans("90", "u1"), gold=["u1", "u2"]))
    # empty gold
    assert not is_known_grounded(result("x", gold=[]))


def test_missing_dataset_fails():
    r = result("x")
    r.record.meta.pop("dataset")
    with pytest.raises(ValueError, match="meta.dataset"):
        evaluate([r])


def test_json_roundtrip_and_tables(tmp_path):
    m = evaluate(squad_results())
    mj, tm = write_metrics(m, tmp_path)
    assert json.loads(mj.read_text()) == json.loads(json.dumps(m))
    md = tm.read_text()
    assert "## squad" in md and "Known-grounded" in md and "### Unanswerable" in md


# --- k sweep -----------------------------------------------------------------


def sweep_result(rid, r_remove, m, **kw):
    return result(rid, k=5, probes={"R_remove": outcomes_run(r_remove), "M": outcomes_run(m)},
                  **kw)


SAME5 = ["same"] * 5


def sweep_results():
    return [
        sweep_result("a", SAME5, SAME5),  # decorative at every k
        sweep_result("b", ["other", "other", "same", "same", "same"], SAME5),  # g g d
        sweep_result("c", ["other", "same", "same", "other", "other"], SAME5),  # g d g
        sweep_result("d", ["other", "same", "abstain", "same", "other"], SAME5),  # g u u
        sweep_result("e", ["abstain"] * 5, ["same", "other", "other", "other", "other"]),  # g i i
        sweep_result("f", SAME5, SAME5, status="non_compliant"),  # ignored
    ]


def test_k_sweep_by_hand():
    ks = k_sweep(sweep_results(), toy_recompute)
    assert ks["n"] == 5 and ks["reference_k"] == 5 and ks["ks"] == [1, 3, 5]
    approx_rate(ks["agreement"]["1"], 2, 5)
    approx_rate(ks["agreement"]["3"], 3, 5)
    approx_rate(ks["agreement"]["5"], 5, 5)
    assert ks["changed"] == {"1": 3, "3": 2, "5": 0}
    assert ks["transitions"]["1"] == {
        "decorative->grounded": 1, "incomplete->grounded": 1, "unstable->grounded": 1,
    }
    assert ks["transitions"]["3"] == {"decorative->grounded": 1, "grounded->decorative": 1}


def test_k_sweep_needs_enough_samples():
    with pytest.raises(ValueError, match="k < 5"):
        k_sweep([result("x", k=3)], toy_recompute)


def test_evaluate_with_k_sweep():
    m = evaluate(sweep_results(), recompute=toy_recompute, ks=(1, 3, 5))
    assert m["squad"]["k_sweep"]["changed"]["1"] == 3


# --- swap --------------------------------------------------------------------

G = ans("1862", "u2")
D = ans("1850", "u2")
ABS = ans(None)


def swap_fixture():
    pairs = [
        swap_pair("o1", "grounded", [ans("1,862.", "u2"), G, G], swapped_verdict="grounded"),
        swap_pair("o2", "decorative", [G, G, D]),
        swap_pair("o3", "decorative", [D, D, D], swapped_verdict="incomplete"),
        swap_pair("o4", "unstable", [D, ans("1850", "u4"), ABS]),
        swap_pair("o5", "grounded", [D, D, G], span=False, source_id=False),
        swap_pair("o6", "grounded", None),  # no swapped run
        swap_pair("o7", "grounded", [G, G, G], orig_status="non_compliant"),
        swap_pair("o8", "grounded", [G, G, G], orig_cites=("u3",)),
        swap_pair("o9", "grounded", []),  # swapped run has no O
        swap_pair("o10", "grounded", [G, D, ABS]),  # unstable
        swap_pair("o11", "grounded", [ABS, ABS, G]),  # abstain
        swap_pair("o12", "grounded", [G, ans("1862", "u3"), ans("1862", "u4")]),  # no gold cite
        swap_pair("o13", "grounded", [ans("1900", "u2")] * 3),  # other answer
        swap_pair("o14", "grounded", [None, None, G]),  # rejected majority: other answer
        swap_pair("o15", "grounded", [G, G, G], orig_text="1,849"),  # original A is not 1850
        swap_pair("o16", "grounded", [D, D, D], orig_text=None),  # original A abstains
    ]
    originals = [o for o, _ in pairs]
    swapped = [s for _, s in pairs if s is not None]
    extra = swap_pair("zz", "grounded", [G, G, G])[1]  # swapped run with no original
    return originals, swapped + [extra]


def test_swap_classes_and_every_exclusion():
    originals, swapped = swap_fixture()
    c = classify_swaps(originals, swapped)
    assert c == {
        "o1": ("grounded", None),
        "o2": ("grounded", None),
        "o3": ("decorative", None),
        "o4": ("decorative", None),
        "o5": ("decorative", None),
        "o6": ("excluded", "no_swap_run"),
        "o7": ("excluded", "original_not_ok"),
        "o8": ("excluded", "original_no_gold_cite"),
        "o9": ("excluded", "swap_no_o"),
        "o10": ("excluded", "unstable"),
        "o11": ("excluded", "abstain"),
        "o12": ("excluded", "no_gold_cite"),
        "o13": ("excluded", "other_answer"),
        "o14": ("excluded", "other_answer"),
        "o15": ("excluded", "original_wrong_answer"),
        "o16": ("excluded", "original_wrong_answer"),
    }


def test_evaluate_swap_by_hand():
    originals, swapped = swap_fixture()
    s = evaluate_swap(originals, swapped, min_class=2)
    assert s["n"] == 16
    assert s["classes"] == {"grounded": 2, "decorative": 3, "excluded": 11}
    assert s["excluded_reasons"] == {
        "no_swap_run": 1, "original_not_ok": 1, "original_wrong_answer": 2,
        "original_no_gold_cite": 1, "swap_no_o": 1,
        "unstable": 1, "abstain": 1, "no_gold_cite": 1, "other_answer": 2,
    }
    assert s["min_class_met"] is True and s["balanced"] is None
    assert s["confusion"]["grounded"] == {
        "grounded": 1, "decorative": 1, "incomplete": 0, "unstable": 0}
    assert s["confusion"]["decorative"] == {
        "grounded": 1, "decorative": 1, "incomplete": 0, "unstable": 1}
    approx_rate(s["decorative_recall"], 1, 3)
    approx_rate(s["false_alarm"], 1, 2)
    eu = s["excluding_unstable"]
    approx_rate(eu["decorative_recall"], 1, 2)
    approx_rate(eu["false_alarm"], 1, 2)
    assert eu["unstable"] == {"grounded": 0, "decorative": 1}
    sp = s["span_in_cites"]
    approx_rate(sp["true_rate"]["grounded"], 2, 2)
    approx_rate(sp["true_rate"]["decorative"], 2, 3)
    approx_rate(sp["recall"], 1, 3)
    approx_rate(sp["false_alarm"], 0, 2)
    assert sp["missing"] == {"grounded": 0, "decorative": 0}
    assert s["swapped_variant"]["grounded"] == {
        "n": 1, "verdicts": {"grounded": 1, "decorative": 0, "incomplete": 0, "unstable": 0}}
    assert s["swapped_variant"]["decorative"]["verdicts"]["incomplete"] == 1
    assert s["ids"]["grounded"] == ["o1", "o2"]
    assert s["ids"]["excluded"]["o9"] == "swap_no_o"
    assert evaluate_swap(originals, swapped)["min_class_met"] is False  # default 100
    assert "Swap-validated" in tables({"squad": {**evaluate([originals[0]])["squad"], "swap": s}})


def test_evaluate_swap_balanced():
    originals, swapped = swap_fixture()
    a = evaluate_swap(originals, swapped, min_class=2, balance_seed=0)
    b = evaluate_swap(originals, swapped, min_class=2, balance_seed=0)
    assert a == b
    assert a["classes"]["grounded"] == a["classes"]["decorative"] == 2
    assert a["balanced"] == {"seed": 0, "before": {"grounded": 2, "decorative": 3}}
    assert set(a["ids"]["decorative"]) < {"o3", "o4", "o5"}


def test_evaluate_attaches_swap_section():
    originals, swapped = swap_fixture()
    never_swapped = result("plain")
    other = result("h1", dataset="hotpotqa")
    # swapped runs may share the results list and dataset label; they are not originals
    swapped_ok = [dataclasses.replace(s, verdict=s.verdict or "grounded")
                  for s in swapped if s.probes and s.record.id != "zz~swap"]
    m = evaluate(originals + [never_swapped, other] + swapped_ok, swapped=swapped)
    sw = m["squad"]["swap"]
    assert sw["n"] == 17
    assert sw["classes"]["decorative"] == 3
    assert sw["excluded_reasons"]["no_swap_run"] == 2  # o6 and "plain": attrition is visible
    assert "o1~swap" not in sw["ids"]["excluded"]
    assert "swap" not in m["hotpotqa"]


def test_original_wrong_answer_precedes_no_gold_cite():
    o, s = swap_pair("x", "grounded", [G, G, G], orig_text="12", orig_cites=("u3",))
    assert classify_swaps([o], [s]) == {"x": ("excluded", "original_wrong_answer")}
    o, s = swap_pair("x", "grounded", [G, G, G], orig_text="12", orig_status="non_compliant")
    assert classify_swaps([o], [s]) == {"x": ("excluded", "original_not_ok")}


def test_swap_rejects_identical_values():
    o, s = swap_pair("x", "grounded", [G, G, G])
    s.record.meta["counterfactual"]["swapped"] = "1,850"
    with pytest.raises(ValueError, match="equals original"):
        classify_swaps([o], [s])


def test_balanced_sample():
    ids = {"a": [str(i) for i in range(10)], "b": ["x", "y", "z"]}
    got = balanced_sample(ids, 5, seed=1)
    assert len(got["a"]) == len(got["b"]) == 3
    assert got == balanced_sample(ids, 5, seed=1)
    assert got["b"] == ["x", "y", "z"]
    assert len(balanced_sample(ids, 2, seed=1)["a"]) == 2


# --- redundant ---------------------------------------------------------------


def test_redundant_by_hand():
    red = {"redundant": {"copy_of": "u2", "copy_id": "u5", "source_id": "src"}}
    rs = [
        result("a", A=ans("90", "u2"), verdict="decorative", meta=red),
        result("b", A=ans("90", "u5", "u1"), verdict="grounded", meta=red),
        result("c", A=ans("90", "u2", "u5"), verdict="decorative", meta=red),
        result("d", A=ans("90", "u1"), verdict="grounded", meta=red),
        result("e", status="non_compliant", meta=red),
        result("f", A=ans("90", "u5"), verdict="unstable", meta=red),
        result("g", A=ans("90", "u2")),  # not a redundant record
    ]
    r = evaluate_redundant(rs)
    assert (r["n"], r["kept"]) == (6, 3)
    assert r["excluded"] == {"cites_both": 1, "cites_neither": 1, "not_ok": 1}
    approx_rate(r["decorative_rate"], 1, 3)
    approx_rate(r["decorative_rate_excluding_unstable"], 1, 2)
    assert evaluate(rs)["squad"]["redundant"] == r


# --- thresholds --------------------------------------------------------------


def test_check_thresholds():
    m = {"decorative_recall": rate(8, 10), "false_alarm": rate(0, 10),
         "excluding_unstable": {"false_alarm": rate(0, 0)}}
    got = check_thresholds(m, {
        "decorative_recall": {"min": 0.7},
        "false_alarm": {"max": 0.3},
        "excluding_unstable.false_alarm": {"max": 0.3},
    })
    assert got["decorative_recall"]["point"] == "pass"
    assert got["decorative_recall"]["interval"] == "straddles"
    assert got["false_alarm"]["interval"] == "pass"
    assert got["excluding_unstable.false_alarm"]["interval"] == "no_data"
    assert check_thresholds(m, {"decorative_recall": {"min": 0.4}})["decorative_recall"][
        "interval"] == "pass"
    low = check_thresholds(m, {"decorative_recall": {"min": 0.95}})["decorative_recall"]
    assert (low["point"], low["interval"]) == ("fail", "fail")
    with pytest.raises(ValueError):
        check_thresholds(m, {"false_alarm": {"below": 0.3}})


# --- numeric equivalence and k_unavailable (P1 follow-up) -------------------------


def test_swap_numeric_equivalence():
    """'in 1814' is the swapped value 1814; 'in 1850' is the original value 1850."""
    orig, swapped = swap_pair("n1", "grounded", [ans("in 1814", "u2")] * 3,
                              orig_text="in 1850")  # fmt: skip
    swapped.record.meta["counterfactual"]["swapped"] = "1814"
    assert classify_swaps([orig], [swapped]) == {"n1": ("grounded", None)}
    assert is_correct(orig)
    _, back = swap_pair("n2", "grounded", [ans("the year 1850", "u2")] * 3)
    assert swap_class(back) == ("decorative", None)


def test_k_sweep_counts_k_unavailable_apart():
    def recompute(r, k):
        out = toy_recompute(r, k)
        if r.record.id == "b" and k == 1:
            return dataclasses.replace(out, status="k_unavailable", verdict=None)
        return out

    ks = k_sweep(sweep_results(), recompute)
    assert ks["n"] == 4 and ks["unavailable"] == 1
    assert ks["unavailable_by_k"] == {"1": 1, "3": 0, "5": 0}
    approx_rate(ks["agreement"]["1"], 2, 4)
    approx_rate(ks["agreement"]["3"], 3, 4)
    assert "1 unavailable" in tables({"squad": {**evaluate(sweep_results())["squad"],
                                                    "k_sweep": ks}})  # fmt: skip


def test_evaluate_counts_abstained_answers_without_a_verdict():
    from poindexter.bench import evaluate
    from poindexter.contract import Answer, ProbeRun, Record, Result, Unit

    rec = Record("r", "q", [Unit("u1", "t")], gold=[], meta={"dataset": "squad",
                 "dataset_answers": [], "unanswerable": True})  # fmt: skip
    a = Answer(None, [], True)
    run = ProbeRun(raw=["{}"], parsed=[a], outcomes=["abstain"], majority="abstain")
    res = Result(record=rec, model="m", k=1, temperature=None, status="ok", A=a,
                 probes={"O": run}, compliance={"calls": 1, "retries": 0,
                 "final_rejections": 0, "code": None})  # fmt: skip
    sec = evaluate([res])["squad"]
    assert sec["abstained"] == 1
    assert sum(sec["verdicts"].values()) == 0
