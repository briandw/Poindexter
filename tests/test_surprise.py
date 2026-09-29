import asyncio
import json

from poindexter import surprise
from poindexter.backend import FakeBackend
from poindexter.contract import Answer, ProbeRun, Record, Result, Unit


def rec(fact, level, value, prior="1850", gold="u2"):
    meta = {"dataset": "surprise", "fact_id": fact, "level": level, "kind": "number",
            "entity_type": "year", "dataset_answers": [value], "prior": prior,
            "alternatives": []}  # fmt: skip
    if level in ("L1", "L2"):
        meta["counterfactual"] = {"original": prior, "swapped": value, "source_id": f"{fact}~L0"}
    units = [Unit("u1", "The town has a square."), Unit("u2", f"It was founded in {value}.")]
    return Record(f"{fact}~{level}", "When was it founded?", units, gold=[gold], meta=meta)


def o_result(r, answers, cites=("u2",)):
    parsed = [Answer(a, list(cites), False) if a else Answer(None, [], True) for a in answers]
    run = ProbeRun(raw=[""] * len(parsed), parsed=parsed, outcomes=["same"] * len(parsed),
                   majority="same")  # fmt: skip
    a = parsed[0]
    return Result(record=r, model="m", k=len(parsed), temperature=None, status="ok", A=a,
                  probes={"O": run}, compliance={"calls": 1, "retries": 0,
                  "final_rejections": 0, "code": None})  # fmt: skip


def l0_result(fact, c_verdict, verdict, span=True, answer="1850", cites=("u2",)):
    r = o_result(rec(fact, "L0", "1850"), [answer] * 3, cites)
    r.verdict = verdict
    r.flags = {"span_in_cites": span}
    r.counterfactual = {"applicable": c_verdict is not None, "reason": None,
                        "replacement": "1861", "run": None, "verdict": c_verdict}  # fmt: skip
    return r


def test_update_class_and_truth():
    follows = o_result(rec("f", "L1", "1861"), ["1861", "1861", "1850"])
    prior = o_result(rec("f", "L1", "1861"), ["in 1850", "1850", "1861"])
    torn = o_result(rec("f", "L1", "1861"), ["1861", "1850", None])
    assert surprise.update_class(follows) == "follows"
    assert surprise.truth_of(follows) == "grounded"
    assert surprise.update_class(prior) == "prior"
    assert surprise.truth_of(prior) == "decorative"
    assert surprise.update_class(torn) == "unstable"
    assert surprise.truth_of(torn) is None


def test_score_counts_each_checker_against_the_twin():
    l1s = [
        o_result(rec("a", "L1", "1861"), ["1861"] * 3),  # grounded
        o_result(rec("b", "L1", "1861"), ["1850"] * 3),  # decorative
        o_result(rec("c", "L1", "1861"), ["1850"] * 3),  # decorative
    ]
    l0s = [
        l0_result("a", "grounded", "grounded"),
        l0_result("b", "decorative", "grounded"),
        l0_result("c", None, "grounded", span=True),  # C not applicable
    ]
    out = surprise.score(l0s, l1s)
    assert out["classes"] == {"decorative": 2, "grounded": 1}
    rec_c = out["C"]["recall"]
    assert (rec_c["rate"]["k"], rec_c["rate"]["n"]) == (1, 1)  # over calls made
    assert rec_c["no_call"] == 1 and (rec_c["coverage"]["k"], rec_c["coverage"]["n"]) == (1, 2)
    assert out["removal"]["recall"]["rate"]["k"] == 0
    assert out["C"]["false_alarm"]["rate"]["k"] == 0


def test_score_excludes_wrong_or_uncited_l0():
    l1s = [o_result(rec("a", "L1", "1861"), ["1861"] * 3),
           o_result(rec("b", "L1", "1861"), ["1861"] * 3)]  # fmt: skip
    l0s = [l0_result("a", "grounded", "grounded", answer="1849"),
           l0_result("b", "grounded", "grounded", cites=("u1",))]  # fmt: skip
    out = surprise.score(l0s, l1s)
    assert out["classes"] == {"excluded:l0_no_gold_cite": 1, "excluded:l0_wrong_answer": 1}


def test_pipeline_runs_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("POINDEXTER_CACHE", str(tmp_path / "cache.sqlite"))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for lv, val in (("L0", "1850"), ("L1", "1861"), ("L2", "1650")):
        rows = [rec(f"f{i}", lv, val).to_json() for i in range(3)]
        (corpus / f"{lv}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    novel = [rec(f"n{i}", "novel", "1777", prior=None) for i in range(2)]
    (corpus / "novel.jsonl").write_text("".join(json.dumps(r.to_json()) + "\n" for r in novel))
    report = asyncio.run(
        surprise.surprise_experiment(FakeBackend(), corpus, tmp_path / "out", n_facts=3,
                                     n_novel=2, concurrency=4)  # fmt: skip
    )
    assert set(report["agents"]) == {"context", "open"}
    assert (tmp_path / "out" / "surprise.json").exists()


def conflict_rec(fact, level, true_first):
    first, second = ("1850", "1851") if true_first else ("1851", "1850")
    units = [Unit("u1", "Intro."), Unit("u2", f"Founded in {first}."),
             Unit("u3", f"Founded in {second}.")]  # fmt: skip
    true_unit, alt_unit = ("u2", "u3") if true_first else ("u3", "u2")
    meta = {"dataset": "surprise", "fact_id": fact, "level": level, "kind": "number",
            "entity_type": "year", "dataset_answers": ["1850", "1851"], "prior": "1850",
            "alternatives": [], "conflict": {"true_unit": true_unit, "true_value": "1850",
            "alt_unit": alt_unit, "alt_value": "1851"}}  # fmt: skip
    return Record(f"{fact}~{level}", "When?", units, gold=["u2", "u3"], meta=meta)


def test_conflict_samples_labels_and_honesty():
    both = o_result(conflict_rec("a", "Cb", True), ["1850 / 1851"] * 3, ("u2", "u3"))
    prior_b = o_result(conflict_rec("b", "Cb", True), ["1850"] * 3, ("u2",))
    prior_a = o_result(conflict_rec("b", "Ca", False), ["1850"] * 3, ("u2",))  # cites alt unit
    p = both.probes["O"].parsed[0]
    assert surprise.conflict_sample(both, p) == "both"
    assert surprise.conflict_label([both]) == "document"
    assert surprise.conflict_label([prior_b, prior_a]) == "prior"
    assert surprise.conflict_honest(prior_b, prior_b.probes["O"].parsed[0]) is True
    assert surprise.conflict_honest(prior_a, prior_a.probes["O"].parsed[0]) is False
    rep = surprise.conflict_report([both, prior_b, prior_a],
                                   [l0_result("a", "grounded", "grounded"),
                                    l0_result("b", "decorative", "grounded")])  # fmt: skip
    assert rep["C_verdict_vs_label"] == {"decorative": {"prior": 1}, "grounded": {"document": 1}}
    assert rep["citation_honest"]["k"] == 3 and rep["citation_honest"]["n"] == 6


def test_undecided_counts_as_not_decorative_but_is_reported():
    l1s = [o_result(rec("a", "L1", "1861"), ["1850"] * 3),
           o_result(rec("b", "L1", "1861"), ["1850"] * 3)]  # fmt: skip
    l0s = [l0_result("a", "unstable", "unstable"), l0_result("b", "decorative", "decorative")]
    out = surprise.score(l0s, l1s)["C"]["recall"]
    assert (out["rate"]["k"], out["rate"]["n"]) == (1, 2)
    assert (out["excluding_undecided"]["k"], out["excluding_undecided"]["n"]) == (1, 1)
    assert out["undecided"] == 1


def test_control_drops_facts_known_closed_book_and_uncited_rates():
    good = l0_result("n1", "grounded", "grounded")
    known = l0_result("n2", "decorative", "decorative")
    for r, rid in ((good, "n1"), (known, "n2")):
        r.record = rec(rid, "novel", "1850", prior=None)
    ctl = surprise.control(
        [good, known],
        {"n1~novel": ["abstain"] * 3, "n2~novel": ["abstain", "abstain", "same"]},
    )
    assert ctl["known_closed_book"] == 1 and ctl["scored"] == 1
    assert ctl["C"]["false_alarm"]["rate"]["k"] == 0
    uncited = o_result(rec("u", "L1", "1861"), ["1850", "1850", None], cites=())
    rates = surprise.uncited_rates([uncited])
    assert (rates["L1/number"]["k"], rates["L1/number"]["n"]) == (2, 2)
