import asyncio
import json

from poindexter import sweep
from poindexter.backend import FakeBackend
from poindexter.contract import Answer, ProbeRun, Record, Result, Unit
from poindexter.prompt import AGENTS, SYSTEM, VALIDATORS


def l0(fid="f1", prior="1850", etype="year", kind="number"):
    units = [Unit("u1", "The town has a square built in 1790."),
             Unit("u2", f"The bridge was opened in {prior}."),
             Unit("u3", "It crosses the river.")]  # fmt: skip
    meta = {"dataset": "surprise", "fact_id": fid, "kind": kind, "entity_type": etype,
            "prior": prior, "alternatives": ["Anna Ruiz", "Bo Lind", "Cy Oak"],
            "level": "L0", "dataset_answers": [prior], "heldout_values": ["x", "y"]}  # fmt: skip
    return Record(f"sp-{fid}~L0", "When was the bridge opened?", units, gold=["u2"], meta=meta)


def fact(prior="1850", etype="year", kind="number"):
    return {"fact_id": "f1", "answer": prior, "kind": kind, "entity_type": etype,
            "mild": "Bo Lind", "strong": "Cy Oak", "alternatives": ["Anna Ruiz", "Bo Lind"]}


def test_year_sizes_stay_in_their_windows_and_avoid_mentions():
    rec = l0()
    for size, (lo, hi) in sweep.YEAR_SIZES.items():
        for seed in range(20):
            v = sweep.year_value(1850, size, rec, seed)
            assert v is not None and lo <= abs(v - 1850) <= hi and v != 1790


def test_count_sizes():
    rec = l0(prior="46", etype="count")
    for seed in range(20):
        s1 = sweep.count_value(46, "S1", rec, seed)
        s5 = sweep.count_value(46, "S5", rec, seed)
        assert 1 <= abs(s1 - 46) <= 5
        assert s5 % 46 == 0 and 20 <= s5 // 46 <= 50


def test_sweep_records_edit_only_the_answer_sentence():
    rec = l0()
    out = sweep.sweep_records(rec, fact(), seed=0)
    assert [r.meta["size"] for r in out] == list(sweep.YEAR_SIZES)
    for r in out:
        assert r.units[0] == rec.units[0] and r.units[2] == rec.units[2]
        assert r.meta["dataset_answers"][0] in r.units[1].text
        assert "1850" not in r.units[1].text
        assert r.meta["counterfactual"]["source_id"] == rec.id
        assert "heldout_values" not in r.meta


def test_entity_sizes():
    rec = l0(prior="Ed Voss", etype="person", kind="entity")
    rec.units[1] = Unit("u2", "The bridge was opened by Ed Voss.")
    out = sweep.sweep_records(rec, fact(prior="Ed Voss", etype="person", kind="entity"), 0)
    values = {r.meta["size"]: r.meta["dataset_answers"][0] for r in out}
    assert values == {"E1": "Bo Lind", "E2": "Anna Ruiz", "E3": "Cy Oak"}


def result(rec, answers):
    parsed = [Answer(a, ["u2"], False) for a in answers]
    run = ProbeRun(raw=[""] * 3, parsed=parsed, outcomes=["same"] * 3, majority="same")
    return Result(record=rec, model="m", k=3, temperature=None, status="ok", A=parsed[0],
                  probes={"O": run}, compliance={})  # fmt: skip


def test_curve_agreement_and_tipping():
    recs = {r.meta["size"]: r for r in sweep.sweep_records(l0(), fact(), 0)}
    runs = []
    for size, r in recs.items():
        v = r.meta["dataset_answers"][0]
        follow = size in ("S1", "S2")
        runs.append(result(r, [v if follow else "1850"] * 3))
    c = sweep.curve(runs, {"f1"})
    assert c["number/S1"]["follow_rate"]["k"] == 1 and c["number/S3"]["follow_rate"]["k"] == 0
    a = sweep.agreement(runs, {"f1"})["number"]
    assert a["S1~S2"]["k"] == 1 and a["S2~S3"]["k"] == 0
    assert sweep.tipping(runs, {"f1"}) == {"S3": 1}
    assert sweep.curve(runs, set()) == {}


def test_document_agent_prompt():
    doc = AGENTS["document"]
    assert doc != SYSTEM and "the context's answer is still the correct answer" in doc
    assert VALIDATORS["document"] is VALIDATORS["context"]


def test_pipeline_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("POINDEXTER_CACHE", str(tmp_path / "cache.sqlite"))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rec = l0()
    (corpus / "L0.jsonl").write_text(json.dumps(rec.to_json()) + "\n")
    (corpus / "facts.jsonl").write_text(json.dumps(fact()) + "\n")
    sweep.build_sweep(corpus, tmp_path / "sw")
    report = asyncio.run(
        sweep.sweep_experiment(FakeBackend(), corpus, tmp_path / "sw", tmp_path / "out",
                               n_facts=5, concurrency=2)  # fmt: skip
    )
    assert set(report["agents"]) == set(sweep.AGENTS)
    assert (tmp_path / "out" / "sweep.json").exists()


def test_count_windows_hold_after_rounding():
    rec = l0(prior="46", etype="count")
    for size, (mode, lo, hi) in sweep.COUNT_SIZES.items():
        if mode != "share":
            continue
        for seed in range(50):
            v = sweep.count_value(46, size, rec, seed)
            assert v is None or lo <= abs(v - 46) / 46 <= hi


def test_entity_check_ignores_punctuation():
    rec = l0(prior="Ed Voss", etype="person", kind="entity")
    rec.units[1] = Unit("u2", "The bridge was opened by Ed Voss with Anna-Ruiz.")
    f = fact(prior="Ed Voss", etype="person", kind="entity")
    assert sweep.entity_value(f, "E2", rec, 0) is None  # Anna Ruiz is already mentioned


def test_tipping_unknown_when_a_size_is_undecided():
    recs = {r.meta["size"]: r for r in sweep.sweep_records(l0(), fact(), 0)}
    runs = [result(r, [r.meta["dataset_answers"][0]] * 3) for s, r in recs.items() if s != "S2"]
    runs.append(result(recs["S2"], ["abstain?"] * 3))
    assert sweep.tipping(runs, {"f1"}) == {"unknown": 1}
