import asyncio
from pathlib import Path

from conftest import ABSTAIN_JSON, ScriptedBackend, answer

from poindexter.cli import main
from poindexter.contract import (
    ABSTAIN,
    OTHER,
    SAME,
    Answer,
    Record,
    Result,
    Unit,
    write_jsonl,
)
from poindexter.prompt import build_closed_book_prompt, build_prompt
from poindexter.report import aggregate, markdown
from poindexter.runner import run_record
from poindexter.verdict import finish, recompute, score

TOY = Path(__file__).parent.parent / "examples" / "toy.jsonl"
UNITS = [Unit("u1", "The store opened in 1998."), Unit("u2", "The warranty lasts 90 days.")]
Q = "How long is the warranty?"
CB = build_closed_book_prompt(Q)[1]
A = Answer("90 days", ["u2"], False)
SAMPLE = {SAME: A, OTHER: Answer("30 days", ["u1"], False), ABSTAIN: Answer(None, [], True)}


def run(*outcomes):
    return score(["raw"] * len(outcomes), [SAMPLE[o] for o in outcomes], A)


def ok_result(rid, r_remove, m, n=ABSTAIN, loo=None, retries=0):
    rec = Record(rid, Q, UNITS)
    probes = {"O": run(SAME), "N": run(n), "R_remove": run(r_remove), "M": run(m)}
    res = Result(
        record=rec, model="m", k=1, temperature=None, status="ok", A=A, probes=probes,
        loo={u: run(o) for u, o in (loo or {}).items()},
        compliance={"calls": 4 + retries, "retries": retries, "final_rejections": 0,
                    "code": None},
    )  # fmt: skip
    return finish(res)


def test_aggregate_hand_built():
    results = [
        ok_result("a", SAME, SAME, n=SAME),
        ok_result("b", ABSTAIN, SAME, loo={"u1": SAME, "u2": ABSTAIN}, retries=2),
        ok_result("c", OTHER, OTHER),
        Result(record=Record("d", Q, UNITS), model="m", k=1, temperature=None,
               status="non_compliant", compliance={"calls": 2, "retries": 1,
                                                   "final_rejections": 1, "code": "BAD_KEYS"}),
    ]  # fmt: skip
    agg = aggregate(results)
    assert agg["records"] == 4 and agg["judged"] == 3
    assert agg["status"] == {
        "ok": 3,
        "non_compliant": 1,
        "unstable_original": 0,
        "k_unavailable": 0,
    }
    assert {v: d["count"] for v, d in agg["verdicts"].items()} == {
        "decorative": 1, "grounded": 1, "incomplete": 1, "unstable": 0,
    }  # fmt: skip
    assert agg["flags"]["parametric"] == {"true": 1, "known": 3, "rate": 1 / 3}
    assert agg["flags"]["fabricates"]["true"] == 1
    assert agg["flags"]["position_sensitive"]["known"] == 0
    c = agg["compliance"]
    assert c["records_compliant"] == 0.75
    assert (c["samples"], c["retries"], c["final_rejections"]) == (13, 3, 1)
    assert c["first_try_pass_rate"] == 10 / 13 and c["final_pass_rate"] == 12 / 13
    assert agg["alignment"] == {
        "records": 1, "mean_precision": 1.0, "mean_recall": 1.0, "recall_defined": 1,
    }  # fmt: skip
    text = markdown(agg)
    assert "| decorative | 1 | 0.333 |" in text
    assert "## Rejections per probe" in text


def user(units):
    return build_prompt(units, Q)[1]


def test_report_k1_vs_k3_on_flipped_sample(tmp_path, capsys):
    b = ScriptedBackend([
        (user(UNITS), answer("90 days", "u2")),
        (user(UNITS[:1]), [answer("90 days", "u1"), ABSTAIN_JSON, ABSTAIN_JSON]),
        (user(UNITS[1:]), answer("90 days", "u2")),
        (CB, ABSTAIN_JSON),
    ])  # fmt: skip
    rec = Record("r", Q, UNITS)
    res = asyncio.run(run_record(rec, b, 3, None, 0, None, asyncio.Semaphore(4), "verdict"))
    assert res.verdict == "grounded"
    assert recompute(res, 1).verdict == "decorative"
    path = tmp_path / "results.jsonl"
    write_jsonl(path, [res.to_json()])

    main(["report", str(path)])
    k3 = capsys.readouterr().out
    main(["report", str(path), "--k", "1"])
    k1 = capsys.readouterr().out
    assert "| grounded | 1 | 1.000 |" in k3 and "| decorative | 0 | 0.000 |" in k3
    assert "| decorative | 1 | 1.000 |" in k1 and "| grounded | 0 | 0.000 |" in k1


def test_cli_run_then_report(tmp_path, capsys):
    out = tmp_path / "r.jsonl"
    main(["run", str(TOY), "--backend", "fake", "--k", "3", "--out", str(out)])
    assert "6 records" in capsys.readouterr().out
    main(["report", str(out), "--k", "1"])
    text = capsys.readouterr().out
    assert text.startswith("# Poindexter report")
    assert "| grounded | 3 | 0.600 |" in text
