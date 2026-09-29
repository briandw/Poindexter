import asyncio
import json
from pathlib import Path

from poindexter import datasets, experiment
from poindexter.backend import FakeBackend

FIXTURE = Path(__file__).parent / "fixtures" / "squad_30.json"


def test_swap_experiment_runs_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("POINDEXTER_CACHE", str(tmp_path / "cache.sqlite"))
    candidates = datasets.integer_candidates("dev", path=FIXTURE)
    assert candidates
    section = asyncio.run(
        experiment.swap_experiment(
            FakeBackend(), tmp_path / "out", candidates=candidates, n_unscreened=10,
            grounded_n=10, redundant_n=3, sweep_n=3, k=1, concurrency=4,
        )
    )  # fmt: skip
    assert section["outcome"]["result"] in {"pass", "fail", "inconclusive"}
    assert section["screen"]["candidates"] == len(candidates)
    for name in ["screen.jsonl", "swapped_results.jsonl", "original_results.jsonl",
                 "metrics.json", "tables.md", "headline.json"]:  # fmt: skip
        assert (tmp_path / "out" / name).exists(), name
    assert json.loads((tmp_path / "out" / "headline.json").read_text())["classes"]


def test_outcome_rules():
    def rate(r, lo, hi):
        return {"k": 0, "n": 100, "rate": r, "lo": lo, "hi": hi}

    def section(recall, false_alarm, baseline, agree, classes=(120, 110)):
        return {
            "swap": {
                "decorative_recall": recall, "false_alarm": false_alarm,
                "span_in_cites": {"recall": baseline},
                "classes": {"grounded": classes[0], "decorative": classes[1], "excluded": 0},
            },
            "k_sweep": {"n": 100, "unavailable": 0, "changed": {"3": round(100 * (1 - agree))}},
        }  # fmt: skip

    good = section(rate(.9, .83, .95), rate(.05, .02, .1), rate(.01, 0, .05), .95)
    assert experiment.outcome(good)["result"] == "pass"
    missed = section(rate(.7, .6, .8), rate(.05, .02, .1), rate(.01, 0, .05), .95)
    assert experiment.outcome(missed)["result"] == "fail"
    wide = section(rate(.85, .6, .95), rate(.05, .02, .1), rate(.01, 0, .05), .95)
    assert experiment.outcome(wide)["result"] == "inconclusive"
    small = section(rate(.9, .83, .95), rate(.05, .02, .1), rate(.01, 0, .05), .95, (120, 40))
    assert experiment.outcome(small)["result"] == "inconclusive"
    noisy = section(rate(.9, .83, .95), rate(.05, .02, .1), rate(.01, 0, .05), .8)
    assert experiment.outcome(noisy)["result"] == "fail"


def test_outcome_counts_unavailable_sweep_records_as_disagreement():
    def rate(r, lo, hi):
        return {"k": 0, "n": 100, "rate": r, "lo": lo, "hi": hi}

    section = {
        "swap": {
            "decorative_recall": rate(.9, .83, .95), "false_alarm": rate(.05, .02, .1),
            "span_in_cites": {"recall": rate(.01, 0, .05)},
            "classes": {"grounded": 120, "decorative": 110, "excluded": 0},
        },
        "k_sweep": {"n": 80, "unavailable": 20, "changed": {"3": 0}},
    }  # fmt: skip
    out = experiment.outcome(section)
    assert out["k_agreement"] == 0.8
    assert out["result"] == "fail"
