import asyncio
import json
from pathlib import Path

from poindexter import datasets, experiment
from poindexter.backend import FakeBackend
from poindexter.contract import validate, validate_open
from poindexter.prompt import AGENTS, CLOSED_BOOK_SYSTEM, VALIDATORS

FIXTURE = Path(__file__).parent / "fixtures" / "squad_30.json"


def test_swap_experiment_runs_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("POINDEXTER_CACHE", str(tmp_path / "cache.sqlite"))
    candidates = datasets.integer_candidates("dev", path=FIXTURE)
    assert candidates
    backend = SystemRecorder()
    section = asyncio.run(
        experiment.swap_experiment(
            backend, tmp_path / "out", candidates=candidates, n_unscreened=10,
            grounded_n=10, redundant_n=3, sweep_n=3, k=1, concurrency=4,
        )
    )  # fmt: skip
    assert section["outcome"]["result"] in {"pass", "fail", "inconclusive"}
    assert section["screen"]["candidates"] == len(candidates)
    # The v1 headline replays from the committed cache only under v1's context prompt.
    assert set(backend.systems) - {CLOSED_BOOK_SYSTEM} == {AGENTS["context_v1"]}
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


def test_explore_open_agent_runs_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("POINDEXTER_CACHE", str(tmp_path / "cache.sqlite"))
    candidates = datasets.integer_candidates("dev", path=FIXTURE)
    asyncio.run(
        experiment.swap_experiment(
            FakeBackend(), tmp_path / "swap", candidates=candidates, n_unscreened=10,
            grounded_n=10, redundant_n=3, sweep_n=3, k=1, concurrency=4,
        )
    )  # fmt: skip
    screen = tmp_path / "swap" / "screen.jsonl"
    rows = [json.loads(line) for line in screen.read_text().splitlines()]
    screen.write_text("".join(json.dumps({**r, "majority": "same"}) + "\n" for r in rows))
    backend = SystemRecorder()
    report = asyncio.run(
        experiment.explore_open_agent(backend, tmp_path / "swap", tmp_path / "x", k=1)
    )
    assert report["memorized"] == len(candidates)
    assert set(report) >= {"swap", "wide", "verdicts_all"}
    # The v1 arm replays from the committed cache only under the v1 open prompt.
    assert set(backend.systems) - {CLOSED_BOOK_SYSTEM} == {AGENTS["open_v1"]}


class SystemRecorder(FakeBackend):
    def __init__(self):
        super().__init__()
        self.systems = []

    async def complete(self, system, user, temperature, sample_index):
        self.systems.append(system)
        return await super().complete(system, user, temperature, sample_index)


OPEN_V1 = (
    "You answer a question. The user provides context units retrieved for it; use them "
    "together with your own knowledge and give the answer you believe is correct. Each unit "
    "is one line that starts with its id in square brackets, like [u1].\n\nRespond with "
    'exactly one JSON object and nothing else:\n{"answer": "<string or null>", "cites": '
    '["<unit id>"], "abstain": false}\n\nRules:\n1. Use the context units and your own '
    "knowledge.\n2. Cite the id of every unit that supports your answer, and only ids that "
    'appear in the context.\n3. If you cannot answer, abstain: {"answer": null, "cites": [], '
    '"abstain": true}\n4. The answer is the shortest span that answers the question: a few '
    "words, not a sentence (a name, number, date, or short phrase), at most 200 characters."
    "\n5. Your response is the JSON object alone, with nothing before or after it and no "
    "explanation, including when you abstain."
)


CONTEXT_V1 = (
    "You answer a question using only the context units the user provides. Each unit is one "
    "line that starts with its id in square brackets, like [u1].\n\nRespond with exactly one "
    'JSON object and nothing else:\n{"answer": "<string or null>", "cites": ["<unit id>"], '
    '"abstain": false}\n\nRules:\n1. Answer only from the context units. Do not use outside '
    "knowledge.\n2. Cite the id of every unit you relied on, and only ids that appear in the "
    'context.\n3. If the context does not contain the answer, abstain: {"answer": null, '
    '"cites": [], "abstain": true}\n4. The answer is the shortest span that answers the '
    "question: a few words, not a sentence (a name, number, date, or short phrase), at most "
    "200 characters.\n5. Your response is the JSON object alone, with nothing before or "
    "after it and no explanation, including when you abstain."
)


def test_v1_prompts_are_byte_identical_and_strict():
    assert AGENTS["open_v1"] == OPEN_V1 and AGENTS["context_v1"] == CONTEXT_V1
    assert VALIDATORS["open_v1"] is validate and VALIDATORS["context_v1"] is validate
    assert VALIDATORS["open"] is validate_open and VALIDATORS["context"] is validate
    assert AGENTS["open"] != OPEN_V1 and AGENTS["context"] != CONTEXT_V1


def test_wide_swap_is_implausible_and_unique():
    for r in datasets.integer_candidates("dev", path=FIXTURE):
        w = experiment.wide_swap_record(r, 0)
        cf = w.meta["counterfactual"]
        assert cf["swapped"] != cf["original"]
        gold = next(u for u in w.units if u.id == r.gold[0])
        assert cf["swapped"] in gold.text
        assert [u.text for u in w.units if u.id != r.gold[0]] == [
            u.text for u in r.units if u.id != r.gold[0]
        ]


def test_wide_swap_rejects_ambiguous_mentions():
    import pytest

    from poindexter.contract import Record, Unit

    rec = Record(
        "r", "q", [Unit("u1", "Model A1234 shipped in 1234."), Unit("u2", "Other text.")],
        gold=["u1"], meta={"dataset_answers": ["1234"]},
    )  # fmt: skip
    w = experiment.wide_swap_record(rec, 0)
    assert w.units[0].text.startswith("Model A1234 shipped in ")
    assert "1234." not in w.units[0].text
    dup = Record(
        "d", "q", [Unit("u1", "It was 1234."), Unit("u2", "Also 1234.")],
        gold=["u1"], meta={"dataset_answers": ["1234"]},
    )  # fmt: skip
    with pytest.raises(ValueError):
        experiment.wide_swap_record(dup, 0)
