import pytest
from builders import ans, swap_pair, synthetic_run, toy_recompute

from poindexter import charts
from poindexter.bench import evaluate, evaluate_swap

G = ans("1862", "u2")
D = ans("1850", "u2")


def fixture_metrics():
    by_run = {}
    for i, (model, dataset) in enumerate(
        [("haiku", "squad"), ("haiku", "hotpotqa"), ("sonnet", "squad")]
    ):
        m = evaluate(synthetic_run(model, dataset, seed=i), recompute=toy_recompute)
        by_run[f"{model}/{dataset}"] = m[dataset]
    pairs = [swap_pair(f"o{i}", "decorative" if i % 3 else "grounded", [G, G, G] if i % 2 else
                       [D, D, D], span=i != 3) for i in range(12)]
    by_run["haiku/squad"]["swap"] = evaluate_swap([o for o, _ in pairs], [s for _, s in pairs])
    return by_run


@pytest.mark.parametrize(
    "fn", [charts.verdicts, charts.checker, charts.parametric, charts.unanswerable,
           charts.k_sweep],
)
def test_chart_writes_png(fn, tmp_path):
    out = fn(fixture_metrics(), tmp_path / "c.png")
    assert out is not None and out.stat().st_size > 1000
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_headline_and_all(tmp_path):
    m = fixture_metrics()
    out = charts.headline({"haiku": m["haiku/squad"]["swap"]}, tmp_path / "h.png",
                          thresholds={"decorative_recall": {"min": 0.7}})
    assert out.stat().st_size > 1000
    written = charts.all_charts(m, tmp_path / "all")
    assert len(written) == 6 and all(p.stat().st_size > 1000 for p in written)


def test_absent_sections_are_skipped(tmp_path):
    m = fixture_metrics()
    hotpot = {"haiku/hotpotqa": {k: v for k, v in m["haiku/hotpotqa"].items()
                                 if k != "k_sweep"}}
    assert charts.unanswerable(hotpot, tmp_path / "u.png") is None
    assert charts.k_sweep(hotpot, tmp_path / "k.png") is None
    assert not (tmp_path / "u.png").exists()
    assert charts.checker(hotpot, tmp_path / "c.png") is not None  # known-grounded panel only
