"""The headline experiment: swap-validated verdicts (PLAN.md, Goal; THRESHOLDS.md).

Stages, each written under the output directory; every model call is cached, so a rerun
with larger sizes only pays for the new calls:

1. candidates: SQuAD integer-answer records from the given splits, shuffled under the seed.
2. screen: one closed-book call per candidate. Memorized facts go to the pool, plus a
   fixed number of unmemorized ones.
3. swapped: probe O on each pool record's swap assigns the ground-truth class.
4. originals: verdict probes on the unswapped record, for every decorative record and a
   stable sample of grounded ones. These are what get scored.
5. redundant: verdict probes on gold-sentence-duplicated copies of grounded records.
6. sweep: a slice of the originals again at k=5 for the k sweep.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from poindexter import bench, datasets, runner, verdict
from poindexter.backend import Backend
from poindexter.contract import SAME, ProbeRun, Record, Result, write_jsonl

SWEEP_K = 5

# THRESHOLDS.md, as committed before any evaluation run.
MIN_CLASS = 100
RECALL_POINT, RECALL_LOWER = 0.80, 0.65
FALSE_ALARM_POINT, FALSE_ALARM_UPPER = 0.10, 0.20
MIN_K_AGREEMENT = 0.85


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _stable_order(ids: list[str], seed: int) -> list[str]:
    """Order that doesn't reshuffle when more ids join, so growing a run reuses the cache."""
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


def _write_results(path: Path, results: list[Result]) -> None:
    write_jsonl(path, (r.to_json() for r in sorted(results, key=lambda r: r.record.id)))


async def _screen(
    records: list[Record], backend: Backend, k: int, concurrency: int
) -> list[ProbeRun]:
    semaphore = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(runner.closed_book(r, backend, k, semaphore) for r in records))


async def swap_experiment(
    backend: Backend,
    out_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("dev", "train"),
    n_candidates: int = 1000,
    n_unscreened: int = 200,
    grounded_n: int = 150,
    redundant_n: int = 60,
    sweep_n: int = 50,
    k: int = 3,
    screen_k: int = 1,
    seed: int = 0,
    concurrency: int = runner.DEFAULT_CONCURRENCY,
    candidates: list[Record] | None = None,
) -> dict[str, Any]:
    """`candidates` overrides loading the splits (tests use fixture records)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if candidates is None:
        candidates = [r for split in splits for r in datasets.integer_candidates(split)]
    candidates = list(candidates)
    random.Random(seed).shuffle(candidates)
    candidates = candidates[:n_candidates]
    write_jsonl(out / "candidates.jsonl", (r.to_json() for r in candidates))
    _log(f"[{backend.model}] screening {len(candidates)} candidates")

    screens = await _screen(candidates, backend, screen_k, concurrency)
    pairs = list(zip(candidates, screens, strict=True))
    write_jsonl(out / "screen.jsonl", ({"id": r.id, **s.to_json()} for r, s in pairs))
    memorized = [r for r, s in pairs if s.majority == SAME]
    unmemorized = [r for r, s in pairs if s.majority != SAME]
    pool = memorized + unmemorized[:n_unscreened]
    _log(f"[{backend.model}] memorized {len(memorized)}; swapping pool of {len(pool)}")

    swapped = await runner.run_records(
        [datasets.swap_record(r, seed) for r in pool], backend, k, None, seed, concurrency,
        probes="O",
    )  # fmt: skip
    _write_results(out / "swapped_results.jsonl", swapped)

    by_class: dict[str, list[str]] = {"grounded": [], "decorative": []}
    for s in swapped:
        cls, _ = bench.swap_class(s)
        if cls in by_class:
            by_class[cls].append(s.record.meta["counterfactual"]["source_id"])
    grounded_ids = _stable_order(by_class["grounded"], seed)
    chosen = set(by_class["decorative"]) | set(grounded_ids[:grounded_n])
    originals = [r for r in pool if r.id in chosen]
    _log(
        f"[{backend.model}] swap classes {({c: len(v) for c, v in by_class.items()})}; "
        f"running verdict probes on {len(originals)} originals"
    )
    original_results = await runner.run_records(
        originals, backend, k, None, seed, concurrency, probes="verdict"
    )
    _write_results(out / "original_results.jsonl", original_results)

    # The plan's original variant: Poindexter on the swapped record itself. Same O prompt,
    # so O comes from the cache and the class assignment is unchanged.
    swapped_verdict = await runner.run_records(
        [datasets.swap_record(r, seed) for r in originals], backend, k, None, seed,
        concurrency, probes="verdict",
    )  # fmt: skip
    by_id = {s.record.id: s for s in swapped_verdict}
    swapped = [by_id.get(s.record.id, s) for s in swapped]
    _write_results(out / "swapped_results.jsonl", swapped)

    redundant_sources = {i for i in grounded_ids[:redundant_n]}
    redundant_records = [
        datasets.redundant_record(r, seed)
        for r in pool
        if r.id in redundant_sources and len(r.units) > 1
    ]
    _log(f"[{backend.model}] redundant-evidence records: {len(redundant_records)}")
    redundant_results = await runner.run_records(
        redundant_records, backend, k, None, seed, concurrency, probes="verdict"
    )
    _write_results(out / "redundant_results.jsonl", redundant_results)

    sweep_ids = set(_stable_order([r.id for r in originals], seed)[:sweep_n])
    sweep_records = [r for r in originals if r.id in sweep_ids]
    _log(f"[{backend.model}] k sweep on {len(sweep_records)} records at k={SWEEP_K}")
    sweep_results = await runner.run_records(
        sweep_records, backend, SWEEP_K, None, seed, concurrency, probes="verdict"
    )
    _write_results(out / "sweep_results.jsonl", sweep_results)

    section = bench.evaluate(original_results, swapped=swapped)["squad"]
    section["redundant"] = (
        bench.evaluate(redundant_results)["squad"].get("redundant") if redundant_results else None
    )
    section["k_sweep"] = bench.k_sweep(sweep_results, verdict.recompute, (1, 3, SWEEP_K))
    section["screen"] = {
        "candidates": len(candidates),
        "memorized": len(memorized),
        "pool": len(pool),
        "swap_classes": {c: len(v) for c, v in by_class.items()},
        "originals_run": len(originals),
    }
    section["model"] = backend.model
    section["outcome"] = outcome(section)
    bench.write_metrics({"squad": section}, out)
    (out / "headline.json").write_text(json.dumps(section["outcome"], indent=2) + "\n")
    _log(f"[{backend.model}] outcome: {section['outcome']['result']}")
    return section


def outcome(section: dict[str, Any]) -> dict[str, Any]:
    """Score one model's headline section against THRESHOLDS.md."""
    swap = section["swap"]
    recall, false_alarm = swap["decorative_recall"], swap["false_alarm"]
    baseline = swap["span_in_cites"]["recall"]
    # Records whose k=3 answer differs from the k=5 one count as disagreement, so the bar
    # applies to the whole sweep slice (THRESHOLDS.md), not only the comparable records.
    sweep = section["k_sweep"]
    compared = sweep["n"] + sweep["unavailable"]
    agreement_rate = (sweep["n"] - sweep["changed"]["3"]) / compared if compared else None
    classes = swap["classes"]

    def at_least(value: float | None, bar: float) -> bool | None:
        return None if value is None else value >= bar

    def at_most(value: float | None, bar: float) -> bool | None:
        return None if value is None else value <= bar

    rows = {
        "recall_point": at_least(recall["rate"], RECALL_POINT),
        "recall_lower": at_least(recall["lo"], RECALL_LOWER),
        "false_alarm_point": at_most(false_alarm["rate"], FALSE_ALARM_POINT),
        "false_alarm_upper": at_most(false_alarm["hi"], FALSE_ALARM_UPPER),
        "beats_baseline": (
            None
            if recall["lo"] is None or baseline["hi"] is None
            else recall["lo"] > baseline["hi"]
        ),
        "class_size": min(classes["grounded"], classes["decorative"]) >= MIN_CLASS,
        "k_agreement": at_least(agreement_rate, MIN_K_AGREEMENT),
    }
    points = [rows["recall_point"], rows["false_alarm_point"]]
    if any(p is False for p in points) or rows["k_agreement"] is False:
        result = "fail"
    elif all(v is True for v in rows.values()):
        result = "pass"
    else:
        result = "inconclusive"
    return {
        "result": result,
        "checks": rows,
        "decorative_recall": recall,
        "false_alarm": false_alarm,
        "baseline_recall": baseline,
        "classes": classes,
        "k_agreement": agreement_rate,
    }
