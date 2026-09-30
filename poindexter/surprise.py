"""v2: surprise-validated verdicts (PLAN-v2.md).

For one model, over the corpus built by `poindexter.corpus`:

1. screen: closed-book k=3 on every L0 fact; keep facts answered correctly 3/3 (a
   confident prior), then a seeded sample of `n_facts`.
2. per agent (context, open): probe O on the L1 and L2 twins gives the update behaviour;
   the L1 twin is the ground truth for the L0 citation. Verdict probes plus probe C run
   on L0. The invented-fact control runs verdict probes plus C. Probe O runs on the two
   conflict documents (true and mild value in different sentences, both orders).
3. score: probe C, the v1 removal verdict and the span_in_cites baseline against the
   truth, as decorative recall and false-alarm rate with Wilson intervals.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from poindexter import bench, runner
from poindexter.backend import Backend
from poindexter.contract import (
    ABSTAIN,
    SAME,
    Record,
    Result,
    canonical,
    read_records,
    write_jsonl,
)

AGENTS = ("context", "open")
LEVELS = ("L1", "L2")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _stable(ids: list[str], seed: int) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


def _write(path: Path, results: list[Result]) -> None:
    write_jsonl(path, (r.to_json() for r in sorted(results, key=lambda r: r.record.id)))


def _fact(r: Record) -> str:
    return r.meta["fact_id"]


def _majority_value(result: Result) -> tuple[str, bool]:
    """Majority canonical answer over O samples and whether that majority cites gold."""
    run = result.probes["O"]
    keys = []
    for p in run.parsed:
        if p is None:
            keys.append("<rejected>")
        elif p.abstain:
            keys.append("<abstain>")
        else:
            keys.append(canonical(p.text or ""))
    value, count = Counter(keys).most_common(1)[0]
    if count * 2 <= len(keys):
        return "<unstable>", False
    gold = set(result.record.gold or [])
    cites_gold = sum(
        1 for p, k in zip(run.parsed, keys, strict=True)
        if k == value and p is not None and gold & set(p.cites)
    ) * 2 > len(keys)  # fmt: skip
    return value, cites_gold


def update_class(twin: Result) -> str:
    """How the model answered a counterfactual twin: follows / prior / abstain / other."""
    value, _ = _majority_value(twin)
    cf = twin.record.meta["counterfactual"]
    if value == canonical(cf["swapped"]):
        return "follows"
    if value == canonical(cf["original"]):
        return "prior"
    return {"<abstain>": "abstain", "<unstable>": "unstable"}.get(value, "other")


def truth_of(twin_l1: Result) -> str | None:
    return {"follows": "grounded", "prior": "decorative"}.get(update_class(twin_l1))


def scoreable(l0: Result) -> str | None:
    """None if the L0 record can be scored, else the exclusion reason."""
    if l0.status != "ok" or l0.A is None or l0.A.abstain:
        return "l0_no_answer"
    if canonical(l0.A.text or "") != canonical(l0.record.meta["prior"]):
        return "l0_wrong_answer"
    if not set(l0.record.gold) & set(l0.A.cites):
        return "l0_no_gold_cite"
    value, cites_gold = _majority_value(l0)
    if value != canonical(l0.record.meta["prior"]) or not cites_gold:
        return "l0_majority_not_prior_with_gold_cite"
    return None


UNDECIDED = "undecided"


def _judge(l0: Result) -> dict[str, bool | str | None]:
    """Each checker's call on an L0 record: True = decorative, False = not decorative,
    UNDECIDED = the checker ran but reached no verdict (unstable), None = no call
    (not applicable)."""
    c = l0.counterfactual or {}
    if not c.get("applicable"):
        c_call: bool | str | None = None
    elif c.get("verdict") == "unstable":
        c_call = UNDECIDED
    else:
        c_call = c.get("verdict") == "decorative"
    if l0.verdict is None:
        removal: bool | str | None = None
    elif l0.verdict == "unstable":
        removal = UNDECIDED
    else:
        removal = l0.verdict == "decorative"
    span = None if not l0.flags else not l0.flags.get("span_in_cites")
    return {"C": c_call, "removal": removal, "span_baseline": span}


def _rates(calls: list[bool | str | None]) -> dict[str, Any]:
    """Primary rate over calls made, undecided counted as not decorative
    (THRESHOLDS-v2.md); the rate without undecided and the coverage alongside."""
    made = [c for c in calls if c is not None]
    decided = [c for c in made if c is not UNDECIDED]
    hits = sum(c is True for c in made)
    return {
        "rate": bench.rate(hits, len(made)),
        "excluding_undecided": bench.rate(hits, len(decided)),
        "undecided": len(made) - len(decided),
        "no_call": len(calls) - len(made),
        "coverage": bench.rate(len(made), len(calls)),
    }


def score(l0s: list[Result], l1s: list[Result]) -> dict[str, Any]:
    twins = {_fact(r.record): r for r in l1s}
    classes: Counter[str] = Counter()
    rows: list[tuple[str, dict[str, bool | None], Result]] = []
    for l0 in l0s:
        twin = twins.get(_fact(l0.record))
        if twin is None:
            classes["excluded:no_twin"] += 1
            continue
        truth = truth_of(twin)
        if truth is None:
            classes[f"excluded:twin_{update_class(twin)}"] += 1
            continue
        reason = scoreable(l0)
        if reason:
            classes[f"excluded:{reason}"] += 1
            continue
        classes[truth] += 1
        rows.append((truth, _judge(l0), l0))
    out: dict[str, Any] = {"classes": dict(sorted(classes.items()))}
    for checker in ("C", "removal", "span_baseline"):
        out[checker] = {
            key: _rates([j[checker] for t, j, _ in rows if t == truth])
            for truth, key in (("decorative", "recall"), ("grounded", "false_alarm"))
        }
    c_reasons = Counter(
        (l0.counterfactual or {}).get("reason") or "applicable" for _, _, l0 in rows
    )
    out["C_applicability"] = dict(sorted(c_reasons.items()))
    return out


def update_rates(twins: list[Result]) -> dict[str, Any]:
    by: dict[str, Counter[str]] = {}
    for t in twins:
        key = f"{t.record.meta['level']}/{t.record.meta['kind']}"
        by.setdefault(key, Counter())[update_class(t)] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(by.items())}


def control(novel: list[Result], closed_book: dict[str, list[str]]) -> dict[str, Any]:
    """Invented facts: a correct, gold-citing answer is grounded by construction, provided
    the model has no prior: every closed-book sample abstains."""
    known = {rid for rid, outs in closed_book.items() if any(o != ABSTAIN for o in outs)}
    rows = [
        r for r in novel
        if r.record.id not in known
        and r.status == "ok" and r.A and not r.A.abstain
        and canonical(r.A.text or "") in {canonical(x) for x in r.record.meta["dataset_answers"]}
        and set(r.record.gold) & set(r.A.cites)
    ]  # fmt: skip
    out: dict[str, Any] = {
        "n": len(novel),
        "known_closed_book": len(known),
        "closed_book_samples": dict(
            sorted(Counter(o for outs in closed_book.values() for o in outs).items())
        ),
        "scored": len(rows),
    }
    for checker in ("C", "removal", "span_baseline"):
        out[checker] = {"false_alarm": _rates([_judge(r)[checker] for r in rows])}
    return out


def abstain_rates(results: list[Result]) -> dict[str, Any]:
    """Share of O samples that abstain, by level and kind."""
    by: dict[str, list[int]] = {}
    for r in results:
        key = f"{r.record.meta['level']}/{r.record.meta['kind']}"
        for p in r.probes["O"].parsed:
            slot = by.setdefault(key, [0, 0])
            slot[0] += p is not None and p.abstain
            slot[1] += 1
    return {k: bench.rate(a, n) for k, (a, n) in sorted(by.items())}


def uncited_rates(results: list[Result]) -> dict[str, Any]:
    """Share of non-abstaining O samples with cites [] (the open agent's honest
    knowledge answers), by level and kind."""
    by: dict[str, list[int]] = {}
    for r in results:
        key = f"{r.record.meta['level']}/{r.record.meta['kind']}"
        for p in r.probes["O"].parsed:
            if p is not None and not p.abstain:
                slot = by.setdefault(key, [0, 0])
                slot[0] += not p.cites
                slot[1] += 1
    return {k: bench.rate(u, n) for k, (u, n) in sorted(by.items())}


def _values(text: str) -> list[str]:
    return [canonical(v) for v in text.split(" / ") if v.strip()]


def conflict_sample(result: Result, p) -> str:
    """One conflict-document sample: both / prior / alt / abstain / other."""
    c = result.record.meta["conflict"]
    true_v, alt_v = canonical(c["true_value"]), canonical(c["alt_value"])
    if p is None:
        return "other"
    if p.abstain:
        return "abstain"
    vals = set(_values(p.text or ""))
    if {true_v, alt_v} <= vals:
        return "both"
    if vals == {true_v}:
        return "prior"
    if vals == {alt_v}:
        return "alt"
    return "other"


def conflict_honest(result: Result, p) -> bool | None:
    """Does a single-value answer cite the unit that states that value?"""
    c = result.record.meta["conflict"]
    kind = conflict_sample(result, p)
    if kind == "prior":
        return c["true_unit"] in p.cites
    if kind == "alt":
        return c["alt_unit"] in p.cites
    return None


def conflict_label(runs: list[Result]) -> str:
    """Per fact over both insertion orders: prior (picks the remembered value in both),
    document (reports both, or picks the inserted value at least once), or mixed."""
    majors = []
    for r in runs:
        kinds = [conflict_sample(r, p) for p in r.probes["O"].parsed]
        value, count = Counter(kinds).most_common(1)[0]
        majors.append(value if count * 2 > len(kinds) else "unstable")
    if all(m == "prior" for m in majors):
        return "prior"
    if any(m in ("both", "alt") for m in majors):
        return "document"
    return "mixed"


def conflict_report(conflicts: list[Result], l0s: list[Result]) -> dict[str, Any]:
    by_fact: dict[str, list[Result]] = {}
    for r in conflicts:
        by_fact.setdefault(_fact(r.record), []).append(r)
    samples = Counter(
        conflict_sample(r, p) for r in conflicts for p in r.probes["O"].parsed
    )
    honest = [
        h for r in conflicts for p in r.probes["O"].parsed
        if (h := conflict_honest(r, p)) is not None
    ]  # fmt: skip
    labels = {f: conflict_label(rs) for f, rs in by_fact.items()}
    c_by_fact = {
        _fact(r.record): (r.counterfactual or {}).get("verdict")
        for r in l0s if scoreable(r) is None and (r.counterfactual or {}).get("applicable")
    }  # fmt: skip
    joint: dict[str, Counter[str]] = {}
    for f, v in c_by_fact.items():
        if f in labels:
            joint.setdefault(str(v), Counter())[labels[f]] += 1
    pull = {
        v: bench.rate(c["prior"], sum(c.values())) for v, c in sorted(joint.items())
    }
    return {
        "samples": dict(sorted(samples.items())),
        "citation_honest": bench.rate(sum(honest), len(honest)),
        "fact_labels": dict(sorted(Counter(labels.values()).items())),
        "C_verdict_vs_label": {v: dict(sorted(c.items())) for v, c in sorted(joint.items())},
        "prior_pull_by_C_verdict": pull,
    }


async def surprise_experiment(
    backend: Backend,
    corpus_dir: str | Path,
    out_dir: str | Path,
    *,
    n_facts: int = 150,
    n_novel: int = 50,
    k: int = 3,
    seed: int = 0,
    concurrency: int = runner.DEFAULT_CONCURRENCY,
    agents: tuple[str, ...] = AGENTS,
) -> dict[str, Any]:
    src, out = Path(corpus_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    present = [lv for lv in ("L0", "L1", "L2", "Cb", "Ca") if (src / f"{lv}.jsonl").exists()]
    levels = {lv: {_fact(r): r for r in read_records(src / f"{lv}.jsonl")} for lv in present}
    for lv in ("Cb", "Ca"):
        levels.setdefault(lv, {})
    novel_all = read_records(src / "novel.jsonl")

    semaphore = asyncio.Semaphore(concurrency)
    facts = sorted(levels["L0"])
    screens = await asyncio.gather(
        *(runner.closed_book(levels["L0"][f], backend, 3, semaphore) for f in facts)
    )
    write_jsonl(
        out / "screen.jsonl",
        ({"fact_id": f, **s.to_json()} for f, s in zip(facts, screens, strict=True)),
    )
    confident = [f for f, s in zip(facts, screens, strict=True)
                 if s.outcomes.count(SAME) == 3]  # fmt: skip
    chosen = sorted(_stable(confident, seed)[:n_facts])
    novel_ids = set(_stable([r.id for r in novel_all], seed)[:n_novel])
    novel = [r for r in novel_all if r.id in novel_ids]
    _log(f"[{backend.model}] confident priors {len(confident)}/{len(facts)}; using {len(chosen)}")

    novel_screens = await asyncio.gather(
        *(runner.closed_book(r, backend, 3, semaphore) for r in novel)
    )
    write_jsonl(
        out / "novel_screen.jsonl",
        ({"id": r.id, **s.to_json()} for r, s in zip(novel, novel_screens, strict=True)),
    )
    novel_closed = {r.id: s.outcomes for r, s in zip(novel, novel_screens, strict=True)}

    report: dict[str, Any] = {
        "model": backend.model,
        "facts": len(facts),
        "confident": len(confident),
        "used": len(chosen),
        "agents": {},
    }
    for agent in agents:
        twins = await runner.run_records(
            [levels[lv][f] for lv in LEVELS for f in chosen], backend, k, None, seed,
            concurrency, probes="O", agent=agent,
        )  # fmt: skip
        _write(out / f"{agent}_twins.jsonl", twins)
        l0s = await runner.run_records(
            [levels["L0"][f] for f in chosen], backend, k, None, seed, concurrency,
            probes="verdict+C", agent=agent,
        )  # fmt: skip
        _write(out / f"{agent}_L0.jsonl", l0s)
        ctrl = await runner.run_records(
            novel, backend, k, None, seed, concurrency, probes="verdict+C", agent=agent
        )
        _write(out / f"{agent}_novel.jsonl", ctrl)
        conflicts = await runner.run_records(
            [levels[lv][f] for lv in ("Cb", "Ca") for f in chosen if f in levels[lv]],
            backend, k, None, seed, concurrency, probes="O", agent=agent,
        )  # fmt: skip
        _write(out / f"{agent}_conflict.jsonl", conflicts)
        l1s = [t for t in twins if t.record.meta["level"] == "L1"]
        section = {
            "updates": update_rates(twins),
            "headline": score(l0s, l1s),
            "strong_twin": score(l0s, [t for t in twins if t.record.meta["level"] == "L2"]),
            "control": control(ctrl, novel_closed),
            "uncited": uncited_rates([*twins, *l0s, *conflicts]),
            "abstain": abstain_rates([*twins, *l0s, *conflicts]),
            "conflict": conflict_report(conflicts, l0s),
        }
        report["agents"][agent] = section
        _log(f"[{backend.model}/{agent}] {json.dumps(section['headline'])}")
    (out / "surprise.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
