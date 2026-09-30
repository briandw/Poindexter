"""v3: sweep the size of probe C's edit (PLAN-v3.md).

Each agreeing (L0) passage from the v2 corpus is edited at graded sizes. Only the
answer sentence changes, exactly as in L1/L2:

- years: S1 +/-1-2, S2 +/-3-5, S3 +/-10-25, S4 +/-50-100, S5 +/-200-500 (within 1000-2025)
- counts: S1 5-10%, S2 15-25%, S3 40-60%, S4 x3-5, S5 x20-50
- entities: E1 the mild alternative, E2 a same-type pool alternative, E3 the strong one

For every model x agent x size the question is how often the answer follows the edit.
Probe C is this measurement at one size, so the sweep shows how a C verdict depends on
its edit size.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from poindexter import bench, runner, surprise
from poindexter.backend import Backend
from poindexter.contract import SAME, Record, Result, Unit, canonical, read_records, write_jsonl
from poindexter.corpus import YEAR_MAX, YEAR_MIN, _swap
from poindexter.datasets import number_mentions

YEAR_SIZES = {"S1": (1, 2), "S2": (3, 5), "S3": (10, 25), "S4": (50, 100), "S5": (200, 500)}
COUNT_SIZES = {
    "S1": ("share", 0.05, 0.10),
    "S2": ("share", 0.15, 0.25),
    "S3": ("share", 0.40, 0.60),
    "S4": ("factor", 3, 5),
    "S5": ("factor", 20, 50),
}
ENTITY_SIZES = ("E1", "E2", "E3")
AGENTS = ("context", "open", "document")
DRAWS = 40


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _rng(*parts: object) -> random.Random:
    return random.Random(":".join(map(str, parts)))


def _mentioned(value: int, record: Record) -> bool:
    texts = [record.question, *(u.text for u in record.units)]
    return any(v == value for t in texts for _, _, v in number_mentions(t))


def year_value(prior: int, size: str, record: Record, seed: int) -> int | None:
    lo, hi = YEAR_SIZES[size]
    rng = _rng(seed, record.id, size)
    for _ in range(DRAWS):
        d = rng.randint(lo, hi)
        signs = [s for s in (-1, 1) if YEAR_MIN <= prior + s * d <= YEAR_MAX]
        if not signs:
            continue
        v = prior + rng.choice(signs) * d
        if v != prior and not _mentioned(v, record):
            return v
    return None


def count_value(prior: int, size: str, record: Record, seed: int) -> int | None:
    mode, lo, hi = COUNT_SIZES[size]
    rng = _rng(seed, record.id, size)
    for _ in range(DRAWS):
        if mode == "share":
            d = max(1, round(prior * rng.uniform(lo, hi)))
            v = prior + rng.choice([-1, 1]) * d
            # Rounding can leave the window (46 * 5% rounds to 2, which is 4.3%).
            if not lo <= abs(v - prior) / prior <= hi:
                continue
        else:
            v = prior * rng.randint(lo, hi)
        if v >= 1 and v != prior and not _mentioned(v, record):
            return v
    return None


def _tokens(text: str) -> str:
    """Lowercase words with punctuation and hyphens as spaces and articles dropped, padded
    so a whole-word match is a substring match."""
    words = re.sub(r"[^\w]+", " ", text.lower()).split()
    return " " + " ".join(w for w in words if w not in ("a", "an", "the")) + " "


def entity_value(fact: dict[str, Any], size: str, record: Record, seed: int) -> str | None:
    """E1 mild, E2 a pool alternative (not mild or strong), E3 strong; none mentioned in the
    question or any unit (whole words, punctuation-insensitive: Mary-Jane = Mary Jane)."""
    text = _tokens(" ".join([record.question, *(u.text for u in record.units)]))
    if size == "E1":
        options = [fact["mild"]]
    elif size == "E3":
        options = [fact["strong"]]
    else:
        held = {canonical(fact["mild"]), canonical(fact["strong"]), canonical(fact["answer"])}
        options = [a for a in fact["alternatives"] if canonical(a) not in held]
        _rng(seed, record.id, size).shuffle(options)
    for v in options:
        if v and _tokens(v).strip() and _tokens(v) not in text:
            return v
    return None


def sweep_records(l0: Record, fact: dict[str, Any], seed: int) -> list[Record]:
    """The sized edits of one agreeing passage; sizes with no valid value are skipped."""
    gold = l0.gold[0]
    sentence = next(u.text for u in l0.units if u.id == gold)
    prior = fact["answer"]
    out = []
    sizes = ENTITY_SIZES if fact["kind"] == "entity" else tuple(YEAR_SIZES)
    for size in sizes:
        if fact["kind"] == "entity":
            value = entity_value(fact, size, l0, seed)
        elif fact["entity_type"] == "year":
            v = year_value(int(prior), size, l0, seed)
            value = None if v is None else str(v)
        else:
            v = count_value(int(prior.replace(",", "")), size, l0, seed)
            value = None if v is None else str(v)
        if value is None:
            continue
        edited = _swap(sentence, prior, value, fact["entity_type"])
        units = [Unit(u.id, edited) if u.id == gold else u for u in l0.units]
        meta = {
            **{k: v for k, v in l0.meta.items() if k not in ("heldout_values",)},
            "level": "sweep",
            "size": size,
            "dataset_answers": [value],
            "counterfactual": {"original": prior, "swapped": value, "source_id": l0.id},
        }
        if fact["kind"] == "number":
            p, n = int(prior.replace(",", "")), int(value)
            meta["delta"] = {"abs": abs(n - p), "rel": abs(n - p) / p}
        out.append(Record(f"{l0.id.removesuffix('~L0')}~{size}", l0.question, units,
                          gold=[gold], meta=meta))  # fmt: skip
    return out


def build_sweep(corpus_dir: str | Path, out_dir: str | Path, seed: int = 0) -> dict[str, Any]:
    src, out = Path(corpus_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    facts = {
        json.loads(line)["fact_id"]: json.loads(line)
        for line in (src / "facts.jsonl").read_text().splitlines() if line.strip()
    }  # fmt: skip
    records, missing = [], Counter()
    for l0 in read_records(src / "L0.jsonl"):
        fact = facts[l0.meta["fact_id"]]
        recs = sweep_records(l0, fact, seed)
        sizes = ENTITY_SIZES if fact["kind"] == "entity" else tuple(YEAR_SIZES)
        missing.update(f"{fact['kind']}/{s}" for s in set(sizes) - {r.meta["size"] for r in recs})
        records.extend(recs)
    write_jsonl(out / "sweep.jsonl", (r.to_json() for r in records))
    report = {
        "records": len(records),
        "by_size": dict(sorted(Counter(f"{r.meta['kind']}/{r.meta['size']}"
                                       for r in records).items())),  # fmt: skip
        "missing": dict(sorted(missing.items())),
    }
    (out / "sweep_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _stable(ids: list[str], seed: int) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


def follows(result: Result) -> bool | None:
    """True if the majority answer is the edited value, False if the prior, else None."""
    cls = surprise.update_class(result)
    return {"follows": True, "prior": False}.get(cls)


def curve(sweeps: list[Result], eligible: set[str]) -> dict[str, Any]:
    """Follow rate per kind/size over facts whose L0 answer qualified."""
    by: dict[str, Counter[str]] = {}
    for r in sweeps:
        if r.record.meta["fact_id"] not in eligible:
            continue
        key = f"{r.record.meta['kind']}/{r.record.meta['size']}"
        by.setdefault(key, Counter())[surprise.update_class(r)] += 1
    out = {}
    for key, c in sorted(by.items()):
        decided = c["follows"] + c["prior"]
        out[key] = {
            "classes": dict(sorted(c.items())),
            "follow_rate": bench.rate(c["follows"], sum(c.values())),
            "follow_rate_decided": bench.rate(c["follows"], decided),
        }
    return out


def agreement(sweeps: list[Result], eligible: set[str]) -> dict[str, Any]:
    """For each pair of sizes, how often a fact's follow/keep call at one size matches the
    call at the other. A C verdict at size s is the call at s; the truth in v2 was the call
    at the mild size."""
    calls: dict[str, dict[str, bool]] = {}
    for r in sweeps:
        f = r.record.meta["fact_id"]
        v = follows(r)
        if f in eligible and v is not None:
            calls.setdefault(f"{r.record.meta['kind']}", {}).setdefault(f, {})
            calls[r.record.meta["kind"]][f][r.record.meta["size"]] = v
    out: dict[str, Any] = {}
    for kind, per_fact in calls.items():
        sizes = sorted({s for d in per_fact.values() for s in d})
        table: dict[str, Any] = {}
        for a in sizes:
            for b in sizes:
                if a >= b:
                    continue
                pairs = [(d[a], d[b]) for d in per_fact.values() if a in d and b in d]
                table[f"{a}~{b}"] = bench.rate(sum(x == y for x, y in pairs), len(pairs))
                followed = [y for x, y in pairs if x]
                table[f"{b}|{a}"] = bench.rate(sum(followed), len(followed))
        # H3 compares pairs on one cohort: facts with a decided call at every size.
        common = [d for d in per_fact.values() if all(s in d for s in sizes)]
        table["common"] = {
            "n": len(common),
            "excluded": len(per_fact) - len(common),
            **{
                f"{a}~{b}": bench.rate(sum(d[a] == d[b] for d in common), len(common))
                for a in sizes for b in sizes if a < b
            },
        }  # fmt: skip
        out[kind] = table
    return out


def tipping(sweeps: list[Result], eligible: set[str]) -> dict[str, Any]:
    """Per numeric fact, the smallest size at which the answer stops following the edit.
    A size that is missing or undecided before the first keep makes the fact `unknown`."""
    per: dict[str, dict[str, bool | None]] = {}
    for r in sweeps:
        f = r.record.meta["fact_id"]
        if f in eligible and r.record.meta["kind"] == "number":
            per.setdefault(f, {})[r.record.meta["size"]] = follows(r)
    counts: Counter[str] = Counter()
    for d in per.values():
        label = "never"
        for size in YEAR_SIZES:
            v = d.get(size)
            if v is None:
                label = "unknown"
                break
            if v is False:
                label = size
                break
        counts[label] += 1
    return dict(sorted(counts.items()))


async def sweep_experiment(
    backend: Backend,
    corpus_dir: str | Path,
    sweep_dir: str | Path,
    out_dir: str | Path,
    *,
    agents: tuple[str, ...] = AGENTS,
    n_facts: int = 150,
    k: int = 3,
    seed: int = 0,
    concurrency: int = runner.DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    l0s = {r.meta["fact_id"]: r for r in read_records(Path(corpus_dir) / "L0.jsonl")}
    sweeps_all = read_records(Path(sweep_dir) / "sweep.jsonl")
    semaphore = asyncio.Semaphore(concurrency)
    facts = sorted(l0s)
    screens = await asyncio.gather(
        *(runner.closed_book(l0s[f], backend, 3, semaphore) for f in facts)
    )
    confident = [f for f, s in zip(facts, screens, strict=True) if s.outcomes.count(SAME) == 3]
    chosen = set(_stable(confident, seed)[:n_facts])
    sweeps = [r for r in sweeps_all if r.meta["fact_id"] in chosen]
    _log(f"[{backend.model}] confident {len(confident)}; facts {len(chosen)}; "
         f"sweep records {len(sweeps)}")  # fmt: skip
    report: dict[str, Any] = {"model": backend.model, "confident": len(confident),
                              "facts": len(chosen), "agents": {}}  # fmt: skip
    for agent in agents:
        base = await runner.run_records(
            [l0s[f] for f in sorted(chosen)], backend, k, None, seed, concurrency,
            probes="O", agent=agent,
        )  # fmt: skip
        eligible = {
            r.record.meta["fact_id"] for r in base
            if surprise.scoreable(r) is None
        }  # fmt: skip
        runs = await runner.run_records(
            sweeps, backend, k, None, seed, concurrency, probes="O", agent=agent
        )
        write_jsonl(out / f"{agent}_L0.jsonl", (r.to_json() for r in base))
        write_jsonl(out / f"{agent}_sweep.jsonl", (r.to_json() for r in runs))
        section = {
            "eligible": len(eligible),
            "curve": curve(runs, eligible),
            "agreement": agreement(runs, eligible),
            "tipping": tipping(runs, eligible),
            "uncited": surprise.uncited_rates(runs),
        }
        report["agents"][agent] = section
        _log(f"[{backend.model}/{agent}] " + json.dumps(
            {k: v["follow_rate"]["rate"] for k, v in section["curve"].items()}))  # fmt: skip
    (out / "sweep.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
