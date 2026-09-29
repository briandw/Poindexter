"""Runs probe contexts through the cache and the backend, and builds Results."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from poindexter import cache
from poindexter import probes as probe_fns
from poindexter.backend import Backend
from poindexter.contract import (
    ABSTAIN,
    OTHER,
    PROBES,
    SAME,
    UNSTABLE,
    Answer,
    ProbeRun,
    Record,
    Rejection,
    Result,
    Unit,
    canonical,
    majority,
    read_records,
)
from poindexter.probes import counterfactual_plan
from poindexter.prompt import VALIDATORS, build_closed_book_prompt, build_prompt, retry_prompt
from poindexter.prompt import validate_closed_book as _validate_closed_book
from poindexter.verdict import finish, pick_answer, score, score_counterfactual

PROBE_SETS = ("all", "verdict", "verdict+C", "O")
# The probe sets that include probe C.
C_SETS = ("all", "verdict+C")
DEFAULT_CONCURRENCY = 16

Validator = Callable[[str], Answer | Rejection]


@dataclass
class _Tally:
    calls: int = 0
    retries: int = 0


async def call(
    backend: Backend,
    system: str,
    user: str,
    temperature: float | None,
    sample_index: int,
    semaphore: asyncio.Semaphore,
) -> str:
    """One sample, from the cache if present, else from the backend and then cached."""
    store = cache.open_default()
    k = cache.key(backend.model, temperature, system, user, sample_index)
    hit = store.get(k)
    if hit is not None:
        return hit
    async with semaphore:
        raw = await backend.complete(system, user, temperature, sample_index)
    return store.put(k, raw)


async def _sample_one(
    backend: Backend,
    system: str,
    user: str,
    check: Validator,
    temperature: float | None,
    i: int,
    semaphore: asyncio.Semaphore,
    tally: _Tally,
) -> tuple[str, Answer | None, str | None]:
    """Sample i with one retry on rejection. Returns (raw, answer or None, final code)."""
    tally.calls += 1
    raw = await call(backend, system, user, temperature, i, semaphore)
    got = check(raw)
    if isinstance(got, Answer):
        return raw, got, None
    tally.calls += 1
    tally.retries += 1
    raw = await call(backend, system, retry_prompt(user, got), temperature, i, semaphore)
    got = check(raw)
    if isinstance(got, Answer):
        return raw, got, None
    return raw, None, got.code


async def _sample(
    backend: Backend,
    system: str,
    user: str,
    check: Validator,
    k: int,
    temperature: float | None,
    semaphore: asyncio.Semaphore,
    tally: _Tally,
) -> tuple[list[str], list[Answer | None], list[str | None]]:
    rows = await asyncio.gather(
        *(_sample_one(backend, system, user, check, temperature, i, semaphore, tally)
          for i in range(k))
    )  # fmt: skip
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


async def _probe(
    units: list[Unit],
    question: str,
    a: Answer,
    backend: Backend,
    k: int,
    temperature: float | None,
    semaphore: asyncio.Semaphore,
    tally: _Tally,
    agent: str = "context",
    replacement: str | None = None,
) -> ProbeRun:
    """k samples of one context under `agent`'s prompt and validator, scored vs A (and,
    for probe C, vs the replacement)."""
    system, user = build_prompt(units, question, agent)
    ids, check = {u.id for u in units}, VALIDATORS[agent]
    raw, parsed, _ = await _sample(
        backend, system, user, lambda r: check(r, ids), k, temperature, semaphore, tally
    )
    if replacement is not None:
        return score_counterfactual(raw, parsed, a, replacement)
    return score(raw, parsed, a)


async def _n_probe(
    question: str,
    a: Answer,
    backend: Backend,
    k: int,
    temperature: float | None,
    semaphore: asyncio.Semaphore,
    tally: _Tally,
) -> ProbeRun:
    """Probe N: no units, under the closed-book prompt that permits answering from memory
    (cites must be []). Scored against A like every other probe."""
    system, user = build_closed_book_prompt(question)
    raw, parsed, _ = await _sample(
        backend, system, user, _validate_closed_book, k, temperature, semaphore, tally
    )
    return score(raw, parsed, a)


async def run_probe(
    context_units: list[Unit],
    question: str,
    A: Answer,
    backend: Backend,
    k: int,
    temperature: float | None,
    semaphore: asyncio.Semaphore,
) -> ProbeRun:
    """k samples of one context under the context agent, validated against that context's
    unit ids, scored vs A."""
    return await _probe(context_units, question, A, backend, k, temperature, semaphore, _Tally())


async def run_record(
    record: Record,
    backend: Backend,
    k: int,
    temperature: float | None,
    seed: int,
    donor_units: list[Unit] | None,
    semaphore: asyncio.Semaphore,
    probes: str = "all",
    agent: str = "context",
) -> Result:
    """Audit one record. `agent` picks the audited agent's prompt and validator
    (prompt.AGENTS, prompt.VALIDATORS) for every probe but N. probes="verdict" runs only
    O, N, R_remove, M; "verdict+C" adds probe C; "all" runs everything, C included.
    probes="O" runs only O (A and status as usual) and gives no verdict, flags, or
    alignment."""
    if probes not in PROBE_SETS:
        raise ValueError(f"probes must be one of {PROBE_SETS}, got {probes!r}")
    if agent not in VALIDATORS:
        raise ValueError(f"agent must be one of {sorted(VALIDATORS)}, got {agent!r}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    supplied = record.answer is not None
    if record.answer is not None and not record.answer.abstain and not record.answer.cites:
        if agent != "open":
            raise ValueError(
                f"record {record.id}: supplied answer cites nothing, which only the open "
                f"agent may do; got agent {agent!r}"
            )
    tally = _Tally()
    system, user = build_prompt(record.units, record.question, agent)
    ids, check = record.unit_ids, VALIDATORS[agent]
    raw, parsed, codes = await _sample(
        backend, system, user, lambda r: check(r, ids), k, temperature, semaphore, tally
    )
    base = Result(
        record=record, model=backend.model, k=k, temperature=temperature, status="ok",
        supplied=supplied,
    )  # fmt: skip
    if record.answer is not None:
        a: Answer | None = record.answer
    else:
        base.status, a = pick_answer(parsed)
    if a is None:
        o = ProbeRun(raw, parsed, [], UNSTABLE, sum(p is None for p in parsed))
        base.probes = {"O": o}
        code = None
        if base.status == "non_compliant":
            code = Counter(c for c in codes if c).most_common(1)[0][0]
        base.compliance = _compliance(tally, [o], code)
        return base

    base.A = a
    runs = {"O": score(raw, parsed, a)}
    c_run: ProbeRun | None = None
    c_context: list[Unit] | None = None
    if probes in C_SETS:
        base.counterfactual, c_context = counterfactual_plan(record, a, seed)
    if not a.abstain and probes != "O":
        units, cites = record.units, a.cites
        contexts = {
            "R_remove": probe_fns.remove_cited(units, cites, seed),
            "M": probe_fns.cited_only(units, cites, seed),
        }
        loo_contexts: dict[str, list[Unit]] = {}
        if probes == "all":
            contexts["R_replace"] = probe_fns.replace_cited(units, cites, seed, donor_units)
            contexts["S"] = probe_fns.shuffle(units, cites, seed)
            loo_contexts = {u.id: probe_fns.leave_one_out(units, i) for i, u in enumerate(units)}
        coros = [
            _n_probe(record.question, a, backend, k, temperature, semaphore, tally),
            *(_probe(ctx, record.question, a, backend, k, temperature, semaphore, tally, agent)
              for ctx in [*contexts.values(), *loo_contexts.values()]),
        ]  # fmt: skip
        if c_context is not None:
            replacement = base.counterfactual["replacement"]
            coros.append(_probe(
                c_context, record.question, a, backend, k, temperature, semaphore, tally,
                agent, replacement,
            ))  # fmt: skip
        n_run, *done = await asyncio.gather(*coros)
        if c_context is not None:
            c_run = done.pop()
            base.counterfactual = {**base.counterfactual, "run": c_run.to_json()}
        runs["N"] = n_run
        runs.update(zip(contexts, done[: len(contexts)], strict=True))
        base.loo = dict(zip(loo_contexts, done[len(contexts) :], strict=True))
    base.probes = {n: runs[n] for n in PROBES if n in runs}
    all_runs = [*base.probes.values(), *base.loo.values(), *([c_run] if c_run else [])]
    base.compliance = _compliance(tally, all_runs, None)
    return base if probes == "O" else finish(base)


def _compliance(tally: _Tally, runs: list[ProbeRun], code: str | None) -> dict:
    return {
        "calls": tally.calls,
        "retries": tally.retries,
        "final_rejections": sum(r.rejections for r in runs),
        "code": code,
    }


async def closed_book(
    record: Record,
    backend: Backend,
    k: int,
    semaphore: asyncio.Semaphore,
    temperature: float | None = None,
) -> ProbeRun:
    """k closed-book samples (question only, answering from memory allowed).

    Outcome per sample: `same` if the answer is `canonical`-equal to one of
    meta["dataset_answers"], `abstain`, else `other` (rejected twice counts as other).
    """
    if record.meta is None or "dataset_answers" not in record.meta:
        raise ValueError(f"record {record.id}: closed_book needs meta.dataset_answers")
    gold = {canonical(x) for x in record.meta["dataset_answers"]}
    system, user = build_closed_book_prompt(record.question)
    raw, parsed, _ = await _sample(
        backend, system, user, _validate_closed_book, k, temperature, semaphore, _Tally()
    )
    outcomes = [
        OTHER if p is None else ABSTAIN if p.abstain else
        SAME if canonical(p.text or "") in gold else OTHER
        for p in parsed
    ]  # fmt: skip
    return ProbeRun(raw, parsed, outcomes, majority(outcomes), sum(p is None for p in parsed))


async def run_records(
    records: list[Record],
    backend: Backend,
    k: int,
    temperature: float | None,
    seed: int,
    concurrency: int = DEFAULT_CONCURRENCY,
    out_path: str | Path | None = None,
    probes: str = "all",
    agent: str = "context",
) -> list[Result]:
    """Run records, at most `concurrency` records and `concurrency` model calls in
    flight. Results are appended to out_path (JSONL) as they finish, and returned in
    finishing order. R_replace's donor is the next record's units, wrapping."""
    if probes == "all" and len(records) < 2:
        raise ValueError("R_replace takes donor units from another record; need 2+ records")
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Result] = []
    out = open(out_path, "w") if out_path is not None else None
    pending: set[asyncio.Task[Result]] = set()

    def drain(done: set[asyncio.Task[Result]]) -> None:
        for task in done:
            result = task.result()
            results.append(result)
            if out is not None:
                out.write(json.dumps(result.to_json()) + "\n")
                out.flush()

    try:
        for i, record in enumerate(records):
            if len(pending) >= concurrency:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                drain(done)
            donor = records[(i + 1) % len(records)].units
            coro = run_record(
                record, backend, k, temperature, seed, donor, semaphore, probes, agent
            )
            pending.add(asyncio.create_task(coro))
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            drain(done)
    finally:
        for task in pending:
            task.cancel()
        if out is not None:
            out.close()
    return results


async def run_file(
    path: str | Path,
    backend: Backend,
    k: int,
    temperature: float | None,
    seed: int,
    concurrency: int = DEFAULT_CONCURRENCY,
    out_path: str | Path | None = None,
    probes: str = "all",
    agent: str = "context",
) -> list[Result]:
    return await run_records(
        read_records(path), backend, k, temperature, seed, concurrency, out_path, probes, agent
    )
