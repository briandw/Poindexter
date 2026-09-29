"""Verdicts, flags, and alignment from probe outcomes (the tables in PLAN.md).

`recompute(result, k)` re-derives all of it from the first k stored samples, with no
model calls. It is how `report --k` and the k sweep work.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from poindexter.contract import (
    OTHER,
    SAME,
    UNSTABLE,
    Answer,
    ProbeRun,
    Result,
    Unit,
    canonical,
    majority,
    outcome_of,
    span_in_cites,
    validate,
)

DECORATIVE = "decorative"
GROUNDED = "grounded"
INCOMPLETE = "incomplete"
K_UNAVAILABLE = "k_unavailable"
VERDICTS = (DECORATIVE, GROUNDED, INCOMPLETE, UNSTABLE)
FLAGS = (
    "parametric",
    "fabricates",
    "fabricates_on_replace",
    "position_sensitive",
    "span_in_cites",
    "reproduced",
)


# Majority keys for O samples that have no answer text. canonical() output has no
# angle brackets, so these cannot collide with an answer.
_REJECTED = "<rejected>"
_ABSTAINED = "<abstain>"


def _answer_key(p: Answer | None) -> str:
    return _REJECTED if p is None else _ABSTAINED if p.abstain else canonical(p.text or "")


def pick_answer(parsed: list[Answer | None]) -> tuple[str, Answer | None]:
    """A from the O samples: strict majority over `canonical` answer text, then the first
    sample that matches it. Returns (status, A); A is None unless status is `ok`."""
    keys = [_answer_key(p) for p in parsed]
    top, count = Counter(keys).most_common(1)[0]
    if count * 2 <= len(keys):
        return "unstable_original", None
    if top == _REJECTED:
        return "non_compliant", None
    return "ok", parsed[keys.index(top)]


def score(raw: list[str], parsed: list[Answer | None], a: Answer) -> ProbeRun:
    """A ProbeRun from k samples: outcomes relative to A, strict majority, rejections."""
    outcomes = [outcome_of(p, a) for p in parsed]
    rejections = sum(p is None for p in parsed)
    return ProbeRun(list(raw), list(parsed), outcomes, majority(outcomes), rejections)


def verdict(majorities: dict[str, str]) -> str:
    """From the R_remove and M majorities. Unstable in either wins over every other row."""
    r, m = majorities["R_remove"], majorities["M"]
    if r == UNSTABLE or m == UNSTABLE:
        return UNSTABLE
    if r == SAME:
        return DECORATIVE
    if m == SAME:
        return GROUNDED
    return INCOMPLETE


def flags(
    probes: dict[str, ProbeRun], answer: Answer, units: list[Unit], supplied: bool
) -> dict[str, bool | None]:
    """Flags independent of the verdict. None when the probe it needs was not run."""
    replace_run, s_run = probes.get("R_replace"), probes.get("S")
    return {
        "parametric": probes["N"].majority == SAME,
        "fabricates": probes["R_remove"].majority == OTHER,
        "fabricates_on_replace": None if replace_run is None else replace_run.majority == OTHER,
        "position_sensitive": None if s_run is None else _position_sensitive(probes["O"], s_run),
        "span_in_cites": span_in_cites(answer, units),
        "reproduced": probes["O"].majority == SAME if supplied else None,
    }


def cite_majority(run: ProbeRun) -> frozenset[str] | str:
    """Strict majority over the samples' cite sets (a rejected sample is its own value),
    else `unstable`."""
    sets = [frozenset(p.cites) if p is not None else None for p in run.parsed]
    top, count = Counter(sets).most_common(1)[0]
    return top if count * 2 > len(sets) and top is not None else UNSTABLE


def answer_majority(run: ProbeRun) -> str:
    """Strict majority over the samples' answer keys (canonical text; abstentions and
    rejected samples are their own values), else "<unstable>", which no answer key
    can equal."""
    keys = [_answer_key(p) for p in run.parsed]
    top, count = Counter(keys).most_common(1)[0]
    return top if count * 2 > len(keys) else "<unstable>"


def _position_sensitive(o_run: ProbeRun, s_run: ProbeRun) -> bool:
    """S's majority answer or majority cite set differs from O's, by the same
    strict-majority rule. Compared with O, not A, so a supplied A that O and S both
    disagree with does not set the flag."""
    return (
        answer_majority(s_run) != answer_majority(o_run)
        or cite_majority(s_run) != cite_majority(o_run)
    )


def alignment(cites: list[str], loo_majorities: dict[str, str]) -> dict[str, Any]:
    """load_bearing = units whose removal changes the answer (L_i majority not same)."""
    load_bearing = [uid for uid, m in loo_majorities.items() if m != SAME]
    hit = len(set(cites) & set(load_bearing))
    return {
        "precision": hit / len(cites),
        "recall": hit / len(load_bearing) if load_bearing else None,
        "load_bearing": load_bearing,
    }


def finish(result: Result) -> Result:
    """Fill verdict, flags, and alignment from the probe runs of an `ok` result.

    An abstained A has no citations to judge, and an O-only run has nothing to judge
    with, so neither gets a verdict.
    """
    a = result.A
    if result.status != "ok" or a is None or a.abstain or "R_remove" not in result.probes:
        return result
    result.verdict = verdict({n: p.majority for n, p in result.probes.items()})
    result.flags = flags(result.probes, a, result.record.units, result.supplied)
    if result.loo:
        result.alignment = alignment(a.cites, {u: p.majority for u, p in result.loo.items()})
    return result


def recompute(result: Result, k: int) -> Result:
    """The result as a k-sample run would have produced it, from the first k stored
    samples of every probe, with no model calls.

    A is re-derived from the first k O samples (a supplied A stays). If it differs
    from the stored A in canonical text, cite set, or abstention, the k-sample run
    would have built different probe contexts, so the result is `k_unavailable` with
    only O and no verdict. `compliance` keeps what the stored samples determine:
    `final_rejections` and `code` are exact for the first k samples; `calls` and
    `retries` are None because retries of samples that passed are not recorded.
    """
    if not 1 <= k <= result.k:
        raise ValueError(f"record {result.record.id}: k={k} outside 1..{result.k}")
    o = result.probes["O"]
    raw, parsed = o.raw[:k], o.parsed[:k]
    if result.supplied:
        status, a = "ok", result.A
    else:
        status, a = pick_answer(parsed)
    out = replace(
        result, k=k, status=status, A=a, probes={}, loo={},
        verdict=None, flags=None, alignment=None,
    )  # fmt: skip
    if a is None:  # non_compliant or unstable_original at k: only O ran
        out.probes = {"O": ProbeRun(raw, parsed, [], UNSTABLE, sum(p is None for p in parsed))}
        code = _rejection_code(result, raw, parsed) if status == "non_compliant" else None
        out.compliance = _k_compliance(out, code)
        return out
    if not _same_answer(a, result.A):
        out.status = K_UNAVAILABLE
        out.probes = {"O": score(raw, parsed, a)}
        out.compliance = _k_compliance(out, None)
        return out
    out.probes = {n: score(p.raw[:k], p.parsed[:k], a) for n, p in result.probes.items()}
    out.loo = {u: score(p.raw[:k], p.parsed[:k], a) for u, p in result.loo.items()}
    out.compliance = _k_compliance(out, None)
    return finish(out)


def _same_answer(a: Answer, b: Answer | None) -> bool:
    """Same probe contexts and same outcomes: abstention, canonical text, cite set."""
    return (
        b is not None
        and a.abstain == b.abstain
        and canonical(a.text or "") == canonical(b.text or "")
        and set(a.cites) == set(b.cites)
    )


def _rejection_code(result: Result, raw: list[str], parsed: list[Answer | None]) -> str:
    """Most common final rejection code, re-derived by validating the stored raw text."""
    codes = []
    for r, p in zip(raw, parsed, strict=True):
        if p is None:
            got = validate(r, result.record.unit_ids)
            assert not isinstance(got, Answer), f"record {result.record.id}: {r!r} validates"
            codes.append(got.code)
    return Counter(codes).most_common(1)[0][0]


def _k_compliance(out: Result, code: str | None) -> dict[str, Any]:
    runs = [*out.probes.values(), *out.loo.values()]
    return {
        "calls": None,
        "retries": None,
        "final_rejections": sum(r.rejections for r in runs),
        "code": code,
    }
