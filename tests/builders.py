"""Hand-built Results for bench and chart tests."""

from __future__ import annotations

import dataclasses
import random
from typing import Any

from poindexter.contract import (
    Answer,
    ProbeRun,
    Record,
    Result,
    Unit,
    majority,
    outcome_of,
)

UNITS = [Unit(f"u{i}", f"Sentence {i}.") for i in range(1, 6)]
NO_FLAGS = {
    "parametric": False,
    "fabricates": False,
    "fabricates_on_replace": False,
    "position_sensitive": False,
    "span_in_cites": True,
    "reproduced": None,
}


def ans(text: str | None, *cites: str) -> Answer:
    if text is None:
        return Answer(None, [], True)
    return Answer(text, list(cites or ("u1",)), False)


def run(samples: list[Answer | None], a: Answer | None = None) -> ProbeRun:
    outcomes = [outcome_of(s, a) for s in samples] if a is not None else ["other"] * len(samples)
    return ProbeRun(
        raw=["{}"] * len(samples), parsed=samples, outcomes=outcomes, majority=majority(outcomes)
    )


def outcomes_run(outcomes: list[str]) -> ProbeRun:
    return ProbeRun(raw=[""] * len(outcomes), parsed=[None] * len(outcomes),
                    outcomes=outcomes, majority=majority(outcomes))


def result(
    rid: str,
    *,
    dataset: str = "squad",
    status: str = "ok",
    A: Answer | None = None,
    verdict: str | None = "grounded",
    gold: list[str] | None = None,
    answers: list[str] | None = None,
    flags: dict[str, Any] | None = None,
    alignment: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    probes: dict[str, ProbeRun] | None = None,
    model: str = "haiku",
    k: int = 3,
    compliance: dict[str, Any] | None = None,
) -> Result:
    if status != "ok":
        verdict = None
    m = {"dataset": dataset, "dataset_answers": answers if answers is not None else ["90"]}
    m.update(meta or {})
    return Result(
        record=Record(id=rid, question="q?", units=UNITS,
                      gold=gold if gold is not None else ["u1"], meta=m),
        model=model,
        k=k,
        temperature=None,
        status=status,
        A=(A if A is not None else ans("90", "u1")) if status == "ok" else None,
        probes=probes or {},
        verdict=verdict,
        flags={**NO_FLAGS, **(flags or {})} if status == "ok" else None,
        alignment=alignment,
        compliance=compliance or {"calls": 10, "retries": 0, "final_rejections": 0, "code": None},
    )


# --- toy recompute for the k sweep: the PLAN.md verdict table on R_remove and M ---------


def toy_verdict(r_remove: str, m: str) -> str:
    if "unstable" in (r_remove, m):
        return "unstable"
    if r_remove == "same":
        return "decorative"
    return "grounded" if m == "same" else "incomplete"


def toy_recompute(r: Result, k: int) -> Result:
    rr = majority(r.probes["R_remove"].outcomes[:k])
    mm = majority(r.probes["M"].outcomes[:k])
    return dataclasses.replace(r, verdict=toy_verdict(rr, mm))


# --- a larger synthetic set for charts ---------------------------------------------------


def synthetic_run(model: str, dataset: str, seed: int, n: int = 60) -> list[Result]:
    """Plausible-looking results with k=5 R_remove/M samples; verdicts from toy_recompute."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        unans = dataset == "squad" and i % 5 == 0
        p_same = rng.choice([0.1, 0.5, 0.85])
        rr = ["same" if rng.random() < p_same else rng.choice(["other", "abstain"])
              for _ in range(5)]
        mm = ["same" if rng.random() < 0.8 else "other" for _ in range(5)]
        probes = {"R_remove": outcomes_run(rr), "M": outcomes_run(mm)}
        abstain = unans and rng.random() < 0.6
        a = ans(None) if abstain else ans("90" if rng.random() < 0.8 else "7", "u1")
        status = "non_compliant" if rng.random() < 0.03 else "ok"
        r = result(
            f"{dataset}-{i}", dataset=dataset, model=model, k=5, status=status, A=a,
            gold=[] if unans else ["u1"], answers=[] if unans else ["90"],
            meta={"unanswerable": True} if unans else None, probes=probes,
            flags={"parametric": rng.random() < 0.3, "fabricates": rr[0] == "other",
                   "span_in_cites": rng.random() < 0.9},
            alignment={"precision": 1.0, "recall": 1.0,
                       "load_bearing": ["u1"] if rng.random() < 0.8 else []},
        )
        if status == "ok":
            r = toy_recompute(r, 5)
        out.append(r)
    return out


def swap_pair(
    rid: str,
    verdict: str,
    swapped_samples: list[Answer | None] | None,
    *,
    orig_cites: tuple[str, ...] = ("u2",),
    orig_text: str | None = "1850",
    orig_status: str = "ok",
    span: bool | None = True,
    swapped_verdict: str | None = None,
    source_id: bool = True,
    model: str = "haiku",
) -> tuple[Result, Result | None]:
    """Original result (answer 1850 citing u2, gold u2) and its swapped run (O only).

    `swapped_samples=None` means no swapped run; `[]` means a swapped run without O.
    """
    original = result(rid, status=orig_status, A=ans(orig_text, *orig_cites), verdict=verdict,
                      gold=["u2"], answers=["1850"], flags={"span_in_cites": span}, model=model)
    if swapped_samples is None:
        return original, None
    cf = {"original": "1850", "swapped": "1862"}
    if source_id:
        cf["source_id"] = rid
    probes = {"O": run(swapped_samples)} if swapped_samples else {}
    swapped = result(f"{rid}~swap", verdict=swapped_verdict, gold=["u2"], answers=["1862"],
                     meta={"counterfactual": cf}, probes=probes, model=model)
    return original, swapped
