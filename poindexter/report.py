"""Aggregate results into counts and rates, and print them as markdown tables."""

from __future__ import annotations

from collections import Counter
from typing import Any

from poindexter.contract import PROBES, Result
from poindexter.verdict import FLAGS, VERDICTS

STATUSES = ("ok", "non_compliant", "unstable_original", "k_unavailable")


def _rate(num: float, den: float) -> float | None:
    return num / den if den else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def aggregate(results: list[Result]) -> dict[str, Any]:
    n = len(results)
    status = Counter(r.status for r in results)
    ok = [r for r in results if r.status == "ok"]
    judged = [r for r in ok if r.verdict is not None]
    verdicts = Counter(r.verdict for r in judged)

    flags = {}
    for name in FLAGS:
        known = [r.flags[name] for r in judged if r.flags and r.flags[name] is not None]
        true = sum(known)
        flags[name] = {"true": true, "known": len(known), "rate": _rate(true, len(known))}

    final = sum(r.compliance["final_rejections"] for r in results)
    # Recomputed (--k) results cannot know their retries; then calls and retries are None.
    known = all(r.compliance["calls"] is not None for r in results)
    calls = sum(r.compliance["calls"] for r in results) if known else None
    retries = sum(r.compliance["retries"] for r in results) if known else None
    samples = (
        calls - retries
        if calls is not None and retries is not None
        else sum(len(p.parsed) for r in results for p in [*r.probes.values(), *r.loo.values()])
    )
    aligned = [r.alignment for r in judged if r.alignment is not None]
    recalls = [a["recall"] for a in aligned if a["recall"] is not None]

    rejections = {p: sum(r.probes[p].rejections for r in results if p in r.probes) for p in PROBES}
    rejections["L"] = sum(run.rejections for r in results for run in r.loo.values())

    return {
        "models": sorted({r.model for r in results}),
        "k": sorted({r.k for r in results}),
        "records": n,
        "status": {s: status[s] for s in STATUSES},
        "abstained": sum(1 for r in ok if r.A is not None and r.A.abstain),
        "judged": len(judged),
        "verdicts": {v: {"count": verdicts[v], "rate": _rate(verdicts[v], len(judged))}
                     for v in VERDICTS},
        "flags": flags,
        "compliance": {
            "records_compliant": _rate(n - status["non_compliant"], n),
            "samples": samples,
            "retries": retries,
            "final_rejections": final,
            "first_try_pass_rate": None if retries is None else _rate(samples - retries, samples),
            "final_pass_rate": _rate(samples - final, samples),
        },
        "alignment": {
            "records": len(aligned),
            "mean_precision": _mean([a["precision"] for a in aligned]),
            "mean_recall": _mean(recalls),
            "recall_defined": len(recalls),
        },
        "rejections": rejections,
    }  # fmt: skip


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def markdown(agg: dict[str, Any]) -> str:
    n = agg["records"]
    c, al = agg["compliance"], agg["alignment"]
    parts = [
        "# Poindexter report",
        f"models: {', '.join(agg['models'])}; k: {', '.join(map(str, agg['k']))}; "
        f"records: {n}; judged: {agg['judged']}; abstained: {agg['abstained']}",
        "## Status",
        _table(["status", "count", "rate"],
               [[s, v, _rate(v, n)] for s, v in agg["status"].items()]),
        "## Verdicts",
        _table(["verdict", "count", "rate"],
               [[v, d["count"], d["rate"]] for v, d in agg["verdicts"].items()]),
        "## Flags",
        _table(["flag", "true", "known", "rate"],
               [[f, d["true"], d["known"], d["rate"]] for f, d in agg["flags"].items()]),
        "## Compliance",
        _table(["metric", "value"], [[k, v] for k, v in c.items()]),
        "## Alignment",
        _table(["records", "mean precision", "mean recall", "recall defined"],
               [[al["records"], al["mean_precision"], al["mean_recall"], al["recall_defined"]]]),
        "## Rejections per probe",
        _table(["probe", "rejections"], [[p, v] for p, v in agg["rejections"].items()]),
    ]  # fmt: skip
    return "\n\n".join(parts) + "\n"
