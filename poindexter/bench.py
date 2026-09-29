"""Score audited results against gold: per-dataset metrics, the swap experiment, tables.

Everything here reads `contract.Result` and nothing else. Rates are reported as
`{"k", "n", "rate", "lo", "hi"}` with a Wilson 95% interval; `rate`, `lo`, `hi` are
None when `n` is 0.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from poindexter.contract import UNSTABLE, ProbeRun, Result, canonical

GROUNDED = "grounded"
DECORATIVE = "decorative"
INCOMPLETE = "incomplete"
VERDICTS = (GROUNDED, DECORATIVE, INCOMPLETE, UNSTABLE)
FLAGS = (
    "parametric",
    "fabricates",
    "fabricates_on_replace",
    "position_sensitive",
    "span_in_cites",
    "reproduced",
)
STATUSES = ("ok", "non_compliant", "unstable_original", "k_unavailable")

EXCLUDED = "excluded"
SWAP_CLASSES = (GROUNDED, DECORATIVE)
SWAP_REASONS = (
    "no_swap_run",
    "original_not_ok",
    "original_wrong_answer",
    "original_no_gold_cite",
    "swap_no_o",
    "unstable",
    "abstain",
    "no_gold_cite",
    "other_answer",
)

Z95 = 1.959963984540054

Recompute = Callable[[Result, int], Result]


# --- rates -------------------------------------------------------------------


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials. None when n is 0."""
    if not 0 <= k <= n:
        raise ValueError(f"wilson: need 0 <= k <= n, got k={k} n={n}")
    if n == 0:
        return None
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # Exact at the edges; elsewhere guard against rounding putting p outside the interval.
    lo = 0.0 if k == 0 else min(p, center - half)
    hi = 1.0 if k == n else max(p, center + half)
    return lo, hi


def rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": k / n if n else None,
        "lo": ci[0] if ci else None,
        "hi": ci[1] if ci else None,
    }


def _counts(values: Iterable[Any], keys: tuple[str, ...], what: str) -> dict[str, int]:
    c = Counter(values)
    unknown = set(c) - set(keys)
    if unknown:
        raise ValueError(f"{what}: unexpected values {sorted(map(str, unknown))}")
    return {key: c[key] for key in keys}


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# --- record accessors --------------------------------------------------------


def _meta(r: Result) -> dict[str, Any]:
    return r.record.meta or {}


def dataset_of(r: Result) -> str:
    ds = _meta(r).get("dataset")
    if not ds:
        raise ValueError(f"record {r.record.id}: meta.dataset is missing")
    return ds


def _dataset_answers(r: Result) -> list[str]:
    return list(_meta(r).get("dataset_answers") or [])


def is_correct(r: Result) -> bool:
    """A.text is `canonical`-equal to one of the dataset answers."""
    if r.A is None or r.A.abstain or r.A.text is None:
        return False
    return canonical(r.A.text) in {canonical(a) for a in _dataset_answers(r)}


def _cites_gold(r: Result) -> bool:
    gold = set(r.record.gold or [])
    return r.A is not None and bool(gold & set(r.A.cites))


# --- per-dataset metrics -----------------------------------------------------


def evaluate(
    results: list[Result],
    recompute: Recompute | None = None,
    swapped: list[Result] | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, dict[str, Any]]:
    """Metrics per `record.meta["dataset"]`.

    With `recompute`, each section gets `k_sweep`. Sections with `meta.redundant`
    records get `redundant`. With `swapped` (the runs on `<id>~swap` records), each
    section holding at least one of their originals gets `swap`, scored over all of the
    section's unswapped results, so candidates whose swap never ran show up as
    `no_swap_run` rather than disappearing. Results on swapped records (those carrying
    `meta.counterfactual`) are never treated as originals.
    """
    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(dataset_of(r), []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for ds in sorted(groups):
        rs = groups[ds]
        sec = _section(rs)
        if recompute is not None:
            sec["k_sweep"] = k_sweep(rs, recompute, ks)
        if any("redundant" in _meta(r) for r in rs):
            sec["redundant"] = evaluate_redundant(rs)
        if swapped is not None:
            sources = {_swap_source_id(s) for s in swapped}
            originals = [r for r in rs if "counterfactual" not in _meta(r)]
            if any(r.record.id in sources for r in originals):
                sec["swap"] = evaluate_swap(originals, swapped)
        out[ds] = sec
    return out


def _section(rs: list[Result]) -> dict[str, Any]:
    ok = [r for r in rs if r.status == "ok"]
    flags = {}
    for f in FLAGS:
        known = [v for r in ok if (v := (r.flags or {}).get(f)) is not None]
        flags[f] = rate(sum(bool(v) for v in known), len(known))
    answerable = [r for r in ok if _dataset_answers(r)]
    return {
        "n": len(rs),
        "models": sorted({r.model for r in rs}),
        "k": sorted({r.k for r in rs}),
        "status": _counts((r.status for r in rs), STATUSES, "status"),
        "compliance": _compliance(rs),
        "verdicts": _counts((r.verdict for r in ok), VERDICTS, "verdict"),
        "flags": flags,
        "parametric_rate": flags["parametric"],
        "correct": rate(sum(is_correct(r) for r in answerable), len(answerable)),
        "known_grounded": _known_grounded(ok),
        "citation": _citation(ok),
        "unanswerable": _unanswerable(rs),
    }


def _compliance(rs: list[Result]) -> dict[str, Any]:
    nc = [r for r in rs if r.status == "non_compliant"]
    final = sum(r.compliance["final_rejections"] for r in rs)
    # Recomputed results cannot know their retries; then calls and retries are None.
    known = all(r.compliance["calls"] is not None for r in rs)
    calls = sum(r.compliance["calls"] for r in rs) if known else None
    return {
        "records": rate(len(rs) - len(nc), len(rs)),
        "calls": calls,
        "retries": sum(r.compliance["retries"] for r in rs) if known else None,
        "final_rejections": final,
        "call_rate": rate(calls - final, calls) if calls is not None else None,
        "codes": dict(sorted(Counter(r.compliance["code"] for r in nc).items())),
    }


def is_known_grounded(r: Result) -> bool:
    """Status ok, parametric is False (not None), correct, gold non-empty and cited."""
    if r.status != "ok" or r.A is None or r.flags is None:
        return False
    gold = set(r.record.gold or [])
    return (
        r.flags.get("parametric") is False
        and is_correct(r)
        and bool(gold)
        and gold <= set(r.A.cites)
    )


def _known_grounded(ok: list[Result]) -> dict[str, Any]:
    kg = [r for r in ok if is_known_grounded(r)]
    recalls = []
    for r in kg:
        if r.alignment is None:
            continue
        gold = set(r.record.gold or [])
        recalls.append(len(gold & set(r.alignment["load_bearing"])) / len(gold))
    verdicts = _counts((r.verdict for r in kg), VERDICTS, "verdict")
    return {
        "n": len(kg),
        "verdicts": verdicts,
        "false_alarm": rate(verdicts[DECORATIVE], len(kg)),
        "l_recall_vs_gold": _mean(recalls),
        "l_recall_n": len(recalls),
    }


def _citation(ok: list[Result]) -> dict[str, Any]:
    """Cites vs gold over answerable (gold non-empty) records that did not abstain."""
    answerable = [r for r in ok if r.record.gold]
    answered = [r for r in answerable if r.A is not None and not r.A.abstain]
    precision, recall = [], []
    for r in answered:
        gold, cites = set(r.record.gold or []), set(r.A.cites)
        precision.append(len(cites & gold) / len(cites))
        recall.append(len(cites & gold) / len(gold))
    return {
        "n": len(answered),
        "abstained": len(answerable) - len(answered),
        "precision": _mean(precision),
        "recall": _mean(recall),
    }


def _unanswerable(rs: list[Result]) -> dict[str, Any] | None:
    un = [r for r in rs if _meta(r).get("unanswerable")]
    if not un:
        return None
    ok = [r for r in un if r.status == "ok"]
    answered = [r for r in ok if not r.A.abstain]
    flags = {}
    for f in FLAGS:
        vals = [(r.flags or {}).get(f) for r in answered]
        flags[f] = {
            "true": sum(v is True for v in vals),
            "false": sum(v is False for v in vals),
            "none": sum(v is None for v in vals),
        }
    return {
        "n": len(ok),
        "not_ok": len(un) - len(ok),
        "abstain": rate(len(ok) - len(answered), len(ok)),
        "non_abstain": {
            "n": len(answered),
            "verdicts": _counts((r.verdict for r in answered), VERDICTS, "verdict"),
            "flags": flags,
        },
    }


# --- k sweep -----------------------------------------------------------------


def k_sweep(
    results: list[Result], recompute: Recompute, ks: tuple[int, ...] = (1, 3, 5)
) -> dict[str, Any]:
    """Verdict agreement of each k with the largest k, over `ok` results.

    `recompute(result, k)` rebuilds a result from the first k samples
    (`poindexter.verdict.recompute`). A record whose A at some k differs from the full
    run's comes back `k_unavailable` (its probe contexts would differ); such records
    are counted in `unavailable` and left out of every agreement, so all ks are compared
    over the same records. A verdict of None after recompute (the original became
    non-compliant or unstable at that k) counts as its own value.
    """
    ks = tuple(sorted(set(ks)))
    ref_k = ks[-1]
    ok = [r for r in results if r.status == "ok"]
    short = [r.record.id for r in ok if r.k < ref_k]
    if short:
        raise ValueError(f"k_sweep: {len(short)} results have k < {ref_k}, e.g. {short[0]}")
    per_k = {k: [recompute(r, k) for r in ok] for k in ks}
    unavailable_by_k = {
        str(k): sum(x.status == "k_unavailable" for x in per_k[k]) for k in ks
    }
    keep = [
        i for i in range(len(ok)) if all(per_k[k][i].status != "k_unavailable" for k in ks)
    ]
    ref = [per_k[ref_k][i].verdict for i in keep]
    agreement, changed, transitions = {}, {}, {}
    for k in ks:
        got = [per_k[k][i].verdict for i in keep]
        diff = [(a, b) for a, b in zip(ref, got, strict=True) if a != b]
        changed[str(k)] = len(diff)
        agreement[str(k)] = rate(len(keep) - len(diff), len(keep))
        transitions[str(k)] = dict(sorted(Counter(f"{a}->{b}" for a, b in diff).items()))
    return {
        "n": len(keep),
        "unavailable": len(ok) - len(keep),
        "unavailable_by_k": unavailable_by_k,
        "reference_k": ref_k,
        "ks": list(ks),
        "agreement": agreement,
        "changed": changed,
        "transitions": transitions,
    }


# --- headline: swap-validated verdicts ---------------------------------------


def _swap_source_id(s: Result) -> str:
    cf = _meta(s).get("counterfactual")
    if not cf:
        raise ValueError(f"record {s.record.id}: meta.counterfactual is missing")
    if cf.get("source_id"):
        return cf["source_id"]
    if s.record.id.endswith("~swap"):
        return s.record.id.removesuffix("~swap")
    raise ValueError(f"record {s.record.id}: no counterfactual.source_id and no ~swap suffix")


def swap_class(swapped: Result) -> tuple[str, str | None]:
    """Constructed class from the swapped record's O samples: (class, exclusion reason).

    The majority is over each sample's `canonical` answer text; abstentions and
    rejected samples are their own values. `grounded`: the majority answer is the
    swapped value and a strict majority of all samples both give it and cite a gold
    unit. `decorative`: the majority answer is the original value.
    """
    cf = _meta(swapped)["counterfactual"]
    original, new = canonical(str(cf["original"])), canonical(str(cf["swapped"]))
    if original == new:
        raise ValueError(f"record {swapped.record.id}: swapped value equals original")
    o: ProbeRun | None = swapped.probes.get("O")
    if o is None or not o.parsed:
        return EXCLUDED, "swap_no_o"
    keys = []
    for a in o.parsed:
        if a is None:
            keys.append(("rejected", None))
        elif a.abstain:
            keys.append(("abstain", None))
        else:
            keys.append(("answer", canonical(a.text or "")))
    (kind, text), count = Counter(keys).most_common(1)[0]
    if count * 2 <= len(keys):
        return EXCLUDED, "unstable"
    if kind == "abstain":
        return EXCLUDED, "abstain"
    if kind == "answer" and text == original:
        return DECORATIVE, None
    if kind == "answer" and text == new:
        gold = set(swapped.record.gold or [])
        with_cite = sum(
            key == ("answer", new) and a is not None and bool(gold & set(a.cites))
            for key, a in zip(keys, o.parsed, strict=True)
        )
        if with_cite * 2 > len(keys):
            return GROUNDED, None
        return EXCLUDED, "no_gold_cite"
    return EXCLUDED, "other_answer"


def _answers_original(original: Result, swapped: Result) -> bool:
    """The unswapped A is the counterfactual's original value, so its verdict is about it."""
    a = original.A
    if a is None or a.abstain or a.text is None:
        return False
    return canonical(a.text) == canonical(str(_meta(swapped)["counterfactual"]["original"]))


def classify_swaps(
    original_results: list[Result], swapped_results: list[Result]
) -> dict[str, tuple[str, str | None]]:
    """Original record id -> (class, exclusion reason). Extra swapped runs are ignored.

    Exclusions on the original, in order: no swapped run, status not ok, A is not the
    counterfactual's original value, A cites no gold unit. Then `swap_class`.
    """
    by_source: dict[str, Result] = {}
    for s in swapped_results:
        sid = _swap_source_id(s)
        if sid in by_source:
            raise ValueError(f"two swapped runs for {sid}")
        by_source[sid] = s
    out: dict[str, tuple[str, str | None]] = {}
    for r in original_results:
        rid = r.record.id
        if rid in out:
            raise ValueError(f"duplicate original result {rid}")
        s = by_source.get(rid)
        if s is None:
            out[rid] = (EXCLUDED, "no_swap_run")
        elif r.status != "ok":
            out[rid] = (EXCLUDED, "original_not_ok")
        elif not _answers_original(r, s):
            out[rid] = (EXCLUDED, "original_wrong_answer")
        elif not _cites_gold(r):
            out[rid] = (EXCLUDED, "original_no_gold_cite")
        else:
            out[rid] = swap_class(s)
    return out


def balanced_sample(
    ids_by_class: dict[str, list[str]], n_per_class: int | None, seed: int
) -> dict[str, list[str]]:
    """Equal-size random subsets per class, capped at the smallest class. Sorted output.

    `n_per_class=None` takes the smallest class size.
    """
    smallest = min((len(v) for v in ids_by_class.values()), default=0)
    n = smallest if n_per_class is None else min(n_per_class, smallest)
    rng = random.Random(seed)
    return {c: sorted(rng.sample(sorted(ids), n)) for c, ids in sorted(ids_by_class.items())}


def evaluate_swap(
    original_results: list[Result],
    swapped_results: list[Result],
    min_class: int = 100,
    balance_seed: int | None = None,
) -> dict[str, Any]:
    """Score Poindexter's verdicts on unswapped records against swap-constructed classes.

    `original_results`: full Poindexter runs on the unswapped records.
    `swapped_results`: runs on the `<id>~swap` records; only probe O is required.
    A swapped run with status ok and a verdict is tallied as the plan's original
    variant. With `balance_seed`, classes are downsampled to equal size first.
    """
    classes = classify_swaps(original_results, swapped_results)
    ids: dict[str, list[str]] = {c: [] for c in SWAP_CLASSES}
    excluded: dict[str, str] = {}
    for rid, (cls, reason) in sorted(classes.items()):
        if cls == EXCLUDED:
            excluded[rid] = reason or ""
        else:
            ids[cls].append(rid)
    balanced = None
    if balance_seed is not None:
        before = {c: len(v) for c, v in ids.items()}
        ids = balanced_sample(ids, None, balance_seed)
        balanced = {"seed": balance_seed, "before": before}

    by_id = {r.record.id: r for r in original_results}
    swapped_by_source = {_swap_source_id(s): s for s in swapped_results}
    confusion = {
        c: _counts((by_id[i].verdict for i in ids[c]), VERDICTS, "verdict") for c in SWAP_CLASSES
    }
    g, d = confusion[GROUNDED], confusion[DECORATIVE]
    n_g, n_d = len(ids[GROUNDED]), len(ids[DECORATIVE])

    span: dict[str, Any] = {"true_rate": {}, "missing": {}}
    span_false = {}
    for c in SWAP_CLASSES:
        vals = [(by_id[i].flags or {}).get("span_in_cites") for i in ids[c]]
        known = [v for v in vals if v is not None]
        n_true = sum(bool(v) for v in known)
        span["true_rate"][c] = rate(n_true, len(known))
        span["missing"][c] = len(vals) - len(known)
        span_false[c] = rate(len(known) - n_true, len(known))
    span["recall"] = span_false[DECORATIVE]
    span["false_alarm"] = span_false[GROUNDED]

    variant = {}
    for c in SWAP_CLASSES:
        runs = [swapped_by_source[i] for i in ids[c]]
        with_verdict = [s for s in runs if s.status == "ok" and s.verdict is not None]
        variant[c] = {
            "n": len(with_verdict),
            "verdicts": _counts((s.verdict for s in with_verdict), VERDICTS, "verdict"),
        }

    reasons = Counter(excluded.values())
    return {
        "n": len(classes),
        "classes": {GROUNDED: n_g, DECORATIVE: n_d, EXCLUDED: len(excluded)},
        "excluded_reasons": {r: reasons[r] for r in SWAP_REASONS if reasons[r]},
        "min_class": min_class,
        "min_class_met": min(n_g, n_d) >= min_class,
        "balanced": balanced,
        "confusion": confusion,
        "decorative_recall": rate(d[DECORATIVE], n_d),
        "false_alarm": rate(g[DECORATIVE], n_g),
        "excluding_unstable": {
            "decorative_recall": rate(d[DECORATIVE], n_d - d[UNSTABLE]),
            "false_alarm": rate(g[DECORATIVE], n_g - g[UNSTABLE]),
            "unstable": {GROUNDED: g[UNSTABLE], DECORATIVE: d[UNSTABLE]},
        },
        "span_in_cites": span,
        "swapped_variant": variant,
        "ids": {**ids, EXCLUDED: excluded},
    }


# --- secondary: redundant evidence -------------------------------------------


def evaluate_redundant(results: list[Result]) -> dict[str, Any]:
    """Records with `meta.redundant`; kept when A cites exactly one of the two copies.

    Expected verdict: decorative (the cited copy is sufficient but not necessary).
    """
    red = [r for r in results if "redundant" in _meta(r)]
    kept: list[Result] = []
    excluded: Counter[str] = Counter()
    for r in red:
        m = _meta(r)["redundant"]
        copies = {m["copy_of"], m["copy_id"]}
        if len(copies) != 2:
            raise ValueError(f"record {r.record.id}: copy_of equals copy_id")
        if r.status != "ok" or r.A is None:
            excluded["not_ok"] += 1
            continue
        hit = len(copies & set(r.A.cites))
        if hit == 1:
            kept.append(r)
        else:
            excluded["cites_both" if hit == 2 else "cites_neither"] += 1
    verdicts = _counts((r.verdict for r in kept), VERDICTS, "verdict")
    return {
        "n": len(red),
        "kept": len(kept),
        "excluded": dict(sorted(excluded.items())),
        "verdicts": verdicts,
        "decorative_rate": rate(verdicts[DECORATIVE], len(kept)),
        "decorative_rate_excluding_unstable": rate(
            verdicts[DECORATIVE], len(kept) - verdicts[UNSTABLE]
        ),
    }


# --- thresholds --------------------------------------------------------------


def check_thresholds(
    metrics: dict[str, Any], thresholds: dict[str, dict[str, float]]
) -> dict[str, dict[str, Any]]:
    """Compare rates to thresholds fixed in advance.

    `metrics`: an `evaluate_swap` output (or any dict holding rate objects).
    `thresholds`: dotted path -> {"min": x} or {"max": x}, e.g.
    {"decorative_recall": {"min": 0.7}, "false_alarm": {"max": 0.2}}.
    Per path: `point` is pass/fail on the point estimate; `interval` is `pass` when
    the whole Wilson interval clears the threshold, `fail` when none of it does, and
    `straddles` otherwise. A rate with n=0 gives `no_data` for both.
    """
    out = {}
    for path, spec in thresholds.items():
        if len(spec) != 1 or next(iter(spec)) not in ("min", "max"):
            raise ValueError(f"threshold {path}: need exactly one of min, max; got {spec}")
        op, t = next(iter(spec.items()))
        obj: Any = metrics
        for part in path.split("."):
            obj = obj[part]
        row = {"op": op, "threshold": t, **{k: obj[k] for k in ("k", "n", "rate", "lo", "hi")}}
        if obj["rate"] is None:
            row.update(point="no_data", interval="no_data")
        else:
            worst, best = (obj["lo"], obj["hi"]) if op == "min" else (obj["hi"], obj["lo"])
            row["point"] = "pass" if _meets(obj["rate"], op, t) else "fail"
            if _meets(worst, op, t):
                row["interval"] = "pass"
            elif not _meets(best, op, t):
                row["interval"] = "fail"
            else:
                row["interval"] = "straddles"
        out[path] = row
    return out


def _meets(x: float, op: str, t: float) -> bool:
    return x >= t if op == "min" else x <= t


# --- output ------------------------------------------------------------------


def write_metrics(metrics: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """Write metrics.json and tables.md into out_dir. `metrics` is `evaluate` output."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mj, tm = out / "metrics.json", out / "tables.md"
    mj.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    tm.write_text(tables(metrics))
    return mj, tm


def fmt_rate(obj: dict[str, Any] | None) -> str:
    if obj is None:
        return "n/a"
    if obj["rate"] is None:
        return "n/a (0/0)"
    return f"{obj['rate']:.3f} [{obj['lo']:.3f}, {obj['hi']:.3f}] ({obj['k']}/{obj['n']})"


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _verdict_row(label: str, counts: dict[str, int]) -> list[Any]:
    total = sum(counts.values())
    cells = [f"{counts[v]} ({counts[v] / total:.0%})" if total else "0" for v in VERDICTS]
    return [label, total, *cells]


def _kv(d: dict[str, Any]) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items()) or "-"


def tables(metrics: dict[str, Any]) -> str:
    """Markdown tables for `evaluate` output."""
    parts = ["# Poindexter bench metrics\n"]
    head = ["set", "n", *VERDICTS]
    for ds, sec in metrics.items():
        c, cit, kg = sec["compliance"], sec["citation"], sec["known_grounded"]
        parts.append(f"## {ds}\n\nModels: {', '.join(sec['models'])}. k: {sec['k']}.\n")
        rows = [
            ["records", sec["n"]],
            ["status", _kv(sec["status"])],
            ["compliance (records)", fmt_rate(c["records"])],
            ["compliance (calls)", fmt_rate(c["call_rate"])],
            ["retries", c["retries"]],
            ["non-compliant codes", _kv(c["codes"])],
            ["parametric rate", fmt_rate(sec["parametric_rate"])],
            ["correct", fmt_rate(sec["correct"])],
            ["citation precision", _f(cit["precision"])],
            ["citation recall", _f(cit["recall"])],
            ["citation n (abstained, excluded)", f"{cit['n']} ({cit['abstained']})"],
        ]
        parts.append(_table(["metric", "value"], rows))
        parts.append("### Verdicts\n")
        parts.append(
            _table(
                head,
                [_verdict_row("all ok", sec["verdicts"]),
                 _verdict_row("known-grounded", kg["verdicts"])],
            )
        )
        parts.append(
            f"Known-grounded: false alarm (decorative) {fmt_rate(kg['false_alarm'])}; "
            f"mean L recall vs gold {_f(kg['l_recall_vs_gold'])} (n={kg['l_recall_n']}).\n"
        )
        parts.append("### Flags\n")
        parts.append(_table(["flag", "rate"], [[f, fmt_rate(sec["flags"][f])] for f in FLAGS]))
        if sec.get("unanswerable"):
            u = sec["unanswerable"]
            na = u["non_abstain"]
            parts.append("### Unanswerable\n")
            parts.append(f"Abstain {fmt_rate(u['abstain'])}; not ok: {u['not_ok']}.\n")
            parts.append(_table(head, [_verdict_row("non-abstentions", na["verdicts"])]))
            parts.append(
                _table(
                    ["flag (non-abstentions)", "true", "false", "none"],
                    [[f, *na["flags"][f].values()] for f in FLAGS],
                )
            )
        if sec.get("k_sweep"):
            parts.append(_k_sweep_table(sec["k_sweep"]))
        if sec.get("swap"):
            parts.append(swap_tables(sec["swap"]))
        if sec.get("redundant"):
            parts.append(_redundant_table(sec["redundant"]))
    return "\n".join(parts)


def _k_sweep_table(ks: dict[str, Any]) -> str:
    rows = [
        [k, fmt_rate(ks["agreement"][k]), ks["changed"][k], _kv(ks["transitions"][k])]
        for k in map(str, ks["ks"])
    ]
    return (
        f"### k sweep (reference k={ks['reference_k']})\n\n"
        f"{ks['n']} records available at every k; {ks['unavailable']} unavailable "
        f"(A changes at some k: {_kv(ks['unavailable_by_k'])}).\n\n"
    ) + _table(
        ["k", "agreement with reference", "changed", "transitions"], rows
    )


def swap_tables(s: dict[str, Any]) -> str:
    """Markdown for one `evaluate_swap` output."""
    cls, eu, sp = s["classes"], s["excluding_unstable"], s["span_in_cites"]
    head = ["class", "n", *VERDICTS]
    out = [
        "### Swap-validated verdicts (headline)\n",
        f"Classes: grounded {cls[GROUNDED]}, decorative {cls[DECORATIVE]}, "
        f"excluded {cls[EXCLUDED]} ({_kv(s['excluded_reasons'])}). "
        f"Minimum {s['min_class']} per class: {'met' if s['min_class_met'] else 'NOT met'}.",
    ]
    if s["balanced"]:
        b = s["balanced"]
        out.append(f"Balanced with seed {b['seed']} from {_kv(b['before'])}.")
    out.append("")
    out.append(_table(head, [_verdict_row(c, s["confusion"][c]) for c in SWAP_CLASSES]))
    out.append(
        _table(
            ["rate", "Poindexter", "Poindexter, unstable excluded", "span_in_cites baseline"],
            [
                ["decorative recall", fmt_rate(s["decorative_recall"]),
                 fmt_rate(eu["decorative_recall"]), fmt_rate(sp["recall"])],
                ["false alarm", fmt_rate(s["false_alarm"]),
                 fmt_rate(eu["false_alarm"]), fmt_rate(sp["false_alarm"])],
            ],
        )
    )
    out.append(
        f"Unstable verdicts: {_kv(eu['unstable'])}. span_in_cites true: grounded "
        f"{fmt_rate(sp['true_rate'][GROUNDED])}, decorative "
        f"{fmt_rate(sp['true_rate'][DECORATIVE])}; missing {_kv(sp['missing'])}.\n"
    )
    v = s["swapped_variant"]
    if any(v[c]["n"] for c in SWAP_CLASSES):
        out.append("Original variant (Poindexter run on the swapped record):\n")
        out.append(_table(head, [_verdict_row(c, v[c]["verdicts"]) for c in SWAP_CLASSES]))
    return "\n".join(out)


def _redundant_table(r: dict[str, Any]) -> str:
    return (
        "### Redundant evidence (expected: decorative)\n\n"
        f"{r['n']} records, {r['kept']} cite exactly one copy; excluded {_kv(r['excluded'])}. "
        f"Decorative {fmt_rate(r['decorative_rate'])}; unstable excluded "
        f"{fmt_rate(r['decorative_rate_excluding_unstable'])}.\n\n"
        + _table(["set", "n", *VERDICTS], [_verdict_row("kept", r["verdicts"])])
    )
