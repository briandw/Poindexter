"""Rebuild the README's summary table and figure from results/.

For every ground-truth condition: how often Poindexter says `decorative`, and how often
the `span_in_cites` baseline would (answer text not found in the cited units). Rows where
the truth is `decorative` want a high rate; rows where it is `grounded` want a low one.

    uv run python scripts/summary.py            # prints the markdown table
    uv run python scripts/summary.py --png docs/summary.png
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from matplotlib.figure import Figure

from poindexter import bench
from poindexter.contract import Result, canonical

ROOT = Path(__file__).resolve().parent.parent / "results"
MODELS = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 5.5"}


def load(path: Path) -> list[Result]:
    with gzip.open(path, "rt") as f:
        return [Result.from_json(json.loads(line)) for line in f if line.strip()]


def row(label: str, truth: str, results: list[Result]) -> dict:
    """Rates over results that have a verdict (abstained answers have none)."""
    judged = [r for r in results if r.verdict is not None]
    dec = sum(r.verdict == "decorative" for r in judged)
    span_false = sum(not r.flags["span_in_cites"] for r in judged)
    return {
        "label": label,
        "truth": truth,
        "n": len(judged),
        "poindexter": bench.rate(dec, len(judged)),
        "baseline": bench.rate(span_false, len(judged)),
    }


def swap_rows(model: str) -> list[dict]:
    d = ROOT / f"{model}-swap"
    originals = {r.record.id: r for r in load(d / "original_results.jsonl.gz")}
    swapped = load(d / "swapped_results.jsonl.gz")
    classes = bench.classify_swaps(list(originals.values()), swapped)
    by = {"grounded": [], "decorative": []}
    for oid, (cls, _) in classes.items():
        if cls in by:
            by[cls].append(originals[oid])
    redundant = load(d / "redundant_results.jsonl.gz")
    kept = [
        r for r in redundant
        if r.status == "ok" and r.A and not r.A.abstain and bench.is_correct(r)
        and len({r.record.meta["redundant"]["copy_of"], r.record.meta["redundant"]["copy_id"]}
                & set(r.A.cites)) == 1
    ]  # fmt: skip
    name = MODELS[model]
    return [
        row(f"{name}: swap-confirmed grounded", "grounded", by["grounded"]),
        row(f"{name}: swap-labelled decorative", "decorative", by["decorative"]),
        row(f"{name}: redundant copy (constructed)", "decorative", kept),
    ]


def explore_rows() -> list[dict]:
    d = ROOT / "haiku-explore"
    originals = {r.record.id: r for r in load(d / "original_results.jsonl.gz")}
    swapped = load(d / "swapped_results.jsonl.gz")
    rows = []
    for kind, label in (("swap", "plausible swap"), ("wide", "wide swap")):
        by = {"grounded": [], "decorative": []}
        for s in swapped:
            if not s.record.id.endswith(f"~{kind}"):
                continue
            cls, _ = bench.swap_class(s)
            cf = s.record.meta["counterfactual"]
            o = originals[cf["source_id"]]
            if cls not in by or o.A is None or o.A.abstain:
                continue
            if canonical(o.A.text or "") != canonical(cf["original"]):
                continue
            if not set(o.record.gold) & set(o.A.cites):
                continue
            by[cls].append(o)
        rows.append(row(f"Haiku, open agent: grounded ({label})", "grounded", by["grounded"]))
        rows.append(
            row(f"Haiku, open agent: memory override ({label})", "decorative", by["decorative"])
        )
    return rows


def fmt(r: dict) -> str:
    if r["rate"] is None:
        return "n/a"
    return f"{r['k']}/{r['n']} = {r['rate']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"


def table(rows: list[dict]) -> str:
    lines = [
        "| condition | truth | Poindexter says decorative | baseline (span not in cites) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['label']} | {r['truth']} | {fmt(r['poindexter'])} | "
                     f"{fmt(r['baseline'])} |")  # fmt: skip
    return "\n".join(lines)


def figure(rows: list[dict], out: Path) -> None:
    rows = [r for r in rows if r["n"]]
    fig = Figure(figsize=(9.5, 0.55 * len(rows) + 1.6))
    ax = fig.subplots()
    for i, r in enumerate(reversed(rows)):
        for key, dy, style in (("poindexter", 0.12, dict(color="#2a78d6", marker="o")),
                               ("baseline", -0.12, dict(color="#777", marker="s",
                                                        markerfacecolor="white"))):  # fmt: skip
            v = r[key]
            ax.errorbar(v["rate"], i + dy, xerr=[[v["rate"] - v["lo"]], [v["hi"] - v["rate"]]],
                        fmt=style["marker"], color=style["color"],
                        markerfacecolor=style.get("markerfacecolor", style["color"]),
                        capsize=3, markersize=6)  # fmt: skip
        ax.axhspan(i - 0.45, i + 0.45, color="#fbe9e5" if r["truth"] == "decorative" else
                   "#e8f0fb", zorder=0)  # fmt: skip
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['label']} (n={r['n']})" for r in reversed(rows)], fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("share called decorative (Wilson 95% interval)")
    ax.set_title("Want high on red rows (truth: decorative), low on blue rows (truth: grounded)",
                 fontsize=10, loc="left")  # fmt: skip
    ax.plot([], [], "o", color="#2a78d6", label="Poindexter verdict")
    ax.plot([], [], "s", color="#777", markerfacecolor="white",
            label="span_in_cites baseline")  # fmt: skip
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    rows = swap_rows("haiku") + swap_rows("sonnet") + explore_rows()
    print(table(rows))
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure(rows, args.png)


if __name__ == "__main__":
    main()
