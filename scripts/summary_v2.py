"""Rebuild the README's v2 table and figure from results/v2/.

For every model x agent cell, how often probe C, the v1 removal verdict and the
span_in_cites baseline call a citation decorative, split by the truth from the mild
(L1) twin. Rates are over calls made, unstable counted as not decorative
(THRESHOLDS-v2.md).

    uv run python scripts/summary_v2.py
    uv run python scripts/summary_v2.py --png docs/summary_v2.png
    uv run python scripts/summary_v2.py --root /tmp   # a rerun
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

RESULTS = Path(__file__).resolve().parent.parent / "results" / "v2"
CELLS = [
    ("haiku", "context", "Haiku 4.5, context-only"),
    ("sonnet", "context", "Sonnet 5.5, context-only"),
    ("haiku", "open", "Haiku 4.5, knowledge allowed"),
    ("sonnet", "open", "Sonnet 5.5, knowledge allowed"),
]
CHECKERS = (("C", "probe C"), ("removal", "removal (v1)"), ("span_baseline", "span baseline"))


def load(root: Path = RESULTS) -> list[dict]:
    rows = []
    for model, agent, label in CELLS:
        rep = json.loads((root / f"v2-{model}-{agent}" / "surprise.json").read_text())
        sec = rep["agents"][agent]
        head = sec["headline"]
        for truth, key in (("decorative", "recall"), ("grounded", "false_alarm")):
            n = head["classes"].get(truth, 0)
            if not n:
                continue
            rows.append({
                "label": f"{label}: truth {truth} (n={n})",
                "truth": truth,
                "judged": n >= 100,
                **{name: head[name][key]["rate"] for name, _ in CHECKERS},
            })  # fmt: skip
        ctl = sec["control"]
        rows.append({
            "label": f"{label}: invented facts (n={ctl['scored']})",
            "truth": "grounded",
            "judged": True,
            **{name: ctl[name]["false_alarm"]["rate"] for name, _ in CHECKERS},
        })  # fmt: skip
    return rows


def fmt(r: dict) -> str:
    if r["rate"] is None:
        return "n/a"
    return f"{r['k']}/{r['n']} = {r['rate']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"


def table(rows: list[dict]) -> str:
    lines = [
        "| cell | judged | probe C says decorative | removal says decorative | span baseline |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {'yes' if r['judged'] else 'no'} | {fmt(r['C'])} | "
            f"{fmt(r['removal'])} | {fmt(r['span_baseline'])} |"
        )
    return "\n".join(lines)


def figure(rows: list[dict], out: Path) -> None:
    fig = Figure(figsize=(10, 0.62 * len(rows) + 1.6))
    ax = fig.subplots()
    styles = {
        "C": dict(color="#2a78d6", marker="o", dy=0.2),
        "removal": dict(color="#d6602a", marker="D", dy=0.0),
        "span_baseline": dict(color="#777", marker="s", dy=-0.2),
    }
    for i, r in enumerate(reversed(rows)):
        ax.axhspan(i - 0.45, i + 0.45, zorder=0,
                   color="#fbe9e5" if r["truth"] == "decorative" else "#e8f0fb")  # fmt: skip
        for name, st in styles.items():
            v = r[name]
            if v["rate"] is None:
                continue
            ax.errorbar(v["rate"], i + st["dy"],
                        xerr=[[v["rate"] - v["lo"]], [v["hi"] - v["rate"]]],
                        fmt=st["marker"], color=st["color"], capsize=3, markersize=5)  # fmt: skip
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [r["label"] + ("" if r["judged"] else "  [not judged]") for r in reversed(rows)],
        fontsize=8,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("share called decorative (Wilson 95% interval)")
    ax.set_title("Want high on red rows (truth: decorative), low on blue rows (truth: grounded)",
                 fontsize=10, loc="left")  # fmt: skip
    for name, label in CHECKERS:
        st = styles[name]
        ax.plot([], [], st["marker"], color=st["color"], label=label)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", type=Path)
    parser.add_argument("--root", type=Path, default=RESULTS,
                        help="directory holding v2-<model>-<agent>/ (default: results/v2)")
    args = parser.parse_args()
    rows = load(args.root)
    print(table(rows))
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure(rows, args.png)


if __name__ == "__main__":
    main()
