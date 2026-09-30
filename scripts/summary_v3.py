"""Rebuild the README's v3 table and figure from results/v3/.

Follow rate (the answer switched to the edited value) per model x agent x edit size.

    uv run python scripts/summary_v3.py
    uv run python scripts/summary_v3.py --png docs/sweep_v3.png
    uv run python scripts/summary_v3.py --root /tmp   # a rerun
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

RESULTS = Path(__file__).resolve().parent.parent / "results" / "v3"
MODELS = (("haiku", "Haiku 4.5"), ("sonnet", "Sonnet 5.5"))
AGENTS = (
    ("context", "context-only", "-"),
    ("document", "document is the authority", "--"),
    ("open", "knowledge allowed", ":"),
)
NUMBER = (("S1", "±1–2y / 5–10%"), ("S2", "±3–5y / 15–25%"), ("S3", "±10–25y / 40–60%"),
          ("S4", "±50–100y / ×3–5"), ("S5", "±200–500y / ×20–50"))  # fmt: skip
ENTITY = (("E1", "plausible"), ("E2", "pool"), ("E3", "absurd"))


def load(root: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for model, _ in MODELS:
        for agent, _, _ in AGENTS:
            rep = json.loads((root / f"v3-{model}-{agent}" / "sweep.json").read_text())
            out[(model, agent)] = rep["agents"][agent]["curve"]
    return out


def table(curves: dict) -> str:
    sizes = [s for s, _ in NUMBER] + [s for s, _ in ENTITY]
    lines = ["| model, agent | " + " | ".join(sizes) + " |", "|---" * (len(sizes) + 1) + "|"]
    for model, mlabel in MODELS:
        for agent, alabel, _ in AGENTS:
            c = curves[(model, agent)]
            cells = []
            for s in sizes:
                kind = "number" if s.startswith("S") else "entity"
                r = c.get(f"{kind}/{s}", {}).get("follow_rate")
                cells.append("n/a" if not r or r["rate"] is None else f"{r['rate']:.2f}")
            lines.append(f"| {mlabel}, {alabel} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(curves: dict, out: Path) -> None:
    fig = Figure(figsize=(11, 4.6))
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [5, 3]}, sharey=True)
    colors = {"haiku": "#d6602a", "sonnet": "#2a78d6"}
    for ax, kind, sizes in ((axes[0], "number", NUMBER), (axes[1], "entity", ENTITY)):
        xs = list(range(len(sizes)))
        for model, mlabel in MODELS:
            for agent, alabel, style in AGENTS:
                c = curves[(model, agent)]
                pts = [c.get(f"{kind}/{s}", {}).get("follow_rate") for s, _ in sizes]
                ys = [p["rate"] if p and p["rate"] is not None else float("nan") for p in pts]
                pairs = list(zip(ys, pts, strict=True))
                lo = [y - p["lo"] if p and p["rate"] is not None else 0 for y, p in pairs]
                hi = [p["hi"] - y if p and p["rate"] is not None else 0 for y, p in pairs]
                ax.errorbar(xs, ys, yerr=[lo, hi], color=colors[model], linestyle=style,
                            marker="o", markersize=4, capsize=2,
                            label=f"{mlabel}, {alabel}")  # fmt: skip
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s}\n{lab}" for s, lab in sizes], fontsize=7.5)
        ax.set_title("numbers" if kind == "number" else "entities", fontsize=10, loc="left")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("follow rate (answer = edited value)")
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))
    fig.suptitle("How far does the answer follow an edit to the cited fact?", x=0.01,
                 ha="left", fontsize=11)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", type=Path)
    parser.add_argument("--root", type=Path, default=RESULTS)
    args = parser.parse_args()
    curves = load(args.root)
    print(table(curves))
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure(curves, args.png)


if __name__ == "__main__":
    main()
