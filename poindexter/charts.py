"""PNG charts from bench metrics. Matplotlib with the Agg canvas, no pyplot state.

Chart functions take `metrics_by_run`: run label (e.g. "haiku/squad") -> one dataset
section of `bench.evaluate` output, optionally carrying `swap`, `k_sweep`, `redundant`.
Each writes one PNG and returns its path, or returns None without writing when no run
has the data the chart needs. Runs missing an optional section are left out of that
panel; nothing is drawn as zero that was not measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from poindexter.bench import DECORATIVE, GROUNDED, INCOMPLETE, VERDICTS
from poindexter.contract import UNSTABLE

VERDICT_COLORS = {
    GROUNDED: "#2a78d6",
    DECORATIVE: "#eb6834",
    INCOMPLETE: "#1baf7a",
    UNSTABLE: "#b5b3ad",
}
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
POINDEXTER_COLOR = "#2a78d6"
BASELINE_COLOR = "#6b6a66"
INK = "#52514e"
GRID = "#e4e3df"

Metrics = dict[str, dict[str, Any]]


def _figure(width: float, height: float = 4.5, ncols: int = 1, **kw: Any) -> tuple[Figure, Any]:
    fig = Figure(figsize=(width, height), dpi=150, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols, **kw)
    for ax in axes if ncols > 1 else [axes]:
        _style(ax)
    return fig, axes


def _style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: Figure, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, facecolor="white")
    return p


def _width(n: int, per: float = 1.1, base: float = 3.0) -> float:
    return max(6.0, base + per * n)


def _err(r: dict[str, Any]) -> list[list[float]]:
    return [[r["rate"] - r["lo"]], [r["hi"] - r["rate"]]]


def _verdict_stack(ax: Axes, labels: list[str], counts: list[dict[str, int]]) -> None:
    """Stacked verdict shares; runs with no records get an empty slot labeled n=0."""
    bottoms = [0.0] * len(labels)
    for v in VERDICTS:
        shares = [c[v] / sum(c.values()) if sum(c.values()) else 0.0 for c in counts]
        bars = ax.bar(
            range(len(labels)), shares, 0.6, bottom=bottoms, color=VERDICT_COLORS[v],
            label=v, edgecolor="white", linewidth=1,
        )
        for bar, share, b in zip(bars, shares, bottoms, strict=True):
            if share >= 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, b + share / 2, f"{share:.0%}",
                    ha="center", va="center", fontsize=7,
                    color="black" if v == UNSTABLE else "white",
                )
        bottoms = [b + s for b, s in zip(bottoms, shares, strict=True)]
    ticks = [f"{lab}\nn={sum(c.values())}" for lab, c in zip(labels, counts, strict=True)]
    ax.set_xticks(range(len(labels)), ticks)
    ax.set_ylim(0, 1)


# --- 1. verdicts -------------------------------------------------------------


def verdicts(metrics_by_run: Metrics, out_path: str | Path) -> Path | None:
    """Stacked bar: verdict share over `ok` records, per run."""
    if not metrics_by_run:
        return None
    labels = list(metrics_by_run)
    fig, ax = _figure(_width(len(labels)))
    _verdict_stack(ax, labels, [metrics_by_run[lab]["verdicts"] for lab in labels])
    ax.set_ylabel("share of audited records")
    ax.set_xlabel("run (model/dataset)")
    ax.set_title("Verdict distribution", loc="left", fontsize=11)
    ax.legend(title="verdict", fontsize=8, title_fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), frameon=False)
    return _save(fig, out_path)


# --- 2. checker (headline first) --------------------------------------------


def _swap_panels(axes: list[Axes], swap_by_label: Metrics,
                 thresholds: dict[str, dict[str, float]] | None = None) -> None:
    """Two panels: decorative recall and false-alarm rate, Poindexter vs span_in_cites."""
    labels = list(swap_by_label)
    specs = [
        ("decorative_recall", "recall", "Decorative recall\nP(decorative | decorative class)"),
        ("false_alarm", "false_alarm", "False-alarm rate\nP(decorative | grounded class)"),
    ]
    for ax, (key, span_key, title) in zip(axes, specs, strict=True):
        for i, lab in enumerate(labels):
            s = swap_by_label[lab]
            for dx, r, color, marker, face in (
                (-0.12, s[key], POINDEXTER_COLOR, "o", POINDEXTER_COLOR),
                (0.12, s["span_in_cites"][span_key], BASELINE_COLOR, "s", "white"),
            ):
                if r["rate"] is None:
                    continue
                ax.errorbar(
                    i + dx, r["rate"], yerr=_err(r), fmt=marker, color=color,
                    markerfacecolor=face, markersize=7, capsize=4, linewidth=1.5,
                )
                ax.annotate(f"{r['rate']:.2f}", (i + dx, r["rate"]), xytext=(7, 0),
                            textcoords="offset points", fontsize=7, color=INK, va="center")
        if thresholds and key in thresholds:
            (op, t), = thresholds[key].items()
            ax.axhline(t, color=INK, linestyle="--", linewidth=1)
            ax.annotate(f"threshold ({op} {t:g})", (1, t), xycoords=("axes fraction", "data"),
                        xytext=(-2, 3), textcoords="offset points", ha="right", fontsize=7,
                        color=INK)
        ticks = [
            f"{lab}\ng={swap_by_label[lab]['classes'][GROUNDED]} "
            f"d={swap_by_label[lab]['classes'][DECORATIVE]}"
            for lab in labels
        ]
        ax.set_xticks(range(len(labels)), ticks)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("rate (Wilson 95% interval)")
        ax.set_title(title, loc="left", fontsize=9)
    fig = axes[0].get_figure()
    handles = [
        axes[0].plot([], [], "o", color=POINDEXTER_COLOR, label="Poindexter verdict")[0],
        axes[0].plot([], [], "s", color=BASELINE_COLOR, markerfacecolor="white",
                     label="span_in_cites baseline (span false read as decorative)")[0],
    ]
    fig.legend(handles=handles, loc="outside lower left", ncols=2, fontsize=8, frameon=False)


def checker(metrics_by_run: Metrics, out_path: str | Path) -> Path | None:
    """Swap-validated recall/false alarm when any run has `swap`, then known-grounded."""
    swap = {lab: sec["swap"] for lab, sec in metrics_by_run.items() if sec.get("swap")}
    labels = list(metrics_by_run)
    if not labels:
        return None
    if swap:
        fig, axes = _figure(_width(len(labels), 1.0, 2.0) + 2.4 * len(swap) + 3, 5, ncols=3,
                            width_ratios=[len(swap) + 1, len(swap) + 1, len(labels) + 1])
        _swap_panels(list(axes[:2]), swap)
        kg_ax = axes[2]
        fig.suptitle("Checker: swap-validated verdicts (left), known-grounded set (right)",
                     x=0.01, ha="left", fontsize=11)
    else:
        fig, kg_ax = _figure(_width(len(labels)))
        fig.suptitle("Checker: known-grounded set", x=0.01, ha="left", fontsize=11)
    kg = [metrics_by_run[lab]["known_grounded"] for lab in labels]
    _verdict_stack(kg_ax, labels, [k["verdicts"] for k in kg])
    recall = [(i, k["l_recall_vs_gold"]) for i, k in enumerate(kg)
              if k["l_recall_vs_gold"] is not None]
    if recall:
        kg_ax.plot(*zip(*recall, strict=True), "D", color="black", markersize=6,
                   linestyle="none", label="mean L recall vs gold")
    kg_ax.axhline(0.7, color=INK, linestyle=":", linewidth=1, label="L recall kill line (0.7)")
    kg_ax.set_ylabel("verdict share / mean L recall")
    kg_ax.set_title("Known-grounded set\n(decorative here is a false alarm)", loc="left",
                    fontsize=9)
    handles, names = kg_ax.get_legend_handles_labels()
    kg_ax.legend(handles[::-1], names[::-1], fontsize=7, loc="upper left",
                 bbox_to_anchor=(1.01, 1), frameon=False)
    return _save(fig, out_path)


def headline(metrics_by_model: Metrics, out_path: str | Path,
             thresholds: dict[str, dict[str, float]] | None = None) -> Path | None:
    """The headline result alone: model -> `evaluate_swap` output.

    `thresholds` uses the `check_thresholds` format and draws the committed lines.
    """
    if not metrics_by_model:
        return None
    fig, axes = _figure(max(7.0, 3.0 + 2.4 * len(metrics_by_model)), 4.8, ncols=2, sharey=True)
    _swap_panels(list(axes), metrics_by_model, thresholds)
    axes[1].set_ylabel("")
    fig.suptitle("Can removal probes tell decorative citations from grounded ones?",
                 x=0.01, ha="left", fontsize=11)
    return _save(fig, out_path)


# --- 3. parametric -----------------------------------------------------------


def parametric(metrics_by_run: Metrics, out_path: str | Path) -> Path | None:
    """Bar: parametric rate (N answers the same without context), Wilson error bars."""
    runs = {lab: s["parametric_rate"] for lab, s in metrics_by_run.items()
            if s["parametric_rate"]["rate"] is not None}
    if not runs:
        return None
    labels = list(runs)
    fig, ax = _figure(_width(len(labels)))
    for i, lab in enumerate(labels):
        r = runs[lab]
        ax.bar(i, r["rate"], 0.6, color=SERIES[0])
        ax.errorbar(i, r["rate"], yerr=_err(r), color="black", capsize=4, linewidth=1)
        ax.annotate(f"{r['rate']:.0%}", (i, r["hi"]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(labels)), [f"{lab}\nn={runs[lab]['n']}" for lab in labels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("parametric rate (Wilson 95% interval)")
    ax.set_xlabel("run (model/dataset)")
    ax.set_title("Answered from memory: N probe matches A", loc="left", fontsize=11)
    return _save(fig, out_path)


# --- 4. unanswerable ---------------------------------------------------------


def unanswerable(metrics_by_run: Metrics, out_path: str | Path) -> Path | None:
    """Grouped bar over unanswerable records: abstains, answers and fabricates, decorative."""
    runs = {lab: s["unanswerable"] for lab, s in metrics_by_run.items()
            if s.get("unanswerable") and s["unanswerable"]["n"]}
    if not runs:
        return None
    labels = list(runs)
    series = [
        ("abstains", lambda u: u["abstain"]["k"], SERIES[0]),
        ("answers, fabricates flag", lambda u: u["non_abstain"]["flags"]["fabricates"]["true"],
         SERIES[6]),
        ("answers, verdict decorative",
         lambda u: u["non_abstain"]["verdicts"][DECORATIVE], SERIES[1]),
    ]
    fig, ax = _figure(_width(len(labels), 1.6))
    w = 0.26
    for j, (name, count, color) in enumerate(series):
        xs = [i + (j - 1) * w for i in range(len(labels))]
        ys = [count(runs[lab]) / runs[lab]["n"] for lab in labels]
        ax.bar(xs, ys, w * 0.92, color=color, label=name)
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(f"{y:.0%}", (x, y), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6, color=INK)
    ax.set_xticks(range(len(labels)), [f"{lab}\nn={runs[lab]['n']}" for lab in labels])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of unanswerable records")
    ax.set_xlabel("run (model/dataset)")
    ax.set_title("Unanswerable questions", loc="left", fontsize=11)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    return _save(fig, out_path)


# --- 5. k sweep --------------------------------------------------------------


def k_sweep(metrics_by_run: Metrics, out_path: str | Path) -> Path | None:
    """Line: verdict agreement with the reference k, per run that has a k sweep."""
    runs = {lab: s["k_sweep"] for lab, s in metrics_by_run.items() if s.get("k_sweep")}
    if not runs:
        return None
    fig, ax = _figure(7.0)
    all_ks: set[int] = set()
    for i, (lab, ks) in enumerate(runs.items()):
        all_ks.update(ks["ks"])
        xs = [k + 0.06 * (i - (len(runs) - 1) / 2) for k in ks["ks"]]  # dodge error bars
        rs = [ks["agreement"][str(k)] for k in ks["ks"]]
        ys = [r["rate"] for r in rs]
        err = [[r["rate"] - r["lo"] for r in rs], [r["hi"] - r["rate"] for r in rs]]
        ax.errorbar(xs, ys, yerr=err, marker="o", markersize=6, linewidth=2, capsize=3,
                    color=SERIES[i % len(SERIES)], label=f"{lab} (n={ks['n']})")
    ax.axhline(0.85, color=INK, linestyle="--", linewidth=1)
    ax.annotate("kill line: >15% of verdicts change", (0.01, 0.85), xycoords=("axes fraction",
                "data"), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xticks(sorted(all_ks))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("samples per probe (k)")
    ax.set_ylabel("verdict agreement with largest k")
    ax.set_title("Sampling stability of verdicts", loc="left", fontsize=11)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    return _save(fig, out_path)


def all_charts(metrics_by_run: Metrics, out_dir: str | Path) -> list[Path]:
    """Write every chart that has data into out_dir. Headline uses runs with `swap`."""
    out = Path(out_dir)
    written = [
        verdicts(metrics_by_run, out / "verdicts.png"),
        checker(metrics_by_run, out / "checker.png"),
        parametric(metrics_by_run, out / "parametric.png"),
        unanswerable(metrics_by_run, out / "unanswerable.png"),
        k_sweep(metrics_by_run, out / "k_sweep.png"),
        headline({lab: s["swap"] for lab, s in metrics_by_run.items() if s.get("swap")},
                 out / "headline.png"),
    ]
    return [p for p in written if p is not None]
