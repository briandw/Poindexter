"""Command line: `poindexter run`, `report`, `bench`, `charts`, and `swap`.

Each subcommand is one `add_<name>(subparsers)` function that registers its parser and
sets `func`. Adding a subcommand is one such function plus an entry in SUBCOMMANDS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from poindexter import bench, charts, datasets, experiment, runner, surprise
from poindexter.backend import make_backend
from poindexter.contract import read_results
from poindexter.prompt import AGENTS
from poindexter.report import aggregate, markdown
from poindexter.verdict import recompute


def add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="audit records and write results JSONL")
    p.add_argument("records", help="records JSONL")
    p.add_argument("--model", help="model id (required for --backend claude)")
    p.add_argument("--backend", choices=["claude", "fake"], default="claude")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=runner.DEFAULT_CONCURRENCY)
    p.add_argument("--probes", choices=runner.PROBE_SETS, default="all")
    p.add_argument("--agent", choices=sorted(AGENTS), default="context",
                   help="audited agent: context-only, or open (may answer from knowledge)")
    p.add_argument("--out", required=True, help="results JSONL")
    p.set_defaults(func=cmd_run)


def cmd_run(args: argparse.Namespace) -> None:
    backend = make_backend(args.backend, args.model)
    results = asyncio.run(
        runner.run_file(
            args.records, backend, args.k, args.temperature, args.seed,
            args.concurrency, args.out, args.probes, args.agent,
        )
    )  # fmt: skip
    status = Counter(r.status for r in results)
    verdicts = Counter(r.verdict for r in results if r.verdict is not None)
    print(f"{len(results)} records -> {args.out}; status {dict(status)}; verdicts {dict(verdicts)}")


def add_report(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("report", help="aggregate a results JSONL into markdown tables")
    p.add_argument("results", help="results JSONL")
    p.add_argument("--k", type=int, default=None, help="recompute from the first K samples")
    p.set_defaults(func=cmd_report)


def cmd_report(args: argparse.Namespace) -> None:
    results = read_results(args.results)
    if args.k is not None:
        results = [recompute(r, args.k) for r in results]
    print(markdown(aggregate(results)), end="")


def add_bench(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("bench", help="build gold-labeled records, run them, score against gold")
    p.add_argument("--dataset", choices=["squad", "hotpotqa"], required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", help="model id (required for --backend claude)")
    p.add_argument("--backend", choices=["claude", "fake"], default="claude")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=runner.DEFAULT_CONCURRENCY)
    p.add_argument("--probes", choices=runner.PROBE_SETS, default="all")
    p.add_argument("--source", help="dataset JSON file to use instead of the download")
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_bench)


def cmd_bench(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = datasets.build_records(
        args.dataset, out / "records.jsonl", n=args.n, seed=args.seed, source=args.source
    )
    backend = make_backend(args.backend, args.model)
    results = asyncio.run(
        runner.run_records(
            records, backend, args.k, None, args.seed, args.concurrency,
            out / "results.jsonl", args.probes,
        )
    )  # fmt: skip
    metrics = bench.evaluate(results, recompute if args.k >= 5 else None)
    bench.write_metrics(metrics, out)
    charts.all_charts({f"{out.name}/{d}": m for d, m in metrics.items()}, out)
    print(f"{len(results)} records -> {out}")


def add_charts(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("charts", help="combine metrics.json from bench or swap dirs into charts")
    p.add_argument("dirs", nargs="+", help="bench or swap output directories")
    p.add_argument("--out", required=True, help="directory for the PNGs")
    p.set_defaults(func=cmd_charts)


def cmd_charts(args: argparse.Namespace) -> None:
    by_run = {}
    for d in map(Path, args.dirs):
        for dataset, section in json.loads((d / "metrics.json").read_text()).items():
            by_run[f"{d.name}/{dataset}"] = section
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for path in charts.all_charts(by_run, args.out):
        print(path)


def add_swap(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("swap", help="the headline experiment: swap-validated verdicts")
    p.add_argument("--model", help="model id (required for --backend claude)")
    p.add_argument("--backend", choices=["claude", "fake"], default="claude")
    p.add_argument("--splits", default="dev,train", help="comma-separated SQuAD splits")
    p.add_argument("--candidates", type=int, default=1000)
    p.add_argument("--unscreened", type=int, default=200)
    p.add_argument("--grounded", type=int, default=150)
    p.add_argument("--redundant", type=int, default=60)
    p.add_argument("--sweep", type=int, default=50)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=runner.DEFAULT_CONCURRENCY)
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_swap)


def cmd_swap(args: argparse.Namespace) -> None:
    section = asyncio.run(
        experiment.swap_experiment(
            make_backend(args.backend, args.model), args.out,
            splits=tuple(args.splits.split(",")), n_candidates=args.candidates,
            n_unscreened=args.unscreened, grounded_n=args.grounded,
            redundant_n=args.redundant, sweep_n=args.sweep, k=args.k, seed=args.seed,
            concurrency=args.concurrency,
        )
    )  # fmt: skip
    print(json.dumps(section["outcome"], indent=2))


def add_explore(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("explore", help="exploratory: audit a knowledge-permitted agent")
    p.add_argument("swap_dir", help="a `poindexter swap` output directory (for its screen)")
    p.add_argument("--model", help="model id (required for --backend claude)")
    p.add_argument("--backend", choices=["claude", "fake"], default="claude")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=runner.DEFAULT_CONCURRENCY)
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_explore)


def cmd_explore(args: argparse.Namespace) -> None:
    report = asyncio.run(
        experiment.explore_open_agent(
            make_backend(args.backend, args.model), args.swap_dir, args.out, k=args.k,
            seed=args.seed, concurrency=args.concurrency,
        )
    )  # fmt: skip
    print(json.dumps(report, indent=2))


def add_surprise(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("surprise", help="v2: surprise-validated verdicts (PLAN-v2.md)")
    p.add_argument("corpus", help="corpus directory (L0/L1/L2/novel .jsonl)")
    p.add_argument("--model", help="model id (required for --backend claude)")
    p.add_argument("--backend", choices=["claude", "fake"], default="claude")
    p.add_argument("--facts", type=int, default=150)
    p.add_argument("--novel", type=int, default=50)
    p.add_argument("--agents", default="context,open")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=runner.DEFAULT_CONCURRENCY)
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_surprise)


def cmd_surprise(args: argparse.Namespace) -> None:
    report = asyncio.run(
        surprise.surprise_experiment(
            make_backend(args.backend, args.model), args.corpus, args.out,
            n_facts=args.facts, n_novel=args.novel, k=args.k, seed=args.seed,
            concurrency=args.concurrency, agents=tuple(args.agents.split(",")),
        )
    )  # fmt: skip
    print(json.dumps(report, indent=2))


SUBCOMMANDS = [add_run, add_report, add_bench, add_charts, add_swap, add_explore, add_surprise]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="poindexter", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for add in SUBCOMMANDS:
        add(sub)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
