"""Command line: `poindexter run` and `poindexter report`.

Each subcommand is one `add_<name>(subparsers)` function that registers its parser and
sets `func`. Adding a subcommand is one such function plus an entry in SUBCOMMANDS.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from poindexter import runner
from poindexter.backend import make_backend
from poindexter.contract import read_results
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
    p.add_argument("--out", required=True, help="results JSONL")
    p.set_defaults(func=cmd_run)


def cmd_run(args: argparse.Namespace) -> None:
    backend = make_backend(args.backend, args.model)
    results = asyncio.run(
        runner.run_file(
            args.records, backend, args.k, args.temperature, args.seed,
            args.concurrency, args.out, args.probes,
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


SUBCOMMANDS = [add_run, add_report]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="poindexter", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for add in SUBCOMMANDS:
        add(sub)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
