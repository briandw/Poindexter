# P1: core library

Read `PLAN.md` first. It is the spec. This brief fixes interfaces and order of work.

## Deliverable

A Python package `poindexter` with a CLI. Runs offline under test with a fake backend. Runs against the Anthropic API for real.

Definition of done: `uv run pytest` passes with no network; `poindexter run examples/toy.jsonl --backend fake --out /tmp/r.jsonl` produces verdicts; `poindexter report /tmp/r.jsonl` prints the aggregate table; `ruff check` is clean.

## Layout

```
pyproject.toml
poindexter/
  __init__.py
  contract.py     # Record, Answer, Unit dataclasses; validate(); normalize()
  prompt.py       # build_prompt(units, question) -> (system, user)
  probes.py       # pure functions, one per probe
  backend.py      # Backend protocol; ClaudeCLIBackend; FakeBackend
  cache.py        # sqlite cache keyed by (model, temperature, system, user, sample_index)
  runner.py       # run_record(); run_file(); majority; outcomes
  verdict.py      # verdict(); flags(); alignment()
  report.py       # aggregate(); markdown table
  cli.py          # run, report
tests/
examples/toy.jsonl
```

Keep it to these files. Do not add a config system, plugin system, or abstract base classes beyond the one `Backend` protocol.

## contract.py

Dataclasses: `Unit(id, text)`, `Answer(text: str | None, cites: list[str], abstain: bool)`, `Record(id, question, units, answer: Answer | None, gold: list[str] | None)`. JSONL in, JSONL out. Unknown keys on input records are an error, not ignored.

`validate(raw: str, unit_ids: set[str]) -> Answer | Rejection` where `Rejection(code, message)` and `code` is one of `INVALID_JSON`, `BAD_KEYS`, `BAD_ABSTAIN`, `BAD_ANSWER`, `BAD_CITES`. Rules are in PLAN.md. JSON extraction accepts the whole response as one object, or one object inside a single ```json fence with nothing else outside it. Nothing more lenient.

`normalize(s: str) -> str`: the SQuAD v2 official normalizer. Lowercase, remove punctuation, remove articles (a, an, the), collapse whitespace.

`span_in_cites(answer, units) -> bool`: `normalize(answer.text) in normalize(" ".join(cited texts))`.

## prompt.py

One system prompt and one user template. The system prompt states the output schema verbatim and the three rules from PLAN.md (answer only from context, cite every unit relied on, abstain with null answer and empty cites when the context doesn't contain it). The user message lists units as `[u1] text` lines, then `Question: ...`.

The retry prompt appends one line: `Your previous response was rejected: <code>: <message>. Respond with only the JSON object.`

No few-shot examples. If compliance is low on the smoke run, the fix is a prompt change reviewed by Brian, not a more lenient validator.

## probes.py

All pure. Signature `(units: list[Unit], cites: list[str], seed: int, donor: list[Unit] | None = None) -> list[Unit]`.

- `original` returns units.
- `no_context` returns `[]`.
- `remove_cited` returns units with cited ids dropped.
- `replace_cited` returns units with each cited unit's text replaced by the text of a unit drawn from `donor` with `random.Random(seed)`. Id is preserved. Donor is the units of a different record, supplied by the runner. Raise if donor is None or empty.
- `cited_only` returns cited units in original order.
- `leave_one_out(units, i)` returns units minus index i. Separate signature; the runner loops.
- `shuffle` returns a permutation from `random.Random(seed)`.

Order of units is preserved except in `shuffle`.

## backend.py

```python
class Backend(Protocol):
    model: str
    async def complete(self, system: str, user: str, temperature: float | None, sample_index: int) -> str: ...
```

`ClaudeCLIBackend(model)`: runs `claude -p --model MODEL --system-prompt SYSTEM --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence --settings '{"alwaysThinkingEnabled":false}' --output-format json USER` with `asyncio.create_subprocess_exec` (no shell) and returns the `result` field. Retry with exponential backoff (up to 5 tries) when the JSON reports `is_error` for a rate limit or overload; raise on anything else. It rejects a non-None temperature because the CLI cannot set one. `sample_index` is not sent and exists so the cache can hold k distinct samples.

`FakeBackend`: deterministic function of the prompt text. Answers with the first unit whose text contains the question's last word, cites it; abstains if none. This is enough to exercise every probe and verdict path offline. Tests may also construct a `ScriptedBackend` that returns canned responses by prompt substring, but keep it in `tests/`, not the package.

## cache.py

sqlite, stdlib only, at `.poindexter/cache.sqlite` under the working directory. Key is sha256 of `model | temperature | system | user | sample_index`. Value is the raw response string. The runner checks the cache before every call and writes after. No expiry.

## runner.py

`run_record(record, backend, k, temperature, seed, donor_units, semaphore) -> Result`.

Steps:
1. If `record.answer` is None: run O with k samples, validate each, majority over normalized answer text, A is the first validated sample whose normalized text equals the majority. If no valid sample or no majority, the result is `non_compliant` or `unstable_original` and stops here.
2. If `record.answer` is supplied: run O anyway, record `reproduced` = (O majority normalized == A normalized). A stays the supplied answer.
3. Build contexts for N, R_remove, R_replace, M, S, and L_i for every i. Run each k times through the cache. Validate each sample. A rejected sample gets one retry; a second rejection counts as `other` for outcome purposes and increments a per-probe rejection counter.
4. Outcome per sample: `abstain` if abstain is true, `same` if normalized text == normalized A, else `other`. Majority per probe; no strict majority is `unstable`.
5. Hand outcomes to `verdict.py`.

Result is one JSON object per record, shaped by `contract.Result`: the record fields, `A`, per-probe `{samples: [...raw answers...], outcomes: [...], majority, rejections}`, verdict, flags, alignment, `compliance` block.

Concurrency: one `asyncio.Semaphore(concurrency)` shared across records. Default concurrency 8.

Donor for `replace_cited`: the units of the next record in the file (wrapping). Deterministic.

`run_file(path, backend, k, temperature, seed, concurrency, out_path)` streams results to JSONL as they finish.

## verdict.py

Exactly the tables in PLAN.md. `verdict(outcomes) -> str`, `flags(outcomes, answer, units, reproduced) -> dict`, `alignment(cites, loo_outcomes) -> {precision, recall, load_bearing}`. Recall is `None` when nothing is load-bearing. Tests enumerate the table.

## report.py and cli.py

`poindexter run RECORDS.jsonl --model MODEL --backend claude|fake --k 3 --seed 0 --concurrency 8 --out RESULTS.jsonl`

`poindexter report RESULTS.jsonl [--k K]`: aggregate over results. Counts and rates per verdict and flag, compliance, mean alignment precision/recall, and per-probe rejection counts. `--k` recomputes outcomes and verdicts from the first K samples, which is how the k sweep works. Print a markdown table. Charts are P2's job; `report` writes no images.

## Tests

- `test_contract.py`: one test per rejection code, fence handling, normalizer against the SQuAD examples, `span_in_cites`.
- `test_probes.py`: each probe on a five-unit toy record. Assert ids preserved, order preserved except shuffle, donor text used, cited_only order.
- `test_verdict.py`: enumerate the verdict table and every flag.
- `test_runner.py`: end to end on `examples/toy.jsonl` with `FakeBackend`, k=3. Assert verdicts, that the cache hits on a second run (call counter on the fake), and that a supplied answer produces `reproduced`.
- `test_report.py`: aggregate over a hand-built results list; `--k 1` vs `--k 3` on a scripted backend that flips one sample.

No mocking library. No network. Tests must run under two seconds.

## Rules

- Fail fast on invariant violations (unknown record keys, missing donor, unit id collisions). No fallback paths for states the contract excludes.
- Handle real boundaries: API errors, malformed model output, cache I/O.
- Do not implement anything from PLAN.md's "Later" section.
- Do not touch anything under `briefs/` or `PLAN.md`.
- Work in your own jj workspace. Push one bookmark and open the PR on GitHub with `Closes #N` for the P1 issue. CI is `uv run pytest` and `ruff check`. Report: what changed, what was verified, what is unresolved.
