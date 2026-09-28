# Review brief (independent reviewer, fresh context)

You are reviewing the P1 pull request against `PLAN.md` and `briefs/P1-core.md`. You did not write it. Read the spec first, then the diff. Report findings as: what is wrong, why it matters, what to do instead. Rank by severity. Do not restyle code or propose features.

## Must verify

1. Probes are pure and match the table in PLAN.md. Unit ids survive every probe. Order survives everything except `shuffle`. `replace_cited` uses donor text and keeps the id. `cited_only` preserves original order.
2. Validator is exactly as strict as the spec. Every rejection code has a test that triggers it and only it. JSON extraction accepts a bare object or one fenced object and nothing else. No silent coercion of `abstain`, no trimming of cites.
3. Cache key includes model, temperature, system prompt, user prompt, and sample index. Changing any one produces a miss. Second run hits.
4. Majority and tie handling: no strict majority yields `unstable`, and `unstable` in R_remove or M yields verdict `unstable`. Check the k=2 tie case.
5. Verdict table and every flag match PLAN.md. The tests enumerate all rows.
6. A supplied answer is used as A, O still runs, and `reproduced` is set. A generated answer takes cites from the first sample that matches the majority.
7. Rejected samples: one retry with the code appended, then counted as `other` and tallied. Confirm the retry prompt is what the spec says.
8. Tests run with no network and no mocking library. Fake backend lives in the package, scripted backend in tests.
9. Error handling exists only at real boundaries (API, model output, cache I/O). Flag any defensive branch guarding a state the contract already excludes, and any invariant violation that is swallowed instead of raised.
10. No config files, no plugin machinery, no abstract classes beyond the `Backend` protocol, nothing from PLAN.md's "Later" section.

## Also check

- Concurrency: one semaphore, results streamed as they finish, no unbounded task fan-out.
- Backoff only on rate-limit and overload errors from the CLI, capped, and other errors propagate. Subprocess runs without a shell.
- `report --k K` recomputes from the first K samples rather than re-running.
- The diff touches only files the brief lists. `briefs/` and `PLAN.md` are untouched.

## Output

A ranked list. For each item: file and line, what is wrong, why it matters, the fix. End with a one-line verdict: merge, merge after fixes, or do not merge.
