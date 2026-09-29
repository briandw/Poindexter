---
name: poindexter
description: Citation audit with Poindexter, testing whether an answer's citations are load-bearing or decorative by removing the cited text and asking again. Use when checking grounding of a cited or RAG answer, asking whether a model used its sources, or running `poindexter run` / `poindexter report`.
---

Poindexter answers one question per cited answer: does the answer survive when the cited text is taken away? A citation the answer needs is **load-bearing**; one it survives without is **decorative**. The probes perturb the context and re-ask an audit model k times; majorities give the verdict. Run everything from the Poindexter checkout with `uv run poindexter` (or `uv run --project <checkout> poindexter` from elsewhere); `--help` on each subcommand is the flag reference.

## Steps

### 1. Build records

One JSON object per line. Split the source into **units** (sentences for short passages, paragraphs for long ones) with unique ids; the question must have a short extractive answer.

Let the audit model answer (probe O generates the answer):

```json
{"id": "q1", "question": "How long is the warranty?", "units": [{"id": "u1", "text": "The store opened in 1998."}, {"id": "u2", "text": "The warranty lasts 90 days."}]}
```

Audit an answer another agent gave: add `answer`. `text` is the shortest span (at most 200 characters), `cites` is non-empty and names ids in `units`; a record that breaks this fails at load with the rejection code.

```json
{"id": "q2", "question": "When does the library close?", "units": [{"id": "u1", "text": "The library opens at 9 am."}, {"id": "u2", "text": "Doors close at 6 pm."}], "answer": {"text": "6 pm", "cites": ["u2"], "abstain": false}}
```

Done when `uv run poindexter run records.jsonl --backend fake --probes verdict --out /tmp/check.jsonl` loads every record (the fake backend makes no model calls; its verdicts mean nothing).

### 2. Run the probes

```sh
uv run poindexter run records.jsonl --backend claude --model haiku --probes verdict --out results.jsonl
```

- `--probes verdict` runs O, N, R_remove, M: 4×k calls per record, enough for the verdict and the `parametric`, `fabricates`, `span_in_cites`, `reproduced` flags.
- `--probes all` adds R_replace, S, and leave-one-out per unit: (6 + units)×k calls. Use it when you need alignment or the `fabricates_on_replace` / `position_sensitive` flags. It needs 2+ records (R_replace borrows text from the next record).
- `--backend claude` calls the local `claude` CLI; `--model` is required, `--temperature` stays unset (the CLI cannot set it).
- Responses are cached in `.poindexter/cache.sqlite` under the working directory (or `$POINDEXTER_CACHE`), so a rerun or a higher `--k` pays only for new samples.

Done when the summary line prints and every record has a line in the output.

### 3. Read the results

`uv run poindexter report results.jsonl` gives aggregate tables (status, verdicts, flags, compliance, alignment). `--k N` recomputes from the first N stored samples, with no calls. Per record:

```sh
jq -c '{id: .record.id, status, verdict, A, flags, alignment, probes: (.probes | map_values(.majority))}' results.jsonl
```

Done when every record has a verdict or a status that explains its absence, and each verdict is read with its flags using the reference below.

## Reading results

Status other than `ok` means no verdict:

- `non_compliant`: the model broke the JSON contract twice on O; `compliance.code` says how.
- `unstable_original`: O samples disagree; raise `--k` or suspect an ambiguous question.
- `ok` with `A.abstain` true: the model abstained, so there are no citations to judge. Check the context yourself before concluding the answer isn't there.

Verdicts (from R_remove = cited units removed, M = cited units only):

- `grounded`: removing the cites breaks the answer and the cites alone suffice. The citations are load-bearing.
- `decorative`: the answer survives without the cites. Look for uncited units that carry the answer: cite the ids in `alignment.load_bearing` (needs `--probes all`), or, if that list is empty, look for the same fact repeated in several units.
- `incomplete`: the cites are necessary but not sufficient. Add the missing units from `alignment.load_bearing` (needs `--probes all`).
- `unstable`: no strict majority on R_remove or M. Rerun with a higher `--k` before concluding anything.

Flags (null when the probe was not run):

- `parametric`: the model gives the same answer closed-book, with no units. On a `decorative` verdict it suggests memory could have supplied the answer; without it, look for uncited units that carry it. Neither is proof of where this answer came from.
- `fabricates`: with the cites removed, the model gives a different answer instead of abstaining. Distrust this model's answers when retrieval misses.
- `fabricates_on_replace`: same, with the cited text swapped for unrelated text.
- `position_sensitive`: shuffling units changed the answer or cite set. Treat the exact cite set as noise.
- `span_in_cites`: the answer string occurs in the cited text. Decorative citations pass this too, so it never counts as evidence of grounding.
- `reproduced`: the audit model's own answer matches the supplied one. When false, the verdict describes an answer the audit model does not give; report it as weak.

`alignment` (with `--probes all`): `load_bearing` lists units whose removal changes the answer; precision and recall compare it with the cites.

## Limits

- Grounding is not truth: a faithful answer to a wrong document is still grounded.
- Short extractive answers only; long-form answers need claim decomposition first.
- Leave-one-out misses redundant evidence: a fact stated twice is load-bearing in neither copy.
- A verdict says whether this audit model, under Poindexter's prompt, needs the cited text. It says nothing about how an external agent produced its answer.
- Majorities over k samples carry sampling noise; the default k=3 can flip, so confirm verdicts that matter at `--k 5`.

Validation ([README.md#results](../../README.md#results)): 0 of 300 swap-confirmed grounded citations were called decorative, and about 98% of citations made unnecessary by an uncited copy were caught. Recall on citations that are decorative because the model answered from memory is unvalidated. A `grounded` verdict means the cited text was needed, not that the answer came from it.
