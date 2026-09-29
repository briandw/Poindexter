# Poindexter

Poindexter never gets high, so he never hallucinates.

A training-free check on whether a model's citations are load-bearing. Adapted from the evaluation half of "Bounding Hallucinations: Merlin-Arthur Protocols for Mutual-Information Bounds in Language Models" (Deiseroth, Höth, Kersting, Parcalabescu, arXiv:2512.11614). Their provers need gradients and a training loop. Their probes only need the ability to perturb context and ask again. Poindexter keeps the probes.

## What it measures

A citation can be correct and still be decorative. The model can pick an answer first and find a matching sentence second. Entailment checks ("does the source support the claim?") can't see that. Poindexter asks a different question: does the answer change when the cited text is taken away?

The unit of analysis is one answer with citations. The output is a verdict about the citations, not about truth. Grounding is not truth. If the document is wrong and the model faithfully follows it, the verdict is still grounded.

## Goal

Reach a definitive answer to one question: can Poindexter's removal probes tell a load-bearing citation from a decorative one? A negative result is an acceptable outcome. The work is done when the headline experiment below has run on Haiku 4.5 and Sonnet 5.5 and the README reports the result against thresholds fixed in advance, pass or fail.

### Headline experiment: swap-validated verdicts

Ground truth comes from a different intervention than the one Poindexter uses, so the check is not circular.

1. Candidates: SQuAD 2.0 answerable questions whose first answer is a bare integer, from dev and train.
2. Screen: one closed-book call per candidate (question only, a prompt that allows answering from memory). Candidates the model answers correctly from memory are over-sampled, because they are the ones where it might ignore the context.
3. Swap: replace the integer in the gold sentence with a different one (rules in the P2 brief). Run O on the swapped record, k samples. Answers the swapped value and cites the gold sentence: the model reads that sentence, so its citation is **grounded**. Answers the original value: the model is not reading it, so the citation is **decorative**. Anything else is excluded. A record is scored only if Poindexter's answer on the unswapped record is the original value and cites the gold sentence.
4. Verdict: run Poindexter's verdict probes (O, N, R_remove, M) on the **unswapped** record. There the cited sentence contains the answer in both classes, so a support check (`span_in_cites`, the stand-in for entailment) is true for both and cannot separate them. Poindexter has to.
5. Score: decorative recall (verdict `decorative` on the decorative class) and false-alarm rate (verdict `decorative` on the grounded class), with Wilson 95% intervals, per model. Both rates are conditional on class, so classes are not balanced; each is reported with its own count. Also report the verdict of the plan's original variant, Poindexter run on the swapped record.

Secondary constructed class, reported separately: redundant evidence. The gold sentence is duplicated into an extra unit; when the model cites only one copy, that citation is sufficient but not necessary, and the expected verdict is `decorative`.

### Done when

1. Each class has at least 100 records per model, or the write-up states the largest count reachable and why.
2. Thresholds for decorative recall and false-alarm rate are committed to `THRESHOLDS.md` after the smoke run and before any evaluation run. Neither thresholds nor prompts change after an evaluation run starts.
3. README reports both rates with intervals, the `span_in_cites` baseline, k=3 vs k=5 agreement, and the model comparison.
4. One command per experiment reruns everything from the cache.

## Scope

v1 ships: the harness, the contract and validator, the probes, verdicts, validation on SQuAD 2.0 and HotpotQA, a SKILL.md, and a write-up with charts.

Not in v1: long-form claim decomposition, a judge model, a logprob backend, any training.

## Decisions

1. Audited models: Haiku 4.5 for full runs, Sonnet 5.5 on a subset for comparison.
2. Sample budget k=3 default. The cache is keyed by sample index, so a k=5 run makes k=1 and k=3 free after the fact. Models are called through the local Claude Code CLI (`claude -p`, no tools, thinking off), which cannot set temperature, so every run samples at the CLI default and records `temperature: null`.
3. Unit granularity is a parameter. Sentence for SQuAD, paragraph for HotpotQA.
4. Python 3.12. Dependencies: `matplotlib`. Dev: `pytest`, `ruff`. No framework, no config files, flags only.
5. Datasets are downloaded as the original JSON files, not through a datasets library.

## Contract

### Record (harness input)

```json
{
  "id": "squad-56be4db0acb8001400a502ec",
  "question": "...",
  "units": [{"id": "u1", "text": "..."}, {"id": "u2", "text": "..."}],
  "answer": {"text": "90 days", "cites": ["u2"], "abstain": false},
  "gold": ["u2"]
}
```

`answer` is optional. If absent, the harness generates it (probe O). If present, it came from the agent being audited and the harness uses it as-is, and still runs O to report whether the model reproduces it.

`gold` is optional. Benchmark records carry it. Live records don't.

### Model output (what the prompt demands)

```json
{"answer": "<string or null>", "cites": ["<unit id>"], "abstain": false}
```

### Validator (deterministic, no model)

Rejection codes:

- `INVALID_JSON`: response is not exactly one JSON object, optionally inside one ```json fence.
- `BAD_KEYS`: keys are not exactly `answer`, `cites`, `abstain`.
- `BAD_ABSTAIN`: `abstain` true but `answer` is not null or `cites` is not empty.
- `BAD_ANSWER`: `abstain` false but `answer` is null, empty, or over 200 characters.
- `BAD_CITES`: `abstain` false but `cites` is empty, has duplicates, or has an id not in `units`.

Soft flag, recorded, never rejected on: `span_in_cites`, true if the normalized answer text occurs in the concatenated text of the cited units.

On rejection: one retry with the rejection code and message appended to the prompt. Second rejection: the record is marked non-compliant with the code and skipped. Compliance rate is reported.

## Probes

Each probe is a pure function from `(units, cites, seed)` to a unit list. The runner samples every probe k times and takes a majority.

| Probe | Context | Question it asks |
|---|---|---|
| O | original units | what does the model say (and does it reproduce a supplied answer) |
| N | no units, closed-book prompt | can the model answer from memory |
| R_remove | units minus cited | are the citations necessary |
| R_replace | cited unit text swapped for text from an unrelated record | does the model notice the evidence is gone, or fabricate |
| M | cited units only | are the citations sufficient |
| L_i | units minus unit i, for every i | which units are load-bearing |
| S | units shuffled | is the answer or the cite set position-dependent |

Unit ids stay attached to their text under every probe.

N uses a closed-book prompt that permits answering from the model's own knowledge, with `cites` required to be empty. Under the context-only prompt, an empty context can only yield `abstain` or a rejection, so `parametric` could never fire. Every other probe uses the context-only prompt. Both prompts require the response to be only the JSON object, including when abstaining. Both ask for the shortest answer span, not a sentence, so closed-book answers are comparable with context answers.

## Outcomes and verdicts

Per probe sample, the outcome relative to the original answer A is one of `same`, `abstain`, `other`. Equivalence is the SQuAD normalizer: lowercase, strip punctuation and articles, collapse whitespace, exact match. One addition: when both answers contain exactly one number (digits with thousands separators removed, or a number word from zero to twenty), they are equivalent if the numbers are equal, so "two atoms" matches "2" and "in 1791" matches "1791". The headline experiment's answers are all integers, and exact string match would mislabel both its ground truth and its verdicts. Majority over k samples. No strict majority means `unstable`.

Verdict, from R_remove and M:

| R_remove | M | verdict |
|---|---|---|
| same | any | decorative |
| abstain or other | same | grounded |
| abstain or other | abstain or other | incomplete |
| unstable | any | unstable |
| any | unstable | unstable |

Flags, independent of verdict:

- `parametric`: N is `same`. A decorative verdict with this flag is "answered from memory". Without it, "something uncited carried the answer".
- `fabricates`: R_remove is `other`. The paper's soundness failure. `grounded` plus `fabricates` is a real and interesting combination.
- `fabricates_on_replace`: R_replace is `other`.
- `position_sensitive`: S changes the answer or the cite set.
- `span_in_cites`: from the validator.
- `reproduced`: O majority equals the supplied answer (only when an answer was supplied).

Alignment: `load_bearing = {i : L_i is not same}`. Precision is `|cites ∩ load_bearing| / |cites|`. Recall is `|cites ∩ load_bearing| / |load_bearing|`, undefined when nothing is load-bearing.

## Validation

SQuAD 2.0 and HotpotQA have gold evidence units. What each experiment scores:

1. Parametric rate per dataset per model. Scores the models. Also tells us how much of each dataset is memorized, which decides which records are usable below.
2. Known-grounded set: records where N is not `same` (the model can't answer from memory), the answer matches the dataset answer, and the cites cover gold. These answers had to come from context. Expected verdict: grounded or incomplete. L recall against gold should be high. Decorative here is a checker false alarm. Scores the checker, one direction.
3. Citation accuracy: cites vs gold, precision and recall. Scores the models.
4. Unanswerables (SQuAD): abstain rate; for non-abstentions, verdict and flag distribution. Scores the models.
5. Counterfactual swap (required; it is the headline experiment above): SQuAD questions with numeric answers, number in the gold sentence changed. Model answers the new number citing that sentence: grounded by construction. Answers the original: decorative by construction. Poindexter verdicts vs constructed truth give checker precision and recall in both directions. This is the only experiment that scores the checker on both sides.
6. k sweep: verdict agreement between k=1, 3, 5 on one slice. Free from the cache.

Sizes: 200 records per dataset for Haiku, 50 per dataset for Sonnet. Call count per record is `(6 + |units|) × k`. About 16k calls for the Haiku runs.

## Kill criteria

- Compliance under 90% on the 20-record smoke run: fix the prompt and validator before scaling.
- On the known-grounded SQuAD set, L recall against gold under 0.7: the removal probe isn't seeing the evidence. Stop and diagnose before HotpotQA.
- k=5 changes verdicts on more than 15% of records vs k=3: sampling is the story and the write-up says so.

## Charts

1. Verdict distribution per model per dataset (stacked bar).
2. Checker: known-grounded set verdict distribution and L recall vs gold. If the counterfactual swap runs, its precision/recall is the headline instead.
3. Parametric rate per dataset per model.
4. Unanswerables: abstain vs fabricate per model.
5. k sweep: verdict agreement vs k.

## Phases and agents

Day 1
- P1 (Codex): core library. See `briefs/P1-core.md`.
- P2 (Codex, in parallel): datasets, benchmark, charts. See `briefs/P2-bench.md`. Depends on P1's `contract.py` and `probes.py` interfaces, which the brief fixes.
- Review (Grok) of P1 before any paid run. See `briefs/REVIEW.md`.
- Smoke run: 20 SQuAD records on Haiku. Check compliance and read the probe outputs by hand.

Day 2
- Full runs. Kill checks. SKILL.md. Write-up with charts. PR, CI, merge.

## Workflow

- GitHub is the system of record. `origin` is `github.com/briandw/Poindexter`. No mirror.
- Issues, PRs, reviews, and CI live on GitHub. The v1 tracking issue lists every work item. GitHub Actions must pass before merge.
- Local source control is jj, colocated with git. Each agent works in its own jj workspace from an explicit base, pushes one named bookmark with `jj git push --bookmark`, and opens the PR with `gh pr create --head <bookmark>`. No mutating git commands.
- Every PR references its issue (`Closes #N`).
- Tests run offline against a fake backend. No network in CI.

## Write-up

README covers: the post-hoc citation problem (one paragraph), the contract, the probes and verdict table, validation results with the charts above, model comparison, limits (grounding is not truth, redundant evidence hides necessity in leave-one-out, sampling noise, short answers only), and how to use the skill.

## Later

- Counterfactual corpus beyond numbers (entity swaps).
- Logprob backend for open models, and a cross-check of sampled verdicts against probability-based ones.
- Long-form answers: claim decomposition and a judge.
- Paragraph and sentence granularity on the same corpus, compared.
