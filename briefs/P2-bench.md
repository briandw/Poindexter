# P2: datasets, benchmark, charts

Read `PLAN.md` first. This runs in parallel with P1 and depends on P1's `contract.py` (`Record`, `Unit`, `normalize`) and `runner.run_file`. Code against those names as specified in `briefs/P1-core.md`. If P1 isn't merged yet, develop on a branch off P1's branch.

## Deliverable

`poindexter bench` builds gold-labeled records from SQuAD 2.0 and HotpotQA, runs them, evaluates against gold, and writes tables and charts.

Definition of done: `poindexter bench --dataset squad --n 20 --backend fake --out /tmp/b` runs offline end to end (dataset files cached locally after first download; the test uses a checked-in 30-record fixture, not the download); charts render; `uv run pytest` passes; `ruff check` clean.

## Layout

```
poindexter/
  datasets.py    # load_squad(); load_hotpotqa(); sentence_split(); build_records()
  bench.py       # evaluate(results, records) -> metrics dict; markdown tables
  charts.py      # five chart functions, matplotlib, png out
  cli.py         # add `bench`
tests/fixtures/squad_30.json
tests/fixtures/hotpot_30.json
```

## datasets.py

Download once to `.poindexter/data/`, plain `urllib`:

- SQuAD 2.0 dev: `https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json`
- HotpotQA dev distractor: `http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json`

`sentence_split(text) -> list[str]`: deterministic regex split on `[.!?]` followed by whitespace and an uppercase letter or digit, with a short abbreviation list (Mr, Mrs, Dr, St, No, vs, etc, e.g, i.e, U.S). Return offsets too, so answer spans map to sentences. Good enough is the bar; record the split so it can be inspected.

SQuAD records:
- Units are the paragraph's sentences, ids `u1..un` in order.
- `gold` is every sentence overlapping any gold answer's `[answer_start, answer_start + len)`. Unanswerable questions get `gold: []` and `meta.unanswerable: true`.
- `meta.dataset_answers` is the list of gold answer strings (empty for unanswerable).
- Sample `n` with `random.Random(seed)`, stratified so unanswerables are 30% of the sample.

HotpotQA records:
- Units are the 10 paragraphs, one unit each, text is title + joined sentences, ids `u1..u10` in given order.
- `gold` is the paragraph ids whose title appears in `supporting_facts`.
- `meta.dataset_answers` is `[answer]`. `meta.type` is the HotpotQA type field (bridge or comparison).
- Sample `n` with `random.Random(seed)`.

`meta` is a new optional dict field on `Record`. Coordinate with P1: add it to `contract.py` if P1 hasn't, with unknown-key strictness preserved for everything else.

## bench.py

`evaluate(results, records) -> dict` computes, per dataset:

1. `compliance`: from the runner.
2. `parametric_rate`: fraction of records with flag `parametric`.
3. `correct`: `normalize(A.text)` in normalized `dataset_answers`. Per record.
4. `known_grounded` set: not parametric, correct, `set(gold) <= set(cites)`, gold non-empty. Report its size, verdict distribution, and mean L recall against gold (fraction of gold ids in `load_bearing`).
5. `citation_precision`, `citation_recall`: cites vs gold, mean over answerable records.
6. `unanswerable`: abstain rate; for non-abstentions, verdict and flag distribution.
7. `k_sweep`: verdict agreement between k=1, k=3, k=5 recomputed from the stored samples (needs a run with k=5; agreement is measured against k=5).

Write `metrics.json` and `tables.md`.

## charts.py

Five functions, each `(metrics_by_model_by_dataset, out_path)`. Matplotlib, no seaborn. Plain style, labeled axes, one PNG each:

1. `verdicts`: stacked bar, verdict share per model per dataset.
2. `checker`: known-grounded set verdict share, and L recall vs gold, per model per dataset. If counterfactual metrics exist, a second panel with checker precision/recall.
3. `parametric`: bar, parametric rate per model per dataset.
4. `unanswerable`: grouped bar, abstain vs fabricates vs decorative, per model.
5. `k_sweep`: line, verdict agreement vs k.

## cli.py

`poindexter bench --dataset squad|hotpotqa --n 200 --seed 0 --model MODEL --backend claude|fake --k 3 --out DIR`

Writes `records.jsonl`, `results.jsonl`, `metrics.json`, `tables.md`, and charts under `DIR`. Reruns hit the cache.

`poindexter charts DIR1 DIR2 ...`: combines metrics from several bench dirs (different models or datasets) into the five charts.

## Counterfactual swap

Required. It feeds the headline experiment in PLAN.md's Goal section.

`--counterfactual` on `bench --dataset squad`: keep only answerable records whose first dataset answer is a bare integer (`^\d+$`). In the gold sentence, replace that integer with a different integer of the same digit count drawn from `random.Random(seed)`, never equal to the original, and for four-digit values between 1000 and 2100 keep the result in that range. Set `meta.counterfactual = {"original": "...", "swapped": "..."}` and `meta.dataset_answers = [swapped]`.

Evaluation adds, for these records:
- Constructed truth: `answered_swapped` (normalized A == swapped) means the model read the context, so the citation on the gold sentence is grounded by construction. `answered_original` means it answered from memory, so any citation is decorative by construction. Anything else is excluded.
- Checker precision/recall: Poindexter verdict `grounded` or `incomplete` vs constructed `answered_swapped`; `decorative` vs constructed `answered_original`. Report the 2x2 and both rates.

Only records where the model cites the gold sentence count, so the comparison is about whether the citation was load-bearing, not whether the model found the sentence.

## Tests

- `test_datasets.py`: sentence splitter on a dozen cases including abbreviations and numbers; gold mapping on the fixture; stratified sampling is deterministic under a seed; HotpotQA gold ids match supporting facts.
- `test_bench.py`: `evaluate` on a hand-built results list with known metrics; known-grounded filter; k sweep agreement.
- `test_counterfactual.py` if implemented: swap never equals original, digit count preserved, sentence text updated, gold unchanged.
- Chart functions are smoke-tested: they write a PNG without error on fixture metrics.

No network in tests. Fixtures are 30 records each, checked in.

## Rules

Same as P1: fail fast on invariant violations, handle real boundaries only, nothing from "Later", don't touch `briefs/` or `PLAN.md`, PR on GitHub from your own jj workspace and bookmark, CI green, report changed behavior, verification, and open risks.
