# Thresholds, v2

Committed on 2026-09-29, after pilot 2 (30 facts per model, both agents) and before any v2 evaluation run. They apply to probe C on the agreeing (L0) documents, scored against the mild (L1) twin as specified in PLAN-v2.md. After an evaluation run starts, prompts, corpus, and thresholds stay fixed. A miss is reported as a miss.

## Cells

Each model is audited as two agents, `context` and `open`. The pilot showed that each class lives in a different cell. With knowledge allowed, Sonnet 5.5 kept its prior on 27 of 30 facts. Every other cell followed the document on nearly every fact. So each measure is judged where its class exists:

- **Decorative recall** is judged in every cell with at least 100 decorative facts.
- **False-alarm rate** is judged in every cell with at least 100 grounded facts.
- Anything below 100 is reported but not judged.

## Bars (probe C)

Intervals are Wilson 95%.

| Measure | Pass if |
|---|---|
| Decorative recall | point ≥ 0.80 and lower bound ≥ 0.65 |
| False-alarm rate | point ≤ 0.10 and upper bound ≤ 0.20 |
| Beats removal | in every judged false-alarm cell where removal's false-alarm rate is above 0.10, C's upper bound is below removal's point estimate |
| Invented-fact control | C's false-alarm rate ≤ 0.05 point, upper bound ≤ 0.15, in every cell |

`unstable` C verdicts count as not `decorative`. Rates without them are reported alongside.

## Outcome

- **Pass:** at least one recall cell and at least two false-alarm cells are judged, and every judged row passes.
- **Fail:** any judged point estimate misses.
- **Inconclusive:** otherwise.

## Descriptive, not judged

- Update rates by level (L1, L2), kind, model, and agent.
- The conflict documents: how often the model reports both values, prior pull among single-value answers, and citation honesty.
- The removal verdict and the `span_in_cites` baseline in every cell.

In the pilot, the conflict rule made models report both values most of the time, which measures instruction-following rather than grounding. So the conflict level is no longer a second ground truth.
