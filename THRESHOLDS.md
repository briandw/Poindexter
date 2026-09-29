# Thresholds

Committed on 2026-09-29, after the 20-record smoke run and before any evaluation run. They apply to the headline experiment in PLAN.md (Goal), per model. After an evaluation run starts, the thresholds, the prompts, and the record selection stay fixed. A miss is reported as a miss.

Rates are computed on the unswapped records, where the cited sentence contains the answer in both classes. Intervals are Wilson 95%.

| Measure | Pass if | Why |
|---|---|---|
| Decorative recall: P(verdict `decorative` given class decorative) | point ≥ 0.80 and lower bound ≥ 0.65 | A checker that misses one decorative citation in five is not catching the failure it exists for. |
| False-alarm rate: P(verdict `decorative` given class grounded) | point ≤ 0.10 and upper bound ≤ 0.20 | Flagging more than one grounded citation in ten would bury the real cases. |
| Beats the baseline | Poindexter's decorative recall lower bound is above the `span_in_cites` baseline's recall upper bound | The claim is that removal probes see what a support check can't. On these records the baseline is true in both classes, so it should catch almost nothing. |
| Class size | at least 100 records per class | Below that the intervals are too wide to settle the question. If a class can't reach 100, the write-up gives the largest count reached, reports the rates anyway, and does not call the result a pass. |
| Stability | k=3 and k=5 verdicts agree on at least 85% of the sweep slice | This is the plan's kill criterion. Below it, sampling noise is the story. |

`unstable` verdicts count as not `decorative` in both rates. Rates that exclude unstable records are reported alongside, with their counts.

The outcome:
- **Pass:** every row passes for a model.
- **Fail:** any point estimate misses.
- **Inconclusive:** every point estimate passes but an interval bound misses, or a class is under 100.

The redundant-evidence class (a duplicated gold sentence, with only one copy cited) is reported as a secondary result. Its expected verdict is `decorative`. It has no pass bar because the construction makes it easy by design.
