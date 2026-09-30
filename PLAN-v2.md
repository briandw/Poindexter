# Poindexter v2: surprise

v1 (README, Results) found the tool's blind spot. Removal probes can't tell whether an answer depends on a cited sentence when the document agrees with what the model already knows. An instruction-following model abstains when the sentence is removed, whether or not it read it. v2 fixes both the tool and the test.

## Idea

Grounding is only observable where the document surprises the model.

| | Document agrees with the prior | Document contradicts the prior |
|---|---|---|
| **Model has a prior** | Invisible: the same answer either way. Decorative citations hide here. | Visible: the document's value means grounded, the prior's value means decorative. |
| **No prior** (invented facts) | n/a | The answer must come from the document. |

So v2 makes surprise on purpose. It measures the prior closed-book, writes counterfactual documents that should move the answer, and watches whether the answer moves.

## Probe C (counterfactual)

This is a new probe in Poindexter ([#22](https://github.com/briandw/Poindexter/issues/22)), still training-free. When the answer text occurs in the cited units, C replaces that occurrence with a different value of the same type and asks again, k samples.

- **Numbers:** C uses the v1 plausible-swap rules.
- **Entities:** C uses a same-type alternative, drawn from other entities of that type in a fixed pool shipped with the corpus. In live use, an agent supplies the pool or it comes from the other units.

Outcome per sample: `follows` (the answer is the edited value), `same` (the answer is the original value), or `other`/`abstain`. The verdict:

- majority `follows`: `grounded`, because the answer depends on what the cited text says
- majority `same`: `decorative`
- otherwise: `unstable`

C is not applicable when the answer text isn't in the cited units. That case is reported, not guessed.

## Corpus

**Famous facts with strong priors.** Sonnet writes a pool of about 400 well-known facts, half numeric (years, counts) and half entity (people, places, organizations). Each fact comes with a short synthetic encyclopedic passage of 6 to 8 sentences that states the fact exactly once, in one sentence, with the answer text appearing nowhere else. Checks are deterministic. A fact is kept for a model only if its closed-book answer is correct on 3 of 3 samples (a confident prior).

Each kept fact gets these documents (the answer-bearing sentence is edited; every other sentence is identical):

| level | document | purpose |
|---|---|---|
| L0 | agrees with the prior | where Poindexter is judged |
| L1 | mild counterfactual: years ±1–5, counts ±15%, entities plausible; a model check confirms the edit doesn't contradict the rest of the passage (up to 5 redraws, else the fact is dropped) | ground truth, held-out draw |
| L2 | strong counterfactual (contradicts a famous fact outright) | surprise gradient |
| Cb / Ca | agreeing passage plus one inserted sentence stating the L1 value, before (Cb) or after (Ca) the true sentence | conflict behaviour (descriptive) |

**Control:** invented facts in invented passages (no prior; both models abstained 30/30 closed-book in the probe). Every citation here is grounded by construction. It measures false alarms where no memory exists.

L0, Cb and Ca records carry `meta.heldout_values` (the L1 and L2 values), and probe C never chooses them, so C's edit can't equal the ground-truth edit.

## Agents

Each model is audited as two agents, because stickiness depends on the prompt.

- **context:** the v1 prompt, answering only from context.
Both prompts gain a conflict rule: when the context gives conflicting answers, give each value, separated by " / ", and cite every unit that states one. The v1 prompts are kept byte-identical as `context_v1` and `open_v1` so v1 replays from its cache.

- **open:** may use its own knowledge. The validator allows `cites: []` for an answer from knowledge, so an honest uncited memory answer is expressible. The v1 open arm lacked this and was tilted toward abstaining.

## Ground truth and scoring

For each model × agent × fact:

- **Truth for the L0 citation.** The L1 twin decides: the answer follows the edited value means **grounded**; it keeps the prior means **decorative**; anything else excludes the fact. The L1 draw uses a different seed from probe C's edit, so C is never scored against its own edit.
- **Scored:** only records where the L0 answer is the prior value and cites the answer-bearing sentence.
- **Compared verdicts on L0:**
  - probe C
  - the v1 removal verdict (R_remove, M)
  - the `span_in_cites` baseline

  Metrics are decorative recall and false-alarm rate, with Wilson 95% intervals.
- **Descriptive:** update rate by level (L1, L2), answer type, model, and agent; abstention and uncited-answer rates for the open agent.
- **Control:** false-alarm rates for C and removal on invented facts, where truth is grounded.

## Pilot findings that shaped the thresholds

- Tightening L1 removed Haiku's decorative cases under the context prompt. The 2 in pilot 1 came from mild edits that clashed with their passage.
- With the conflict rule, Haiku also follows the mild edits under the open prompt (0 of 30 kept the prior). Sonnet under the open prompt keeps its prior on 27 of 30. So recall can be judged only for Sonnet/open, and false alarms elsewhere.
- On conflict documents, models mostly report both values, as the rule asks. That's instruction-following, so the conflict level is descriptive.

## Pre-registration

A 30-fact pilot per model checks the construction, compliance, and class balance. After the pilot and before any evaluation run, `THRESHOLDS-v2.md` fixes the pass bars for C. After an evaluation run starts, prompts, construction, and thresholds stay fixed. A negative result is acceptable and is reported as such.

## Budget

About 27 calls per fact, per model and agent (O on L0, L1, L2, Cb and Ca; N, R_remove, M and C on L0). Full runs: 150 facts for each context cell and Haiku/open, and every confident fact for Sonnet/open, where the decorative class lives. Each cell also runs 30 invented facts. Roughly 20k calls, about an hour at the machine's CPU limit. Everything is cached, and results and the cache are committed as in v1.

## Work

- Corpus builder: fact pool, passages, L0/L1/L2 edits, invented control, and deterministic checks.
- Probe C in `probes.py`, `runner.py` and `verdict.py`, and the open agent's validator.
- `poindexter surprise` pipeline and scoring.
- Pilot, thresholds, runs, write-up.
