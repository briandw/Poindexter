# Poindexter v3: edit-size sweep

v2 found that a probe C verdict depends on how far C's edit moves the fact. On the 25 facts where Sonnet 5.5 (knowledge allowed) followed a ±1–5-year edit, C, which moves years by up to 25, called 19 decorative. v3 measures that dependence directly, and adds an agent told explicitly that the document is the authority.

## Design

- **Corpus:** the v2 corpus, with sized edits of each agreeing (L0) passage. Only the answer sentence changes. `poindexter sweep CORPUS --build DIR` builds them deterministically, with no model calls.
  - **years:** S1 ±1–2, S2 ±3–5, S3 ±10–25, S4 ±50–100, S5 ±200–500, all kept in 1000–2025
  - **counts:** S1 5–10%, S2 15–25%, S3 40–60%, S4 ×3–5, S5 ×20–50
  - **entities:** E1 the v2 mild value, E2 a same-type pool value, E3 the v2 strong value

  No value appears elsewhere in the passage or the question. S2 matches v2's ground-truth size, and S3 matches probe C's.
- **Agents:**
  - `context`: v2's context-only prompt.
  - `open`: v2's knowledge-allowed prompt.
  - `document`: new. Rule 1 says the context is the authority: "We want the answer according to the context: if it contradicts what you know or believe to be true, the context's answer is still the correct answer here."
- **Models:** Haiku 4.5 and Sonnet 5.5. For each, the same 150 confident-prior facts as v2 (the same seeded selection), k=3.
- **Scoring:** a fact counts for an agent if its L0 answer is the prior value and cites the answer sentence, by majority of samples (v2's rule).
  - Per size, the answer **follows** the edit (the edited value) or **keeps** the prior; anything else is reported separately.
  - The follow rate uses Wilson 95% intervals.
  - Pairwise agreement between sizes is computed per fact. The call at S3 is probe C's verdict, and the call at S2 is v2's truth.
  - A tipping size is recorded per numeric fact: the first size at which the answer stops following.

## Pre-registered hypotheses

Committed before any sweep run.

- **H1 (explicit instruction):** under `document`, for both models and every kind and size, the follow rate is ≥ 0.90 with a Wilson lower bound ≥ 0.80.
- **H2 (graded dependence):** under `open`, Sonnet's number follow rate at S1 is above its rate at S5, with non-overlapping intervals.
- **H3 (why C disagreed in v2):** under `open` for Sonnet, per-fact agreement between S2 and S3 is below agreement between S1 and S2.

Everything else is descriptive: the full curves, the tipping-size distributions, context vs document at S5 and E3, and uncited-answer rates.

## Budget

About 1,500 sized records in total. Per model, the 150 facts give roughly 600 records × 3 agents × k=3, plus 450 L0 calls for the new agent. That comes to about 12k calls for both models. The L0 calls for `context` and `open` are cached from v2.

## Errata (after the runs, from code review)

Codex reviewed the code after the runs had started. The hypotheses above are unchanged.

- **S2 and S3 only approximate v2's ranges.** They're described above as "v2's ground-truth size" and "probe C's", but v2's mild years span ±1–5 and C's years span ±1–25. So H3 tests two new edit bands; it doesn't reproduce v2's exact comparison.
- **Count edits can round out of their window.** For example, 46 +/- 5% rounds to 48, which is 4.3%. The builder now checks the final value and skips a size when no integer fits. That dropped 38 count edits and changed 7. The runs were repeated on the fixed records, which needed 3 new calls.
- **H3 is scored on one common cohort.** That cohort is the facts with a decided call at every number size (69 for Sonnet, knowledge allowed; 13 excluded). Undecided or missing sizes make a fact's tipping size "unknown" rather than "never".
- **Entity duplicate checks now ignore punctuation** (Mary-Jane = Mary Jane). No entity edit changed.
- **The cohort differs from v2's for Sonnet, knowledge allowed.** v2 ran that cell on all 325 confident facts; v3 uses the first 150 of the same seeded order.
