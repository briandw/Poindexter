# Poindexter bench metrics

## squad

Models: claude-sonnet-5-5. k: [3].

| metric | value |
|---|---|
| records | 30 |
| status | ok 30, non_compliant 0, unstable_original 0, k_unavailable 0 |
| compliance (records) | 1.000 [0.886, 1.000] (30/30) |
| compliance (calls) | 1.000 [0.996, 1.000] (895/895) |
| retries | 1 |
| non-compliant codes | - |
| parametric rate | 0.167 [0.067, 0.359] (4/24) |
| correct | 0.857 [0.654, 0.950] (18/21) |
| citation precision | 1.000 |
| citation recall | 0.976 |
| citation n (abstained, excluded) | 21 (0) |

### Verdicts

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| all ok | 24 | 20 (83%) | 2 (8%) | 2 (8%) | 0 (0%) |
| known-grounded | 14 | 12 (86%) | 0 (0%) | 2 (14%) | 0 (0%) |

Known-grounded: false alarm (decorative) 0.000 [0.000, 0.215] (0/14); mean L recall vs gold 1.000 (n=14).

### Flags

| flag | rate |
|---|---|
| parametric | 0.167 [0.067, 0.359] (4/24) |
| fabricates | 0.125 [0.043, 0.310] (3/24) |
| fabricates_on_replace | 0.125 [0.043, 0.310] (3/24) |
| position_sensitive | 0.125 [0.043, 0.310] (3/24) |
| span_in_cites | 0.958 [0.798, 0.993] (23/24) |
| reproduced | n/a (0/0) |

### Unanswerable

Abstain 0.667 [0.354, 0.879] (6/9); not ok: 0.

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| non-abstentions | 3 | 3 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |

| flag (non-abstentions) | true | false | none |
|---|---|---|---|
| parametric | 0 | 3 | 0 |
| fabricates | 0 | 3 | 0 |
| fabricates_on_replace | 0 | 3 | 0 |
| position_sensitive | 1 | 2 | 0 |
| span_in_cites | 3 | 0 | 0 |
| reproduced | 0 | 0 | 3 |
