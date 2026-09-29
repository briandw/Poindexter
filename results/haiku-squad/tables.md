# Poindexter bench metrics

## squad

Models: claude-haiku-4-5-20251001. k: [3].

| metric | value |
|---|---|
| records | 60 |
| status | ok 60, non_compliant 0, unstable_original 0, k_unavailable 0 |
| compliance (records) | 1.000 [0.940, 1.000] (60/60) |
| compliance (calls) | 0.999 [0.996, 1.000] (1764/1766) |
| retries | 251 |
| non-compliant codes | - |
| parametric rate | 0.049 [0.013, 0.161] (2/41) |
| correct | 0.714 [0.564, 0.828] (30/42) |
| citation precision | 1.000 |
| citation recall | 0.988 |
| citation n (abstained, excluded) | 40 (2) |

### Verdicts

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| all ok | 41 | 35 (85%) | 1 (2%) | 5 (12%) | 0 (0%) |
| known-grounded | 27 | 26 (96%) | 0 (0%) | 1 (4%) | 0 (0%) |

Known-grounded: false alarm (decorative) 0.000 [0.000, 0.125] (0/27); mean L recall vs gold 1.000 (n=27).

### Flags

| flag | rate |
|---|---|
| parametric | 0.049 [0.013, 0.161] (2/41) |
| fabricates | 0.171 [0.085, 0.313] (7/41) |
| fabricates_on_replace | 0.146 [0.069, 0.284] (6/41) |
| position_sensitive | 0.098 [0.039, 0.225] (4/41) |
| span_in_cites | 0.927 [0.806, 0.975] (38/41) |
| reproduced | n/a (0/0) |

### Unanswerable

Abstain 0.944 [0.742, 0.990] (17/18); not ok: 0.

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| non-abstentions | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |

| flag (non-abstentions) | true | false | none |
|---|---|---|---|
| parametric | 0 | 1 | 0 |
| fabricates | 1 | 0 | 0 |
| fabricates_on_replace | 1 | 0 | 0 |
| position_sensitive | 1 | 0 | 0 |
| span_in_cites | 1 | 0 | 0 |
| reproduced | 0 | 0 | 1 |
