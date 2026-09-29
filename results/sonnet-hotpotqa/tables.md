# Poindexter bench metrics

## hotpotqa

Models: claude-sonnet-5-5. k: [3].

| metric | value |
|---|---|
| records | 30 |
| status | ok 30, non_compliant 0, unstable_original 0, k_unavailable 0 |
| compliance (records) | 1.000 [0.886, 1.000] (30/30) |
| compliance (calls) | 1.000 [0.997, 1.000] (1433/1433) |
| retries | 8 |
| non-compliant codes | - |
| parametric rate | 0.300 [0.167, 0.479] (9/30) |
| correct | 0.733 [0.556, 0.858] (22/30) |
| citation precision | 1.000 |
| citation recall | 0.933 |
| citation n (abstained, excluded) | 30 (0) |

### Verdicts

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| all ok | 30 | 22 (73%) | 3 (10%) | 5 (17%) | 0 (0%) |
| known-grounded | 10 | 10 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |

Known-grounded: false alarm (decorative) 0.000 [0.000, 0.278] (0/10); mean L recall vs gold 0.650 (n=10).

### Flags

| flag | rate |
|---|---|
| parametric | 0.300 [0.167, 0.479] (9/30) |
| fabricates | 0.033 [0.006, 0.167] (1/30) |
| fabricates_on_replace | 0.033 [0.006, 0.167] (1/30) |
| position_sensitive | 0.100 [0.035, 0.256] (3/30) |
| span_in_cites | 0.867 [0.703, 0.947] (26/30) |
| reproduced | n/a (0/0) |
