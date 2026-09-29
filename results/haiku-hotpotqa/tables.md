# Poindexter bench metrics

## hotpotqa

Models: claude-haiku-4-5-20251001. k: [3].

| metric | value |
|---|---|
| records | 60 |
| status | ok 60, non_compliant 0, unstable_original 0, k_unavailable 0 |
| compliance (records) | 1.000 [0.940, 1.000] (60/60) |
| compliance (calls) | 0.981 [0.976, 0.985] (3284/3348) |
| retries | 618 |
| non-compliant codes | - |
| parametric rate | 0.193 [0.111, 0.313] (11/57) |
| correct | 0.733 [0.610, 0.829] (44/60) |
| citation precision | 0.965 |
| citation recall | 0.904 |
| citation n (abstained, excluded) | 57 (3) |

### Verdicts

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| all ok | 57 | 47 (82%) | 8 (14%) | 1 (2%) | 1 (2%) |
| known-grounded | 31 | 27 (87%) | 3 (10%) | 1 (3%) | 0 (0%) |

Known-grounded: false alarm (decorative) 0.097 [0.033, 0.249] (3/31); mean L recall vs gold 0.710 (n=31).

### Flags

| flag | rate |
|---|---|
| parametric | 0.193 [0.111, 0.313] (11/57) |
| fabricates | 0.088 [0.038, 0.189] (5/57) |
| fabricates_on_replace | 0.140 [0.073, 0.253] (8/57) |
| position_sensitive | 0.158 [0.085, 0.274] (9/57) |
| span_in_cites | 0.982 [0.907, 0.997] (56/57) |
| reproduced | n/a (0/0) |
