# Poindexter bench metrics

## squad

Models: claude-haiku-4-5-20251001. k: [3].

| metric | value |
|---|---|
| records | 155 |
| status | ok 155, non_compliant 0, unstable_original 0, k_unavailable 0 |
| compliance (records) | 1.000 [0.976, 1.000] (155/155) |
| compliance (calls) | 1.000 [0.998, 1.000] (2109/2109) |
| retries | 249 |
| non-compliant codes | - |
| parametric rate | 0.542 [0.463, 0.618] (84/155) |
| correct | 1.000 [0.976, 1.000] (155/155) |
| citation precision | 0.984 |
| citation recall | 0.987 |
| citation n (abstained, excluded) | 155 (0) |

### Verdicts

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| all ok | 155 | 150 (97%) | 1 (1%) | 4 (3%) | 0 (0%) |
| known-grounded | 71 | 70 (99%) | 0 (0%) | 1 (1%) | 0 (0%) |

Known-grounded: false alarm (decorative) 0.000 [0.000, 0.051] (0/71); mean L recall vs gold n/a (n=0).

### Flags

| flag | rate |
|---|---|
| parametric | 0.542 [0.463, 0.618] (84/155) |
| fabricates | 0.032 [0.014, 0.073] (5/155) |
| fabricates_on_replace | n/a (0/0) |
| position_sensitive | n/a (0/0) |
| span_in_cites | 0.974 [0.936, 0.990] (151/155) |
| reproduced | n/a (0/0) |

### k sweep (reference k=5)

50 records available at every k; 0 unavailable (A changes at some k: 1 0, 3 0, 5 0).

| k | agreement with reference | changed | transitions |
|---|---|---|---|
| 1 | 0.980 [0.895, 0.996] (49/50) | 1 | grounded->incomplete 1 |
| 3 | 0.980 [0.895, 0.996] (49/50) | 1 | grounded->incomplete 1 |
| 5 | 1.000 [0.929, 1.000] (50/50) | 0 | - |

### Swap-validated verdicts (headline)

Classes: grounded 150, decorative 3, excluded 2 (original_no_gold_cite 2). Minimum 100 per class: NOT met.

| class | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| grounded | 150 | 147 (98%) | 0 (0%) | 3 (2%) | 0 (0%) |
| decorative | 3 | 3 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |

| rate | Poindexter | Poindexter, unstable excluded | span_in_cites baseline |
|---|---|---|---|
| decorative recall | 0.000 [0.000, 0.561] (0/3) | 0.000 [0.000, 0.561] (0/3) | 0.000 [0.000, 0.561] (0/3) |
| false alarm | 0.000 [0.000, 0.025] (0/150) | 0.000 [0.000, 0.025] (0/150) | 0.013 [0.004, 0.047] (2/150) |

Unstable verdicts: grounded 0, decorative 0. span_in_cites true: grounded 0.987 [0.953, 0.996] (148/150), decorative 1.000 [0.439, 1.000] (3/3); missing grounded 0, decorative 0.

Original variant (Poindexter run on the swapped record):

| class | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| grounded | 150 | 145 (97%) | 0 (0%) | 5 (3%) | 0 (0%) |
| decorative | 3 | 2 (67%) | 0 (0%) | 1 (33%) | 0 (0%) |

### Redundant evidence (expected: decorative)

59 records, 57 cite exactly one copy; excluded cites_neither 1, wrong_answer 1. Decorative 0.982 [0.907, 0.997] (56/57); unstable excluded 0.982 [0.907, 0.997] (56/57).

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| kept | 57 | 1 (2%) | 56 (98%) | 0 (0%) | 0 (0%) |
