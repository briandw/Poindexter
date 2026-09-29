# Poindexter bench metrics

## squad

Models: claude-sonnet-5-5. k: [3].

| metric | value |
|---|---|
| records | 150 |
| status | ok 150, non_compliant 0, unstable_original 0, k_unavailable 0 |
| compliance (records) | 1.000 [0.975, 1.000] (150/150) |
| compliance (calls) | 1.000 [0.998, 1.000] (1813/1813) |
| retries | 13 |
| non-compliant codes | - |
| parametric rate | 0.660 [0.581, 0.731] (99/150) |
| correct | 1.000 [0.975, 1.000] (150/150) |
| citation precision | 0.959 |
| citation recall | 1.000 |
| citation n (abstained, excluded) | 150 (0) |

### Verdicts

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| all ok | 150 | 150 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| known-grounded | 51 | 51 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |

Known-grounded: false alarm (decorative) 0.000 [0.000, 0.070] (0/51); mean L recall vs gold n/a (n=0).

### Flags

| flag | rate |
|---|---|
| parametric | 0.660 [0.581, 0.731] (99/150) |
| fabricates | 0.033 [0.014, 0.076] (5/150) |
| fabricates_on_replace | n/a (0/0) |
| position_sensitive | n/a (0/0) |
| span_in_cites | 0.993 [0.963, 0.999] (149/150) |
| reproduced | n/a (0/0) |

### k sweep (reference k=5)

50 records available at every k; 0 unavailable (A changes at some k: 1 0, 3 0, 5 0).

| k | agreement with reference | changed | transitions |
|---|---|---|---|
| 1 | 1.000 [0.929, 1.000] (50/50) | 0 | - |
| 3 | 1.000 [0.929, 1.000] (50/50) | 0 | - |
| 5 | 1.000 [0.929, 1.000] (50/50) | 0 | - |

### Swap-validated verdicts (headline)

Classes: grounded 150, decorative 0, excluded 0 (-). Minimum 100 per class: NOT met.

| class | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| grounded | 150 | 150 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| decorative | 0 | 0 | 0 | 0 | 0 |

| rate | Poindexter | Poindexter, unstable excluded | span_in_cites baseline |
|---|---|---|---|
| decorative recall | n/a (0/0) | n/a (0/0) | n/a (0/0) |
| false alarm | 0.000 [0.000, 0.025] (0/150) | 0.000 [0.000, 0.025] (0/150) | 0.007 [0.001, 0.037] (1/150) |

Unstable verdicts: grounded 0, decorative 0. span_in_cites true: grounded 0.993 [0.963, 0.999] (149/150), decorative n/a (0/0); missing grounded 0, decorative 0.

Original variant (Poindexter run on the swapped record):

| class | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| grounded | 150 | 147 (98%) | 1 (1%) | 2 (1%) | 0 (0%) |
| decorative | 0 | 0 | 0 | 0 | 0 |

### Redundant evidence (expected: decorative)

59 records, 50 cite exactly one copy; excluded cites_both 9. Decorative 0.980 [0.895, 0.996] (49/50); unstable excluded 0.980 [0.895, 0.996] (49/50).

| set | n | grounded | decorative | incomplete | unstable |
|---|---|---|---|---|---|
| kept | 50 | 1 (2%) | 49 (98%) | 0 (0%) | 0 (0%) |
