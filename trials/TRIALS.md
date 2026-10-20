# Trials index (every attempt kept; nothing deleted)

| trial | extractor file | run script | output | what changed |
|---|---|---|---|---|
| t01 | extractors/t01_operon_multirep.py | run_t01.py | outputs/t01_operon_multirep.txt | Operon, raw/log/log-log, len 10/20/25, select by held-out fidelity |
| t02 | extractors/t02_prune_refit.py | run_t02.py | outputs/t02_prune_refit.txt | +3 seeds, greedy pruning, least-squares constant refit, rounding |
| t03 | extractors/t03_snap_safe.py | run_t03.py | outputs/t03_snap_safe.txt | +safe operator set, min(holdout, edge) selection, snap constants to k/2 |
| t04 | extractors/t04_structure_rules.py | run_t04.py | outputs/t04_structure_rules.txt | +periodic-feature rule (no trig on trends), deep pruning |
| t05 | extractors/t05_binary_guard_export.py | run_t05.py | outputs/t05_binary_guard_export.txt, exported/t05_*.py | +0/1-variable linearisation, domain guard, standalone numpy export |

Benchmark (fixed for all trials): trials/bench.py — gravity (MLP, truth m1*m2/r^2), diabetes (MLP), airline forecast (XGBoost on engineered features), titanic (MLP classifier).
All scores per trial/task: LEADERBOARD.csv (appended, never overwritten).

## Pivot (formula_score / fidelity / nodes)
```
                    fidelity                                                                                 formula_score                                                                                         nodes                                                                          
trial    t01_operon_multirep t02_prune_refit t03_snap_safe t04_structure_rules t05_binary_guard_export t01_operon_multirep t02_prune_refit t03_snap_safe t04_structure_rules t05_binary_guard_export t01_operon_multirep t02_prune_refit t03_snap_safe t04_structure_rules t05_binary_guard_export
task                                                                                                                                                                                                                                                                                              
airline                0.394           0.751         0.749               0.450                   0.450               0.870           0.300         0.303               0.837                   0.837                31.0            13.0          13.0                 8.0                     8.0
diabetes               0.932           0.929         0.932               0.932                   0.932               0.435           0.433         0.467               0.467                   0.467                39.0            34.0          26.0                26.0                    26.0
gravity                0.995           0.995         0.995               0.994                   0.994               0.999           0.999         1.000               1.000                   1.000                15.0            11.0          12.0                 6.0                     6.0
titanic                0.926           0.916         0.910               0.884                   0.884               0.864           0.862         0.856               0.851                   0.851                32.0            28.0          32.0                29.0                    26.0
```