# Assumptions (numbered; revisit as evidence arrives)
A1. "Raw" input for forecasting = the lag window y[t-1..t-12], price, calendar month. Lags are index shifts and are treated as raw symbols; windowing itself is not symbolized.
A2. Teacher f is deterministic at inference; its predictions are the distillation target.
A3. Complexity = number of nodes in the sympy expression tree (count_ops/preorder), constants count 1.
A4. PySR unavailable (no Julia); gplearn + sparse regression stand in. Conclusions about SR quality are therefore specific to these engines.
A5. Initial datasets are synthetic with known ground truth (to separate "can we recover structure" from "is data noisy"); real data comes in P6.
A6. Tree-ensemble exact export (Piecewise) is the "exact but huge" reference point.
