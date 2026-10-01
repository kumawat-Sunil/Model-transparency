# model2formula — turn a trained ML model + its preprocessing into one formula on raw data

```
pip install -r requirements.txt
python demo.py          # 4 worked examples, writes output/*.py
```

## Use with your model
```python
from model2formula import extract
preprocess = {                                   # write each preprocessing step once (L = math library: L.log, L.sin, L.cos, L.exp, L.sqrt, L.pi)
    "log_price": lambda e, L: L.log(e["price"]),
    "sin_m":     lambda e, L: L.sin(2 * L.pi * e["month"] / 12),
    "qty":       lambda e, L: e["qty"],
}
res = extract(model, preprocess, ["price", "month", "qty"], X_train_raw,
              problem_type="tabular_regression",  # physical_law | forecasting | tabular_regression | classification
              model_input="features",             # "features": model was trained on preprocessed features; "raw": model takes raw columns
              classification=False)               # True -> uses predict_proba, formula returns a probability
res.full_formula      # sympy formula in raw inputs
res.predict(X_raw)    # prediction with the formula only
res.in_domain(X_raw)  # False where inputs are outside the training range (formula not guaranteed there)
res.export("f.py")    # standalone numpy file: predict(**columns), in_domain(**columns), features(**columns)
```
X_train_raw: numpy array, columns in the order of the raw names. Categorical text must be integer-coded first.

## Method per problem type (verified on 3-5 datasets each, 16 total)
| type | method | typical result |
|---|---|---|
| physical_law | symbolic search (Operon), raw/log representations, prune + refit + snap constants | short exact-looking laws, e.g. gravity -> m1*m2/r**2 |
| forecasting | same + no trig on trend features | short formulas that extrapolate trends (often beat tree models out of range) |
| tabular_regression | best numerically-sane of {sparse polynomial, additive symbolic, symbolic search} | fidelity to model 0.96-0.99, 45-180 nodes |
| classification | same, on the logit | fidelity 0.87-0.997, AUC equal to the model |

## Limits
Binary classification only; numeric/time-series data only; ~<30 inputs tested; table-based preprocessing (quantile, target encoding) cannot be expressed.
Files: `model2formula.py` (API), `engine/` (methods), `demo.py`, `data/` (demo data), `output/` (demo exports).
