"""t06 = t05 with FIDELITY-FIRST selection (no node penalty when choosing the search candidate); size reduced only by pruning that costs < tol fidelity."""
import extractors.t05_binary_guard_export as t05
class Model2Formula(t05.Model2Formula):
    def __init__(self, *a, **k): super().__init__(*a, **k); self.penalty = 0.0
