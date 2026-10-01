"""t07 = t06 with more search capacity for many-input problems: lengths (10,20,35,50), generations 400, population 2000."""
import extractors.t06_fidelity_first as t06
class Model2Formula(t06.Model2Formula):
    def __init__(self, *a, **k): super().__init__(*a, **k); self.lengths, self.gens, self.pop = (10, 20, 35, 50), 400, 2000
