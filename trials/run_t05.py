"""t05: binary linearisation + domain guard + standalone export (each task -> trials/exported/t05_<task>.py, then re-imported and checked)."""
import sys, pathlib, importlib.util, numpy as np; HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import bench; from extractors.t05_binary_guard_export import Model2Formula
class Exporting(Model2Formula):
    def fit(self, X):
        super().fit(X); name = {3: "gravity", 10: "diabetes", 4: "airline", 5: "titanic"}[len(self.raw)]
        p = HERE / "exported" / f"t05_{name}.py"; self.export(p, f"{name} formula (trial t05)")
        spec = importlib.util.spec_from_file_location("m", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        Xc = X[:200]; pe = m.predict(**{n: Xc[:, i] for i, n in enumerate(self.raw)}) * np.ones(len(Xc))
        print(f"[export check {name}] standalone file vs in-memory formula max|diff| = {np.max(np.abs(pe - self.predict(Xc))):.2e} ; in-domain rate on train = {self.in_domain(X).mean():.2f}")
        return self
bench.run(Exporting, "t05_binary_guard_export", note="t04 + binary linearisation + domain guard + standalone numpy export")
