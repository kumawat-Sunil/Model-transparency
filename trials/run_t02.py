"""t02: prune + refit + multi-seed (see extractor docstring)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from bench import run; from extractors.t02_prune_refit import Model2Formula
run(Model2Formula, "t02_prune_refit", note="3 seeds + greedy prune (tol .003) + LSQ constant refit + 4-sig-fig rounding")
