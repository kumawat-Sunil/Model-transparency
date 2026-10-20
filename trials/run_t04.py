"""t04: structure rule (periodic features present -> safe ops) + deep pruning."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from bench import run; from extractors.t04_structure_rules import Model2Formula
run(Model2Formula, "t04_structure_rules", note="t03 + periodic-feature rule (safe ops) + deep Add-term pruning")
