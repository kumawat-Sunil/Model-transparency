"""t01: baseline = Operon, representations {raw, log-inputs, log-log}, max_len {10,20,25}, pick by held-out fidelity - 0.0005*nodes."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from bench import run; from extractors.t01_operon_multirep import Model2Formula
run(Model2Formula, "t01_operon_multirep", note="baseline multi-representation Operon")
