"""t03: snapping + safe operator set + edge-aware selection."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from bench import run; from extractors.t03_snap_safe import Model2Formula
run(Model2Formula, "t03_snap_safe", note="full+safe opsets, min(holdout,edge) selection, prune, snap to k/2")
