"""Verify t05 (readability-first) and t06 (fidelity-first) on 3-5 datasets per problem type."""
import sys, pathlib; HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import bench2; from extractors import t05_binary_guard_export as t05, t06_fidelity_first as t06
bench2.run(t05.Model2Formula, "t05_binary_guard_export"); bench2.run(t06.Model2Formula, "t06_fidelity_first")
