"""t07 on tabular_regression + classification (the weak types)."""
import sys, pathlib; HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import bench2; from extractors import t07_capacity as t07
bench2.run(t07.Model2Formula, "t07_capacity", only_types=["tabular_regression", "classification"])
