"""t08 additive symbolic distillation on tabular_regression + classification."""
import sys, pathlib; HERE = pathlib.Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import bench2; from extractors import t08_additive as t08
bench2.run(t08.Model2Formula, "t08_additive", only_types=["tabular_regression", "classification"])
