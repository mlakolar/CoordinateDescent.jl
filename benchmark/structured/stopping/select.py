"""Freeze the candidate before examining any holdout Julia results."""
import csv
import json
import math
from pathlib import Path

rows = [r for p in Path("calibration-v2").glob("calibration-*.tsv")
        for r in csv.DictReader(p.open(), delimiter="\t") if r["setting"] == "gap3e7"]
assert len(rows) == 135, len(rows)
assert len({(r["seed"], r["case"]) for r in rows}) == 45
assert all(r["converged"] == "true" for r in rows)
for r in rows:
    if r["round"] == "1":
        for key, threshold in (("coefficient_error", 1e-4), ("objective_error", 1e-6),
                               ("r_coefficient_error", 1e-4), ("r_objective_error", 1e-6)):
            assert math.isfinite(float(r[key])) and float(r[key]) <= threshold, r
Path("selection.json").write_text(json.dumps(dict(
    selected="gap3e7", relative_gap_factor=0.3, residual_guard_factor=5,
    calibration_datasets=[8201,8202,8203], calibration_cases=15,
    calibration_measurements=135, coefficient_target=1e-4, objective_target=1e-6,
    rationale="All calibration gates pass; retain objective safety margin instead of1e-6 boundary rule",
    holdout_datasets=[8301,8302,8303]), indent=2) + "\n")
print("Selected gap3e7 before holdout Julia execution: all45 dataset/case gates pass.")
