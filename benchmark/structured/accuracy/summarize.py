"""Rebuild compact accuracy/performance summaries without running a solver."""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

root, destination = map(Path, sys.argv[1:3])
destination.mkdir(parents=True, exist_ok=True)

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))

def yes(value):
    return str(value).lower() == "true"

timing = [r for p in root.glob("holdout/timings-*.tsv") for r in rows(p)]
accuracy = [r for p in root.glob("holdout/accuracy-*.tsv") for r in rows(p)]
cases = sorted({r["case"] for r in timing})
result = []
for case in cases:
    cell = {"case": case}
    medians = {}
    for setting in ("baseline", "candidate"):
        rr = [r for r in timing if r["case"] == case and r["setting"] == setting]
        aa = [r for r in accuracy if r["case"] == case and r["setting"] == setting]
        seed_medians = {seed: statistics.median(float(r["seconds"]) for r in rr if r["seed"] == seed)
                        for seed in ("8501", "8502", "8503") if any(r["seed"] == seed for r in rr)}
        medians[setting] = seed_medians
        gaps = [float(r["dual_gap"]) for r in aa if math.isfinite(float(r["dual_gap"]))]
        cell[setting] = {
            "measurements": len(rr), "median_seconds": statistics.median(float(r["seconds"]) for r in rr),
            "seed_medians": seed_medians,
            "max_coefficient_error": max(float(r["coefficient_error"]) for r in aa),
            "max_objective_error": max(float(r["objective_error"]) for r in aa),
            "max_duality_gap": max(gaps) if gaps else None,
            "coverage_and_convergence_pass": len(rr) == 9 and len(aa) == 3 and set(seed_medians) == {"8501", "8502", "8503"}
                and all(r["status"] == "ok" and yes(r["converged"]) for r in rr),
            "accuracy_pass": len(aa) == 3 and all(yes(r["pass"]) for r in aa),
            "median_iterations": statistics.median(int(r["iterations"]) for r in rr),
            "median_allocated_bytes": statistics.median(int(r["bytes"]) for r in rr),
        }
    ratios = [medians["baseline"][seed] / medians["candidate"][seed]
              for seed in sorted(medians["baseline"].keys() & medians["candidate"].keys())]
    cell["paired_baseline_over_candidate"] = statistics.median(ratios)
    cell["ratio_range"] = [min(ratios), max(ratios)]
    result.append(cell)
report = {"cases": result, "workloads": len(result), "timings": len(timing),
          "accuracy_checks": len(accuracy), "independent_datasets": 3, "timings_per_dataset": 3,
          "passing_workloads": {s: sum(c[s]["accuracy_pass"] and c[s]["coverage_and_convergence_pass"] for c in result)
                                for s in ("baseline", "candidate")}}
(destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
with (destination / "comparison.tsv").open("w") as f:
    names = ["case", "baseline_seconds", "candidate_seconds", "paired_baseline_over_candidate", "baseline_pass", "candidate_pass",
             "candidate_coefficient_error", "candidate_objective_error", "candidate_duality_gap"]
    writer = csv.DictWriter(f, fieldnames=names, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for c in result:
        writer.writerow(dict(case=c["case"], baseline_seconds=c["baseline"]["median_seconds"],
            candidate_seconds=c["candidate"]["median_seconds"], paired_baseline_over_candidate=c["paired_baseline_over_candidate"],
            baseline_pass=c["baseline"]["accuracy_pass"], candidate_pass=c["candidate"]["accuracy_pass"],
            candidate_coefficient_error=c["candidate"]["max_coefficient_error"], candidate_objective_error=c["candidate"]["max_objective_error"],
            candidate_duality_gap=c["candidate"]["max_duality_gap"] if c["candidate"]["max_duality_gap"] is not None else "NaN"))
print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
