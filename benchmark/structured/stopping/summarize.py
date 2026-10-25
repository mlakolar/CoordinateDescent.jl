"""Summarize stopping experiments without launching a numerical process."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def summarize(root, output):
    result = {}
    for stage in ("calibration", "holdout/julia"):
        sources = [root / "calibration-v2", root / "calibration-warmfix"] if stage == "calibration" else [root / stage, root / "holdout/julia-final"]
        measured = {}
        for source in sources:
            for path in source.glob("calibration-*.tsv"):
                for row in read(path):
                    measured[row["seed"], row["case"], row["setting"], row["round"]] = row
        rows = list(measured.values())
        if not rows:
            continue
        cells = defaultdict(list)
        for row in rows:
            cells[row["case"], row["setting"]].append(row)
        summaries = []
        for (case, setting), measurements in sorted(cells.items()):
            errors = [r for r in measurements if r["round"] == "1"]
            seeds = sorted({r["seed"] for r in measurements})
            coverage = len(measurements) == 9 and len(seeds) == 3 and all(
                sorted(r["round"] for r in measurements if r["seed"] == s) == ["1", "2", "3"]
                for s in seeds
            )
            maximum = lambda key: max(float(r[key]) for r in errors)
            accuracy = coverage and all(
                math.isfinite(float(r[key])) and float(r[key]) <= threshold
                for r in errors
                for key, threshold in (("coefficient_error", 1e-4), ("objective_error", 1e-6),
                                       ("r_coefficient_error", 1e-4), ("r_objective_error", 1e-6))
            )
            if "r_reference_coefficient_error" in errors[0]:
                accuracy = accuracy and all(
                    math.isfinite(float(r[key])) and float(r[key]) <= threshold
                    for r in errors for key, threshold in
                    (("r_reference_coefficient_error", 1e-4), ("r_reference_objective_error", 1e-6)))
            summaries.append(dict(
                case=case, setting=setting, measurements=len(measurements),
                coverage_pass=coverage, accuracy_pass=accuracy,
                convergence_pass=all(r["converged"] == "true" for r in measurements),
                median_seconds=statistics.median(float(r["seconds"]) for r in measurements),
                seed_medians={s: statistics.median(float(r["seconds"]) for r in measurements
                                                 if r["seed"] == s) for s in seeds},
                median_iterations=statistics.median(int(r["iterations"]) for r in measurements),
                median_allocated_bytes=statistics.median(int(r["bytes"]) for r in measurements),
                max_coefficient_error=maximum("coefficient_error"),
                max_objective_error=maximum("objective_error"),
                max_r_coefficient_error=maximum("r_coefficient_error"),
                max_r_objective_error=maximum("r_objective_error")))
        result[stage] = summaries
        for setting in sorted({r["setting"] for r in summaries}):
            selected = [r for r in summaries if r["setting"] == setting]
            print(stage, setting, "cases", len(selected), "accuracy passes",
                  sum(r["accuracy_pass"] for r in selected), "convergence passes",
                  sum(r["convergence_pass"] for r in selected))
        pairs = []
        for case in sorted({r["case"] for r in summaries}):
            ss = {r["setting"]: r for r in summaries if r["case"] == case}
            if "baseline" not in ss:
                continue
            for setting, cell in ss.items():
                if setting == "baseline":
                    continue
                ratios = [ss["baseline"]["seed_medians"][s] / cell["seed_medians"][s]
                          for s in cell["seed_medians"]]
                pairs.append(dict(case=case, setting=setting, paired_baseline_over_candidate=
                                  statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios),
                                  candidate_pass=cell["accuracy_pass"] and cell["convergence_pass"],
                                  baseline_pass=ss["baseline"]["accuracy_pass"] and ss["baseline"]["convergence_pass"]))
        result[stage + "-ratios"] = pairs
    references = [r for path in (root / "holdout/reference").glob("R-*.tsv") for r in read(path)]
    if references and "holdout/julia" in result:
        comparisons = []
        for cell in result["holdout/julia"]:
            rr = [r for r in references if r["case"] == cell["case"]]
            complete = len(rr) == 9 and all(r["converged"].lower() == "true" and r["status"] == "ok" for r in rr)
            ratios = [statistics.median(float(r["seconds"]) for r in rr if r["seed"] == s) / t
                      for s, t in cell["seed_medians"].items() if any(r["seed"] == s for r in rr)]
            comparisons.append(dict(case=cell["case"], setting=cell["setting"],
                julia_seconds=cell["median_seconds"], reference=rr[0]["solver"] if rr else None,
                r_seconds=statistics.median(float(r["seconds"]) for r in rr) if rr else None,
                paired_r_over_julia=statistics.median(ratios) if ratios else None,
                min_ratio=min(ratios) if ratios else None, max_ratio=max(ratios) if ratios else None,
                accuracy_and_coverage_pass=complete and cell["accuracy_pass"] and cell["convergence_pass"]))
        result["holdout-r-comparisons"] = comparisons
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, cells in result.items():
        if cells and name.endswith("-ratios"):
            with (output / (name.replace("/", "-") + ".tsv")).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(cells[0]), delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.output)
