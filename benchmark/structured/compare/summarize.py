"""Summarize existing measurements; never launches solvers or hides failures."""
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def yes(x):
    return str(x).lower() == "true"


def summarize(root, destination, accuracy_root=None):
    cases = rows(root / "inputs/cases.tsv")
    reports = {}
    reference = {}
    for folder in ("baseline-full", "reference-special", "reference-loose", "reference-qr"):
        for path in (root / folder).glob("R-*.tsv"):
            for row in rows(path):
                reference[(row["seed"], row["case"], row["solver"], row["round"])] = row
    for stage in ("baseline-matched", "optimized-matched", "optimized-default", "optimized-selected"):
        accuracy_path=(accuracy_root or root) / stage / "accuracy.tsv"
        if not accuracy_path.exists():
            continue
        measured = dict(reference)
        sources=[root / stage]
        if stage in ("optimized-matched","optimized-default"):
            sources.append(root / stage.replace("optimized-","optimized-gram-"))
        for source in sources:
            for path in source.glob("Julia-*.tsv"):
                for row in rows(path):
                    measured[(row["seed"], row["case"], row["solver"], row["round"])] = row
        accuracy = rows(accuracy_path)
        gate = {(r["seed"], r["case"], r["comparison"]):
                yes(r["finite"]) and yes(r["coefficient_pass"]) and yes(r["objective_pass"])
                for r in accuracy}
        cells = defaultdict(list)
        failures = []
        for row in measured.values():
            cells[(row["case"], row["solver"])].append(row)
            if row["status"] != "ok" or not yes(row["converged"]):
                failures.append(row)
        summaries = []
        for c in cases:
            primary = "glmgen" if c["family"] == "signal" else "genlasso-special"
            for solver in ("glmgen", "glmgen-default", "glmgen-1e8", "genlasso", "genlasso-qr", "genlasso-special", "julia"):
                rr = cells.get((c["id"], solver), [])
                if not rr:
                    continue
                times = [float(r["seconds"]) for r in rr if r["status"] == "ok"]
                medians = {seed: statistics.median(float(r["seconds"]) for r in rr
                    if r["seed"] == seed and r["status"] == "ok")
                    for seed in ("8201", "8202", "8203")
                    if any(r["seed"] == seed and r["status"] == "ok" for r in rr)}
                summaries.append(dict(case=c["id"], solver=solver, primary_reference=primary,
                    measurements=len(rr), converged=sum(yes(r["converged"]) for r in rr),
                    median_seconds=statistics.median(times) if times else None,
                    min_seconds=min(times) if times else None, max_seconds=max(times) if times else None,
                    seed_medians=medians,
                    median_allocated_bytes=statistics.median(float(r["bytes"]) for r in rr) if solver=="julia" else None,
                    maximum_worker_hwm_bytes=max(float(r["worker_hwm_bytes"]) for r in rr)))
        ratios = []
        for c in cases:
            ref = "glmgen" if c["family"] == "signal" else "genlasso-special"
            ss = {r["solver"]: r for r in summaries if r["case"] == c["id"]}
            candidates = []
            choices=("glmgen-default", "glmgen-1e8", "glmgen") if c["family"]=="signal" else ("genlasso-qr","genlasso","genlasso-special")
            for candidate in choices:
                if candidate not in ss:
                    continue
                comparisons=(f"julia/{candidate}",f"{candidate}/tight","julia/tight") if c["family"]=="signal" else (f"julia/{candidate}",)
                qualifies = ss[candidate]["measurements"] == 9 and ss[candidate]["converged"] == 9 and all(
                    gate.get((s,c["id"],comp),False)
                    for s in ("8201","8202","8203")
                    for comp in comparisons)
                candidates.append(dict(solver=candidate,passes=qualifies,seconds=ss[candidate]["median_seconds"]))
            valid=[r for r in candidates if r["passes"]]
            if valid:
                ref=min(valid,key=lambda r:r["seconds"])["solver"]
            elif candidates:
                ref=min(candidates,key=lambda r:r["seconds"])["solver"]
            for cell in ss.values():
                cell["primary_reference"] = ref
            if not {"julia", ref} <= ss.keys():
                continue
            values = [ss[ref]["seed_medians"][s]/ss["julia"]["seed_medians"][s]
                for s in ("8201", "8202", "8203") if s in ss[ref]["seed_medians"] and s in ss["julia"]["seed_medians"]]
            comparisons = [f"julia/{ref}"]
            if ref.startswith("glmgen"):
                comparisons += ["julia/tight", f"{ref}/tight"]
            passes = all(gate.get((s, c["id"], comp), False)
                for s in ("8201", "8202", "8203") for comp in comparisons)
            coverage = all(ss[s]["measurements"]==9 and ss[s]["converged"]==9 for s in ("julia",ref))
            ratios.append(dict(case=c["id"],family=c["family"],n=int(float(c["n"])),p=int(float(c["p"])),
                order=int(float(c["order"])),path=int(float(c["path"])),reference=ref,
                reference_over_julia=statistics.median(values),min_ratio=min(values),max_ratio=max(values),
                accuracy_pass=passes,coverage_and_convergence_pass=coverage,julia_setting=c.get("julia_setting",stage),
                reference_candidates=json.dumps(candidates,separators=(",",":"))))
        reports[stage] = dict(cells=summaries,ratios=ratios,failures=failures,accuracy=accuracy)
    if {"optimized-matched","optimized-default"} <= reports.keys():
        selected=dict(cells=[],ratios=[],failures=[],accuracy=[])
        for c in cases:
            candidates=[]
            for stage in ("optimized-matched","optimized-default"):
                report=reports[stage]
                ratio=next((r for r in report["ratios"] if r["case"]==c["id"]),None)
                cell=next((r for r in report["cells"] if r["case"]==c["id"] and r["solver"]=="julia"),None)
                if ratio and cell:
                    candidates.append((stage,ratio,cell))
            good=[r for r in candidates if r[1]["accuracy_pass"] and r[1]["coverage_and_convergence_pass"]]
            if not candidates:
                continue
            stage,ratio,_=min(good or candidates,key=lambda r:r[2]["median_seconds"])
            selected["ratios"].append(dict(ratio,julia_setting=stage))
            for field in ("cells","failures","accuracy"):
                key="case"
                selected[field].extend(r for r in reports[stage][field] if r[key]==c["id"])
        reports["optimized-selected"]=selected
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(reports, indent=2)+"\n")
    for stage, report in reports.items():
        ratios = report["ratios"]
        if not ratios:
            continue
        with (destination / f"{stage}-ratios.tsv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(ratios[0]),delimiter="\t")
            writer.writeheader(); writer.writerows(ratios)
        print(stage, "cases",len(ratios),"accuracy passes",sum(r["accuracy_pass"] for r in ratios),
              "coverage passes",sum(r["coverage_and_convergence_pass"] for r in ratios),
              "solver failures",len(report["failures"]))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("root", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--accuracy-root",type=Path)
    a=p.parse_args(); summarize(a.root,a.output,a.accuracy_root)
