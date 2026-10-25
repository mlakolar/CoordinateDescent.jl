"""Summarize paired replications, never treating timing rounds as new inputs."""
import csv
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

root=Path(sys.argv[1])
out=root/"summary";out.mkdir(exist_ok=True)
rows=[]
for file in sorted((root/"main").glob("*-*/*/*.csv")):
    if file.name not in ("julia.csv","python.csv","r.csv","r-component.csv"):continue
    for r in csv.DictReader(open(file)):
        for key in ("frac","seconds","objective","gap","residual","rss_peak_kb"):
            try:r[key]=float(r[key])
            except (ValueError,TypeError):r[key]=float("nan")
        r["rep"]=int(r["rep"]);r["round"]=int(r["round"])
        rows.append(r)
fields=list(rows[0]) if rows else []
with open(out/"all-timings.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

rep_groups=defaultdict(list)
for r in rows:
    rep_groups[(r["case"],r["family"],r["solver"],r["mode"],r["frac"],r["rep"])].append(r)
reps=[]
for key,rs in sorted(rep_groups.items()):
    case,family,solver,mode,frac,rep=key
    passing=[r for r in rs if r["status"]=="passed"]
    rec=dict(case=case,family=family,solver=solver,mode=mode,frac=frac,rep=rep,
             observations=len(rs),passes=len(passing),
             seconds=st.median(r["seconds"] for r in rs if math.isfinite(r["seconds"])) if any(math.isfinite(r["seconds"]) for r in rs) else float("nan"),
             objective=st.median(r["objective"] for r in rs if math.isfinite(r["objective"])) if any(math.isfinite(r["objective"]) for r in rs) else float("nan"),
             certified=len(passing)==len(rs) and len(rs)==3)
    reps.append(rec)
with open(out/"replication-medians.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(reps[0]));w.writeheader();w.writerows(reps)
groups=defaultdict(list)
for r in reps:groups[(r["case"],r["family"],r["solver"],r["mode"],r["frac"])].append(r)
aggregated=[]
for key,rs in sorted(groups.items()):
    case,family,solver,mode,frac=key
    ts=[r["seconds"] for r in rs if math.isfinite(r["seconds"])]
    aggregated.append(dict(case=case,family=family,solver=solver,mode=mode,frac=frac,
        replications=len(rs),certified_replications=sum(r["certified"] for r in rs),
        median_ms=1000*st.median(ts) if ts else float("nan"),
        min_rep_ms=1000*min(ts) if ts else float("nan"),
        max_rep_ms=1000*max(ts) if ts else float("nan"),
        median_objective=st.median(r["objective"] for r in rs)))
with open(out/"aggregate.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(aggregated[0]));w.writeheader();w.writerows(aggregated)
ref={"rank":"cd_rank","nuclear":"cd_nuclear","sparse":"cd_sparse","component":"cd_component"}
lookup={(r["case"],r["solver"],r["mode"],r["frac"],r["rep"]):r for r in reps}
paired=[]
for key,rs in sorted(groups.items()):
    case,family,solver,mode,frac=key
    if mode not in ("fit","path") or solver==ref[family]:continue
    ratios=[];objdiff=[]
    for r in rs:
        b=lookup.get((case,ref[family],mode,frac,r["rep"]))
        if b and r["certified"] and b["certified"]:
            ratios.append(r["seconds"]/b["seconds"])
            objdiff.append(abs(r["objective"]-b["objective"])/max(1,abs(b["objective"])))
    if ratios:
        paired.append(dict(case=case,family=family,solver=solver,mode=mode,frac=frac,
            paired_replications=len(ratios),alternative_over_ours=st.median(ratios),
            min_ratio=min(ratios),max_ratio=max(ratios),max_relative_objective_difference=max(objdiff)))
with open(out/"paired-ratios.csv","w",newline="") as f:
    if paired:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
memory=[]
for file in sorted((root/"memory").glob("*-*/*/memory.*")):
    if file.suffix==".json":rs=[json.loads(file.read_text())]
    elif file.suffix==".csv":rs=list(csv.DictReader(open(file)))
    else:continue
    for r in rs:
        memory.append(dict(case=r["case"],rep=r["rep"],solver=r["solver"],
             rss_peak_mib=float(r.get("rss_peak_kb","nan"))/1024,
             rss_baseline_mib=float(r.get("rss_baseline_kb","nan"))/1024,
             status=r["status"]))
if memory:
    with open(out/"memory.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(memory[0]));w.writeheader();w.writerows(memory)
statuses=Counter(f"{r['solver']}:{r['status']}" for r in rows if r["mode"]=="fit")
profiles=sorted((root/"main").glob("*-*/*/profile-*.txt"))
summary=dict(timing_rows=len(rows),fit_rows=sum(r["mode"]=="fit" for r in rows),
             independent_input_replications=sorted({r["rep"] for r in rows}),
             cases=sorted({r["case"] for r in rows}),
             statuses=dict(statuses),profile_reports=len(profiles),memory_observations=len(memory))
(out/"summary.json").write_text(json.dumps(summary,indent=2))
report=[
    "# Discovery optimization comparison",
    "",
    "Five independent shared inputs per case; three timing rounds per input. Entries below are the median of the five within-input timing medians. All workers use one CPU and one BLAS thread. MATLAB is excluded.",
    "",
    "The main table uses fixed rank 3 or fixed penalty 0.1 times the shared objective-specific lambda-max. A timing is eligible for an accuracy-matched comparison only when all three rounds passed and all five problem replications are present. Nonconvex stationary points can have different objectives; consult paired-ratios.csv before interpreting speed ratios.",
    "",
    "| Case | Family | Solver | Median ms | Replication range ms | Certified inputs |",
    "|---|---|---|---:|---:|---:|"]
for r in aggregated:
    if r["mode"]=="fit" and (r["frac"]==.1 or r["family"]=="rank"):
        report.append(f"| {r['case']} | {r['family']} | {r['solver']} | {r['median_ms']:.4g} | {r['min_rep_ms']:.4g}–{r['max_rep_ms']:.4g} | {r['certified_replications']}/5 |")
report.extend(["","## Coverage","",f"- Timing observations: {len(rows)}.",f"- Separate profile reports: {len(profiles)}.",
               f"- Fresh-process memory observations: {len(memory)}.",
               "- Main accuracy thresholds: relative primal-dual gap and normalized proximal-gradient residual <=1e-6 for convex nuclear regression; objective/rank checks for fixed rank; actual-factor block residuals <=1e-6 for sparse fits.",
               "- Native stopping tolerances are calibrated to meet these checks. Generic Python/JuliaFirstOrder methods check a proximal residual every10 iterations; those monitor costs are included in fit time. Final independent certificates are outside the fit timer.",
               "- Sparse API factors are retained. For the original Julia API, its frozen outer loop is replayed outside the fit timer to recover factors and verify coefficient equality; this prevents false certificate failures from arbitrary SVD null-space completions.",
               "- Peak RSS is total process memory, including interpreter/packages and warm-up; the separate memory table also gives the post-warm-up baseline. Ordinary worker cumulative RSS is not attributed to individual solvers.",
               "- Python cProfile attributes native SVD/BLAS elapsed time to its Python call boundary; Rprof identifies .Call boundaries. Their percentages are not directly comparable to Julia sampling percentages.",
               "- Raw inputs and fitted coefficients remain in the run directory for recreation and independent checking. Tables, profiles, exact versions, and solver source are retained with this report."])
(out/"README.md").write_text("\n".join(report)+"\n")
print(json.dumps(summary,indent=2))
