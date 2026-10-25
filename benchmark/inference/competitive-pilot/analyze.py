"""Aggregate pilot outputs and costs inside the Discovery allocation."""
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

root = Path.cwd()
cases = {c["scenario"]:c for c in json.loads((root/"cases.json").read_text())}
tasks = [x.split("\t") for x in (root / "tasks.tsv").read_text().splitlines()]
records = []
coefficient_records = []
for engine, mode, scenario, rep in tasks:
    tag = "-".join((engine, mode, scenario, rep))
    base = root / "outputs" / tag
    rec = dict(engine=engine, mode=mode, scenario=scenario, rep=int(rep), status="missing")
    for suffix in (".meta", ".time", ".failure"):
        path = Path(str(base) + suffix)
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1); rec[k] = v
    j = Path(str(base) + ".json")
    if j.exists(): rec.update(json.loads(j.read_text()))
    log = root / "logs" / (tag + ".log")
    if log.exists():
        s = log.read_text()
        rec["sihr_fallback_messages"] = s.count("Alternative method is applied for proj direction")
        rec["sihr_caught_warning_messages"] = s.count("Caught a warning using CVXR")
        rec["probability_filter_messages"] = s.count("More than 20 % observations are filtered out")
        rec["sslasso_noise_warning_messages"] = s.count("Warning: Noise estimate problematic")
    p = Path(str(base) + ".csv")
    if rec["status"] == "ok" and p.exists():
        vals = list(csv.DictReader(open(p)))
        truth = np.loadtxt(root / "data" / scenario / rep / "beta.csv", delimiter=",")
        expected_targets = np.loadtxt(root / "data" / scenario / rep / "targets.csv",dtype=int).tolist()
        if [int(float(v["target"])) for v in vals] != expected_targets:
            rec["status"] = "invalid_targets"
            records.append(rec)
            continue
        active, null = [], []
        for v in vals:
            j = int(float(v["target"]))-1
            lo, hi, pv = (float(v[k]) for k in ("lower", "upper", "pvalue"))
            if not all(math.isfinite(t) for t in (lo, hi, pv)) or not 0 <= pv <= 1 or lo > hi:
                rec["status"] = "invalid_output"; break
            (active if truth[j] != 0 else null).append((lo <= truth[j] <= hi, pv <= .05, hi-lo))
            coefficient_records.append(dict(engine=engine,mode=mode,scenario=scenario,rep=int(rep),
                                             truth=float(truth[j]),**v))
        if rec["status"] == "ok":
            for group, vv in (("active",active),("null",null)):
                a = np.array(vv)
                rec[group+"_coverage"] = float(a[:,0].mean())
                rec[group+"_rejection"] = float(a[:,1].mean())
                rec[group+"_length"] = float(a[:,2].mean())
            rec["n_targets"] = len(vals)
            if isinstance(rec.get("warnings"), list):
                rec["warnings"] = " | ".join(rec["warnings"])
    records.append(rec)
if coefficient_records:
    with open("pilot-coefficients.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(coefficient_records[0]));w.writeheader();w.writerows(coefficient_records)
fields = sorted(set().union(*(r.keys() for r in records)))
with open("pilot-records.csv", "w", newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)

groups = defaultdict(list)
for r in records: groups[r["scenario"],r["engine"],r["mode"]].append(r)
summary=[]
for key, rows in sorted(groups.items()):
    good=[r for r in rows if r["status"]=="ok"]
    row=dict(zip(("scenario","engine","mode"),key))
    row.update(expected=len(rows),successful=len(good),unsupported=sum(r["status"]=="unsupported" for r in rows),
               missing=sum(r["status"]=="missing" for r in rows),
               warning_fits=sum(bool(r.get("warnings")) for r in good),
               failed=sum(r["status"] not in ("ok","unsupported","missing") for r in rows))
    row["qp_gap_warning_fits"] = sum(float(r.get("qp_max_abs_gap_all",r.get("qp_max_abs_gap",0))) > 1e-3 for r in good)
    row["convergence_warning_fits"] = sum(
        "ConvergenceWarning" in r.get("warning_categories",[]) or
        ("converg" in str(r.get("warnings","")).lower()) for r in good)
    row["sihr_matched_mu_enlarged_fits"] = sum(
        r["engine"] == "SIHR" and r["mode"] == "matched" and
        float(r.get("projection_mu_max",0)) >
            float(cases[r["scenario"]]["lambda_node"]) * (1+1e-6)
        for r in good)
    for metric in ("active_coverage","null_coverage","active_rejection","null_rejection","active_length","null_length",
                   "fit_seconds","process_seconds","max_rss_kib","cpu_seconds",
                   "qp_max_residual","qp_max_abs_gap","qp_max_residual_all","qp_max_abs_gap_all","projection_mu_min","projection_mu_max",
                   "projection_solver_calls","sihr_fallback_messages","sihr_caught_warning_messages",
                   "probability_filter_messages","sslasso_noise_warning_messages"):
        values=[float(r[metric]) for r in good if metric in r]
        if values:
            row[metric+"_mean"]=st.mean(values)
            row[metric+"_median"]=st.median(values)
            row[metric+"_min"]=min(values); row[metric+"_max"]=max(values)
            row[metric+"_mcse"]=st.stdev(values)/math.sqrt(len(values)) if len(values)>1 else math.nan
    if good:
        row["full1000_fit_cpu_hours"]=row["fit_seconds_mean"]*1000/3600
        row["full1000_process_cpu_hours"]=row["process_seconds_mean"]*1000/3600
    summary.append(row)
fields=sorted(set().union(*(r.keys() for r in summary)))
with open("pilot-summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)

def fmt(v,digits=3): return f"{v:.{digits}f}" if isinstance(v,(int,float)) else str(v)
lines=["# Discovery comparison pilot", "", "20 fresh datasets per case; five active and five null targets; one CPU per worker.",
       "Results are exploratory. Coefficients within a dataset are dependent; Monte Carlo SEs use dataset-level means.", "",
       "| Case | Engine | Tuning | Successful | Coverage active/null | Type I | Power | Length active/null | Fit seconds median | Peak RSS MiB max |", 
       "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in summary:
    lines.append("| "+" | ".join([r["scenario"],r["engine"],r["mode"],f'{r["successful"]}/{r["expected"]}',
       "/".join(fmt(r.get(k,math.nan)) for k in ("active_coverage_mean","null_coverage_mean")),
       fmt(r.get("null_rejection_mean",math.nan)),fmt(r.get("active_rejection_mean",math.nan)),
       "/".join(fmt(r.get(k,math.nan)) for k in ("active_length_mean","null_length_mean")),
       fmt(r.get("fit_seconds_median",math.nan)),fmt(r.get("max_rss_kib_max",math.nan)/1024,1)])+" |")
lines += ["", "## Numerical diagnostics and API warnings", "",
          "| Case | Engine | Tuning | Failed | Warning fits | Convergence warnings | QP gap warnings | SIHR enlarged supplied mu | SIHR fallback messages |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
for r in summary:
    flags = [r["failed"],r["warning_fits"],r["convergence_warning_fits"],r["qp_gap_warning_fits"],r["sihr_matched_mu_enlarged_fits"],
             r.get("sihr_fallback_messages_mean",0)*r["successful"]]
    if any(flags):
        lines.append("| "+" | ".join([r["scenario"],r["engine"],r["mode"]]+[str(round(v)) for v in flags])+" |")
lines += ["", "EconML emits an API warning for a manually specified alpha; this is separate from a numerical convergence warning. Monte Carlo SEs, min/max timings, solver mus, QP residuals/gaps and all per-fit records are in pilot-summary.csv and pilot-records.csv."]
fit_hours=sum(r.get("full1000_fit_cpu_hours",0) for r in summary)
proc_hours=sum(r.get("full1000_process_cpu_hours",0) for r in summary)
peak=max((r.get("max_rss_kib_max",0) for r in summary),default=0)/1024
complete=all(r["successful"]+r["unsupported"]==r["expected"] for r in summary)
cost = dict(complete=complete,replications_per_case=20,case_count=len(cases),targets_per_case=10,
            full_replications_per_case=1000,fit_cpu_hours=fit_hours,
            fresh_process_cpu_hours=proc_hours,ideal_fit_wall_hours_8cpu=fit_hours/8,
            ideal_process_wall_hours_8cpu=proc_hours/8,
            fresh_process_budget_wall_hours_8cpu_25pct_margin=math.ceil(proc_hours/8*1.25),
            maximum_successful_process_rss_mib=peak,recommended_cpus=8,recommended_memory_gib=16)
cost["all_completed_supported_attempts_process_cpu_hours_scaled50"]=sum(
    float(r.get("process_seconds",0)) for r in records if r["status"] not in ("unsupported","missing"))*50/3600
bundles=defaultdict(lambda:defaultdict(float))
for r in records:
    if r["status"]=="ok":
        for k in ("fit_seconds","process_seconds","cpu_seconds"):
            if k in r:bundles[r["rep"]][k]+=float(r[k])
if complete and len(bundles)==20:
    for k in ("fit_seconds","process_seconds","cpu_seconds"):
        v=[x[k] for x in bundles.values()]
        cost[k+"_full1000_cpu_hours_mcse"]=st.stdev(v)/math.sqrt(len(v))*1000/3600
cost["data_bytes_pilot"]=sum(p.stat().st_size for p in (root/"data").glob("*/*/*.csv"))
cost["data_gib_full1000_if_csv_inputs_retained"]=cost["data_bytes_pilot"]*50/2**30
Path("cost-estimate.json").write_text(json.dumps(cost,indent=2)+"\n")

# Matched Gaussian nodewise implementations should agree when scalar penalties,
# scaling, and residual-DF noise estimation agree. This is a check on the adapter,
# not an assumption about the other projection/variance methods.
lookup=defaultdict(dict)
for r in coefficient_records:
    if r["scenario"].startswith("gaussian") and r["mode"]=="matched" and r["engine"] in ("ours","hdi"):
        lookup[(r["scenario"],r["rep"],int(float(r["target"])) )][r["engine"]]=r
parity=[]
for key,row in lookup.items():
    if set(row)=={"ours","hdi"}:
        parity.append(dict(scenario=key[0],rep=key[1],target=key[2],
                           estimate_abs_difference=abs(float(row["ours"]["estimate"])-float(row["hdi"]["estimate"])),
                           se_abs_difference=abs(float(row["ours"]["se"])-float(row["hdi"]["se"]))))
if parity:
    with open("matched-nodewise-check.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(parity[0]));w.writeheader();w.writerows(parity)
    cost["matched_nodewise_max_estimate_difference"]=max(r["estimate_abs_difference"] for r in parity)
    cost["matched_nodewise_max_se_difference"]=max(r["se_abs_difference"] for r in parity)
    Path("cost-estimate.json").write_text(json.dumps(cost,indent=2)+"\n")
lines += ["", "## Conditional cost estimate", "",
          f"Sum over measured supported configurations, 1000 datasets per case: {fit_hours:.3f} CPU-hours of warmed fitting; {proc_hours:.3f} CPU-hours with a fresh process and warm-up per fit.",
          f"Ideal eight-worker elapsed lower bounds: {fit_hours/8:.3f} to {proc_hours/8:.3f} hours, before scheduling/imbalance margin.",
          f"Largest successful per-process RSS: {peak:.1f} MiB. Aggregate RSS must account for all concurrent workers and overhead.",
          f"Conservative fresh-process planning budget with 25% margin: {cost['fresh_process_budget_wall_hours_8cpu_25pct_margin']} allocated hours at 8 CPUs and 16 GiB. Queue time is excluded. This is for the four pilot cases and ten requested targets, not all-p inference or a broader design grid.",
          f"Retaining all CSV inputs at 1000 replications would use about {cost['data_gib_full1000_if_csv_inputs_retained']:.2f} GiB; deterministic generation permits a bounded working set with seed/input-hash records instead.",
          f"Every requested supported fit successful: {complete}. Missing/failed configurations invalidate a complete-campaign estimate; their costs are lower bounds.",
          "", "## Comparability notes", "",
          "- Marginal 95% intervals and unadjusted two-sided 5% tests. No simultaneous/bootstrap comparison in this pilot.",
          "- Native defaults retain package statistical choices. RNG seeds and single-CPU execution are fixed. Requested targets are five active and five null coefficients.",
          "- Julia and SIHR compute requested targets; hdi, EconML and sslasso compute all 500 slopes internally.",
          "- Julia GLM needs an explicit initial lambda: the declared common lambda is supplied; this is a required-input default, not fully automatic tuning.",
          "- Matched nominal initial and projection penalties are common within case; variance estimators and projection objectives remain method-specific. SIHR matched uses rescale=1 and prob.filter=0.",
          "- Native hdi retains default group-test preparation. Matched hdi disables this preparation for the requested individual-coefficient task. Replicated profiling measures this extra preparation separately.",
          "- Native hdi binomial preprocessing does not expose a matched-initial-Fisher hook. Its matched GLM rows are explicitly unsupported, not fabricated.",
          "- sslasso matched repairs the authors' undefined Las helper using exact-lambda glmnet and corrects the diagonal QP fast path from (1-mu0)/G_ii to (1-mu)/G_ii. The iterative solver, noise estimator, interval inflation and testing are retained. Original defaults and superseded matched outputs are preserved.",
          "- Small independent warm-ups precede timing. Peak RSS and process time include runtime/library initialization, warm-up and actual fitting. Cold startup is process time minus warmed fitting only approximately.",
          "- Default author sslasso has a coarse convergence rule; matching nominal penalties does not establish numerical equivalence."]
lines += ["- SIHR can enlarge a supplied mu and can fall back to a diagonal projection after an internal solver error. Instrumentation for replications 2--20 records actual solver mus/statuses without changing outputs; fallback messages are counted.",
          "- sslasso matched post-timing constraint residuals and primal-dual gaps are recorded. Finite coefficient output alone does not establish QP optimality.",
          "- Authors' sslasso intervals include an additional bias allowance that their returned normal p-values do not include; those intervals and tests are not exact duals. Reported coverage and rejection rates use the respective original outputs.",
          "- Coverage, error, power and length summaries condition on successful finite output. Failure and warning counts must accompany comparisons; failed fits are not silently counted as valid inference."]
if (root/"profile-summary.json").exists():
    prof=json.loads((root/"profile-summary.json").read_text())
    lines += ["", "## Replicated profiling", "",
              "Five independent Gaussian-moderate datasets were profiled at 10 ms intervals under native hdi defaults on Discovery. Every profiled coefficient and SE matched the same-seed timed result to 1e-8.",
              f"Median runtime share in nodewise regression: {prof['calculate.Z']['median_percent']:.2f}% (range {prof['calculate.Z']['min_percent']:.2f}--{prof['calculate.Z']['max_percent']:.2f}%).",
              f"Its custom nodewise cross-validation accounted for {prof['cv.nodewise.bestlambda']['median_percent']:.2f}% of total runtime; optional group-test preparation accounted for {prof['preprocess.group.testing']['median_percent']:.2f}%.",
              "Full per-replication stacks and self/total summaries remain in the run's profiles directory; profile-summary.json and profile-totals.csv retain the reporting inputs."]
Path("PILOT_RESULTS.md").write_text("\n".join(lines)+"\n")
print(json.dumps(dict(tasks=len(records),status_counts={s:sum(r["status"]==s for r in records) for s in sorted(set(r["status"] for r in records))},
                      successful_supported_cost_fit_cpu_hours=fit_hours, successful_supported_cost_process_cpu_hours=proc_hours,
                      max_successful_rss_mib=peak,complete=complete),indent=2))
