"""Compact reporting from completed observations; execute under Slurm."""
import csv
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

root = Path("results")
out = root / "summary"
read = lambda p: list(csv.DictReader(open(p)))

ratios = read(out / "paired-ratios.csv")
fit = [r for r in ratios if r["mode"] == "fit"]
metrics = {}
for family in ("nuclear", "sparse", "component", "rank"):
    for solver in sorted({r["solver"] for r in fit if r["family"] == family}):
        rs = [r for r in fit if r["family"] == family and r["solver"] == solver]
        mid = [r for r in rs if float(r["frac"]) in (0, .1)]
        metrics[solver] = dict(
            family=family,
            middle_penalty=[dict(case=r["case"], ratio=float(r["alternative_over_ours"]),
                                min_ratio=float(r["min_ratio"]), max_ratio=float(r["max_ratio"]),
                                objective_difference=float(r["max_relative_objective_difference"])) for r in mid],
            all_penalties_ratio_range=[min(float(r["alternative_over_ours"]) for r in rs),
                                       max(float(r["alternative_over_ours"]) for r in rs)],
            max_relative_objective_difference=max(float(r["max_relative_objective_difference"]) for r in rs))

memory = read(out / "memory.csv")
mem = []
for case in ("responses", "wide"):
    for solver in sorted({r["solver"] for r in memory if r["case"] == case}):
        rs = [r for r in memory if r["case"] == case and r["solver"] == solver]
        assert len(rs) == 5
        peak = [float(r["rss_peak_mib"]) for r in rs]
        baseline = [float(r["rss_baseline_mib"]) for r in rs]
        mem.append(dict(case=case, solver=solver, median_peak_mib=st.median(peak),
                        min_peak_mib=min(peak), max_peak_mib=max(peak),
                        median_baseline_mib=st.median(baseline)))

adaptive = read(root / "adaptive/all-timings.csv")
ad = []
for frac in (.5, .1, .02):
    times = {}
    objectives = {}
    for solver in ("cd_sparse_adaptive", "r_srrr_adaptive"):
        times[solver] = [st.median(float(r["seconds"]) for r in adaptive
                         if r["solver"] == solver and float(r["frac"]) == frac and int(r["rep"]) == rep)
                         for rep in range(1, 6)]
        objectives[solver] = [st.median(float(r["objective"]) for r in adaptive
                              if r["solver"] == solver and float(r["frac"]) == frac and int(r["rep"]) == rep)
                              for rep in range(1, 6)]
    a, b = times.values()
    oa, ob = objectives.values()
    ad.append(dict(frac=frac, julia_median_ms=1000*st.median(a), r_median_ms=1000*st.median(b),
                   r_over_julia=st.median(y/x for x,y in zip(a,b)),
                   max_relative_objective_difference=max(abs(x-y)/max(1,abs(x)) for x,y in zip(oa,ob))))

profiles = sorted((root / "main").glob("*-*/*/profile-*.txt"))
profiles += sorted((root / "adaptive").glob("*-*/*/profile-*.txt"))
profiles += sorted((root / "rank-core").glob("*-*/*/profile-*.txt"))
profiles += sorted((root / "rank-matched").glob("*-*/*/profile-*.txt"))
assert len(profiles) == 290
hotspots = []
for path in profiles:
    body = path.read_text()
    case, rep = path.parent.parent.name.rsplit("-", 1)
    solver = path.stem.removeprefix("profile-")
    if path.parent.name == "julia":
        total = re.search(r"Total snapshots: (\d+)", body)
        count = int(total[1]) if total else 0
        leaves = Counter()
        for line in body.splitlines():
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.+)", line)
            if m:
                leaves[f"{m[3]}:{m[4]} {m[5]}"] += int(m[2])
        top = leaves.most_common(3)
        metric = dict(tool="Julia Profile", samples=count,
                      hotspots="; ".join(f"{v}/{count} {k}" for k,v in top))
    elif path.parent.name == "python":
        total = re.search(r"in ([\d.]+) seconds", body)
        entries = []
        for line in body.splitlines():
            m = re.match(r"\s*[\d/]+\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+(.+)", line)
            if m:
                entries.append((float(m[1]), float(m[2]), m[3]))
        top = sorted(entries, reverse=True)[:3]
        metric = dict(tool="Python cProfile", samples="",
                      hotspots="; ".join(f"self={a}s cumulative={b}s {c}" for a,b,c in top))
        metric["profile_seconds"] = float(total[1]) if total else None
    else:
        section = body.split("$by.total")[0]
        entries = []
        for line in section.splitlines():
            m = re.match(r'"(.+)"\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                entries.append((float(m[3]), m[1]))
        metric = dict(tool="Rprof", samples="",
                      hotspots="; ".join(f"self={v}% {name}" for v,name in sorted(entries, reverse=True)[:3]))
    hotspots.append(dict(path=str(path), case=case, rep=int(rep), solver=solver, **metric))

def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

write_csv(out / "memory-aggregate.csv", mem)
write_csv(out / "adaptive-aggregate.csv", ad)
write_csv(out / "profile-index.csv", hotspots)

core_rows = read(out / "all-timings.csv")
supplement = read(root / "rank-core/all-timings.csv")
supplement += read(root / "rank-matched/all-timings.csv")
checks = read(root / "main/independent-certificates.csv")
checks += read(root / "adaptive/independent-certificates.csv")
checks += read(root / "rank-core/independent-certificates.csv")
checks += read(root / "rank-matched/independent-certificates.csv")
result = dict(timing_rows=len(core_rows)+len(adaptive)+len(supplement), profiles=len(profiles),
              memory_observations=len(memory), coefficient_checks=len(checks),
              coefficient_passes=sum(r["passed"] == "True" for r in checks),
              failed_checks=[r for r in checks if r["passed"] != "True"],
              native_flags=dict(Counter((r["solver"]+":"+r["native_converged"]) for r in core_rows if r["mode"] == "fit")),
              paired_comparisons=metrics)
assert result["timing_rows"] == 5460
assert result["coefficient_checks"] == 1400 and result["coefficient_passes"] == 1395
(out / "final-summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != "paired_comparisons"}, indent=2))
