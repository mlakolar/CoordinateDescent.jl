"""Aggregate completed five-replication measurements and independent checks."""
import csv
import os
import json
import statistics as st
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from common import load_problem, load_array, certificate, prepare, sparse_certificate

root=Path("results");out=root/"summary";out.mkdir(exist_ok=True)
rows=[];sources=[]
for cohort,folder in (("after","main"),("before","before"),("adaptive","adaptive")):
    for file in sorted((root/folder).glob("*-*/*/*.csv")):
        if file.name not in ("julia.csv","python.csv","r.csv","r-component.csv"):continue
        data=list(csv.DictReader(open(file)))
        for row in data:
            row.update(cohort=cohort)
            for key in ("seconds","objective","gap","residual","frac"):
                row[key]=float(row[key])
            row["rep"]=int(row["rep"]);row["round"]=int(row["round"])
        rows.extend(data);sources.append((cohort,file,data))

def write_csv(path,data):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n")
        w.writeheader();w.writerows(data)

write_csv(out/"all-timings.csv",rows)
keys=[(r["cohort"],r["case"],r["rep"],r["solver"],r["mode"],r["frac"],r["round"]) for r in rows]
assert len(keys)==len(set(keys))==7110,(len(keys),len(set(keys)))
grouped=defaultdict(list)
for r in rows:grouped[(r["cohort"],r["case"],r["family"],r["solver"],r["mode"],r["frac"],r["rep"])].append(r)
reps=[]
for (cohort,case,family,solver,mode,frac,rep),rs in sorted(grouped.items()):
    assert len(rs)==3
    reps.append(dict(cohort=cohort,case=case,family=family,solver=solver,mode=mode,frac=frac,rep=rep,
        median_seconds=st.median(r["seconds"] for r in rs),objective=st.median(r["objective"] for r in rs),
        certified=all(r["status"]=="passed" for r in rs),native_flags="/".join(sorted({r["native_converged"] for r in rs}))))
write_csv(out/"replication-medians.csv",reps)
groups=defaultdict(list)
for r in reps:groups[tuple(r[k] for k in ("cohort","case","family","solver","mode","frac"))].append(r)
aggregated=[]
for key,rs in sorted(groups.items()):
    assert {r["rep"] for r in rs}=={1,2,3,4,5}
    cohort,case,family,solver,mode,frac=key;ts=[1000*r["median_seconds"] for r in rs]
    aggregated.append(dict(cohort=cohort,case=case,family=family,solver=solver,mode=mode,frac=frac,
        median_ms=st.median(ts),min_rep_ms=min(ts),max_rep_ms=max(ts),
        certified_replications=sum(r["certified"] for r in rs)))
write_csv(out/"aggregate.csv",aggregated)
lookup={(r["cohort"],r["case"],r["solver"],r["mode"],r["frac"],r["rep"]):r for r in reps}
paired=[]
refs={"rank":"cd_rank","nuclear":"cd_nuclear","sparse":"cd_sparse_api","component":"cd_component"}
for (cohort,case,family,solver,mode,frac),rs in sorted(groups.items()):
    if cohort=="after":
        comparisons=[] if solver=="cd_rankpath" else [("after",refs[family])]
        if solver in ("cd_rank","cd_rankpath","cd_nuclear","cd_sparse_api"):
            comparisons.append(("before",{"cd_rank":"before_rank","cd_rankpath":"before_rankpath",
                "cd_nuclear":"before_nuclear","cd_sparse_api":"before_sparse"}[solver]))
    elif cohort=="adaptive":comparisons=[("adaptive","cd_sparse_adaptive_cd")]
    else:continue
    for refcohort,ref in comparisons:
        if solver==ref:continue
        ratios=[];differences=[]
        for r in rs:
            a=lookup.get((refcohort,case,ref,mode,frac,r["rep"]))
            if a and a["certified"] and r["certified"]:
                ratios.append(r["median_seconds"]/a["median_seconds"])
                differences.append(abs(r["objective"]-a["objective"])/max(1,abs(a["objective"])))
        if ratios:
            paired.append(dict(cohort=cohort,case=case,family=family,solver=solver,mode=mode,frac=frac,
                reference=ref,reference_cohort=refcohort,paired_replications=len(ratios),
                solver_over_reference=st.median(ratios),min_ratio=min(ratios),max_ratio=max(ratios),
                max_relative_objective_difference=max(differences)))
write_csv(out/"paired-ratios.csv",paired)

checks=[]
for cohort,file,data in sources:
    specs={}
    for r in data:
        if r["mode"]=="fit":specs[(r["solver"],r["frac"])]=r["family"]
    d=load_problem(Path("inputs")/file.parent.parent.name)
    for (solver,frac),family in specs.items():
        fits=[p for p in file.parent.glob(solver+"-*.bin") if float(p.stem.rsplit("-",1)[1])==frac]
        assert len(fits)==1,(file,solver,frac)
        p=fits[0];m=d["meta"];B=load_array(p,(m["p"],m["q"]))
        A=V=None
        if family in ("sparse","sparse_adaptive"):
            A=load_array(p.with_suffix(".A"),(m["p"],m["r"]))
            V=load_array(p.with_suffix(".V"),(m["q"],m["r"]))
            assert np.linalg.norm(A@V.T-B)<=1e-7*max(1,np.linalg.norm(B))
        if family=="sparse_adaptive":
            X,Y,*_=prepare(d)
            w=load_array(Path("inputs")/file.parent.parent.name/"adaptive_weights.bin",(m["p"],))
            lm=float((Path("inputs")/file.parent.parent.name/"adaptive_max.txt").read_text())
            obj,gap,res=sparse_certificate(X,Y,A,V,frac*lm,w)
        else:obj,gap,res=certificate(d,family,B,frac,A,V)
        checks.append(dict(cohort=cohort,case=file.parent.parent.name,solver=solver,frac=frac,
            objective=obj,gap=gap,residual=res,passed=bool(res<=1e-6 and (np.isnan(gap) or gap<=1e-6))))
write_csv(out/"independent-certificates.csv",checks)
assert len(checks)==1860,len(checks)
failures=[r for r in checks if not r["passed"]]
assert len(failures)==5 and all(r["solver"]=="r_rrr" and r["case"].startswith("ill-") for r in failures),failures

memory=[]
if os.environ.get("CD81_CORE_ONLY")!="1":
    memory=[]
    for file in sorted((root/"memory").glob("*-*/*/memory.*")):
        data=[json.loads(file.read_text())] if file.suffix==".json" else list(csv.DictReader(open(file)))
        for r in data:
            memory.append(dict(case=r["case"],rep=int(r["rep"]),solver=r["solver"],
                peak_mib=float(r["rss_peak_kb"])/1024,baseline_mib=float(r["rss_baseline_kb"])/1024,status=r["status"]))
    assert len(memory)==230,len(memory)
    write_csv(out/"memory.csv",memory)
    mem=[]
    for key in sorted({(r["case"],r["solver"]) for r in memory}):
        rs=[r for r in memory if (r["case"],r["solver"])==key];assert len(rs)==5
        mem.append(dict(case=key[0],solver=key[1],median_peak_mib=st.median(r["peak_mib"] for r in rs),
            min_peak_mib=min(r["peak_mib"] for r in rs),max_peak_mib=max(r["peak_mib"] for r in rs),
            median_baseline_mib=st.median(r["baseline_mib"] for r in rs)))
    write_csv(out/"memory-aggregate.csv",mem)

profiles=[]
for cohort,folder in (("after","main"),("before","before"),("adaptive","adaptive")):
    for p in sorted((root/folder).glob("*-*/*/profile-*.txt")):
        assert p.stat().st_size>0
        body=p.read_text();hot=[];samples=""
        if p.parent.name=="julia":
            match=re.search(r"Total snapshots: (\d+)",body);samples=int(match[1]) if match else 0
            leaves=Counter()
            for line in body.splitlines():
                match=re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(.+)",line)
                if match:leaves[f"{match[3]}:{match[4]} {match[5]}"]+=int(match[2])
            hot=[f"{v}/{samples} {k}" for k,v in leaves.most_common(3)]
        elif p.parent.name=="python":
            entries=[]
            for line in body.splitlines():
                match=re.match(r"\s*[\d/]+\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+(.+)",line)
                if match:entries.append((float(match[1]),float(match[2]),match[3]))
            hot=[f"self={a}s cumulative={b}s {c}" for a,b,c in sorted(entries,reverse=True)[:3]]
        else:
            entries=[]
            for line in body.split("$by.total")[0].splitlines():
                match=re.match(r'"(.+)"\s+([\d.]+)\s+([\d.]+)',line)
                if match:entries.append((float(match[3]),match[1]))
            hot=[f"self={v}% {name}" for v,name in sorted(entries,reverse=True)[:3]]
        profiles.append(dict(cohort=cohort,case=p.parent.parent.name,language=p.parent.name,
            solver=p.stem.removeprefix("profile-"),path=str(p),samples=samples,hotspots="; ".join(hot)))
assert len(profiles)==430,len(profiles)
write_csv(out/"profile-index.csv",profiles)
summary=dict(timing_rows=len(rows),core_timing_rows=sum(r["cohort"]=="after" for r in rows),
    before_timing_rows=sum(r["cohort"]=="before" for r in rows),adaptive_timing_rows=sum(r["cohort"]=="adaptive" for r in rows),
    profiles=len(profiles),memory_observations=len(memory),coefficient_checks=len(checks),coefficient_passes=len(checks)-len(failures),
    failures=failures,statuses=dict(Counter(r["solver"]+":"+r["status"] for r in rows)),
    native_flags=dict(Counter(r["solver"]+":"+r["native_converged"] for r in rows)))
(out/"summary.json").write_text(json.dumps(summary,indent=2))
report=["# Measured timing tables","","Five independent inputs; median of three timing rounds per input. Times in milliseconds.","",
    "| Cohort | Case | Solver | Mode | Median ms | Replication range ms | Certified |",
    "|---|---|---|---|---:|---:|---:|"]
for r in aggregated:
    if r["mode"] in ("fit","path") and (r["family"]=="rank" or r["frac"] in (.1,.02) and r["mode"]=="path" or r["frac"]==.1):
        report.append(f"| {r['cohort']} | {r['case']} | {r['solver']} | {r['mode']} | {r['median_ms']:.4g} | {r['min_rep_ms']:.4g}–{r['max_rep_ms']:.4g} | {r['certified_replications']}/5 |")
(out/"full-table.md").write_text("\n".join(report)+"\n")
print(json.dumps(summary,indent=2))
