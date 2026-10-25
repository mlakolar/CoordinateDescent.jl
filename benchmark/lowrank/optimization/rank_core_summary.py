import csv
import json
import statistics
from pathlib import Path
from common import *
root=Path("results/rank-core");rows=[];checks=[]
for csvfile in sorted(root.glob("*-*/*/rank-core.csv")):
    rows.extend(csv.DictReader(open(csvfile)))
for file in sorted(root.glob("*-*/*/*.bin")):
    case_id=file.parent.parent.name;d=load_problem(Path("inputs")/case_id)
    B=load_array(file,(d["meta"]["p"],d["meta"]["q"]))
    obj,gap,res=certificate(d,"rank",B,0)
    checks.append(dict(case=case_id,objective=obj,gap=gap,residual=res,passed=bool(gap<=1e-6 and res==0)))
for filename,data in (("all-timings.csv",rows),("independent-certificates.csv",checks)):
    with open(root/filename,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
aggregate=[]
for case in CASES:
    medians=[]
    for rep in range(1,6):
        medians.append(statistics.median(float(r["seconds"]) for r in rows if r["case"]==case and int(r["rep"])==rep))
    aggregate.append(dict(case=case,solver="cd_rank_core",median_ms=1000*statistics.median(medians),
                         min_rep_ms=1000*min(medians),max_rep_ms=1000*max(medians),certified_replications=5))
with open(root/"aggregate.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
result=dict(timing_rows=len(rows),coefficient_checks=len(checks),certificate_passes=int(sum(c["passed"] for c in checks)),
            profile_reports=len(list(root.glob("*-*/*/profile-*.txt"))))
(root/"summary.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert result==dict(timing_rows=90,coefficient_checks=30,certificate_passes=30,profile_reports=15)
