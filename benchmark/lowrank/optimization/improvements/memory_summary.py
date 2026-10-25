"""Validate fresh-process memory without recomputing unchanged fit certificates."""
import csv,json,os,statistics as st
from pathlib import Path
out=Path("results/summary")
def write(name,rows):
    with open(out/name,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
rows=[]
for file in sorted(Path("results/memory").glob("*-*/*/memory.*")):
    data=[json.loads(file.read_text())] if file.suffix==".json" else list(csv.DictReader(open(file)))
    for r in data:
        rows.append(dict(case=r["case"],rep=int(r["rep"]),solver=r["solver"],peak_mib=float(r["rss_peak_kb"])/1024,baseline_mib=float(r["rss_baseline_kb"])/1024,status=r["status"]))
expected=int(os.environ.get("CD81_EXPECTED_MEMORY","230"))
assert len(rows)==expected,len(rows)
assert len({(r["case"],r["rep"],r["solver"]) for r in rows})==expected
assert all(r["status"]=="passed" and 0<=r["baseline_mib"]<=r["peak_mib"] for r in rows)
write("memory.csv",rows)
agg=[]
for key in sorted({(r["case"],r["solver"]) for r in rows}):
    rs=[r for r in rows if (r["case"],r["solver"])==key]
    assert {r["rep"] for r in rs}=={1,2,3,4,5}
    agg.append(dict(case=key[0],solver=key[1],median_peak_mib=st.median(r["peak_mib"] for r in rs),min_peak_mib=min(r["peak_mib"] for r in rs),max_peak_mib=max(r["peak_mib"] for r in rs),median_baseline_mib=st.median(r["baseline_mib"] for r in rs)))
assert len(agg)==expected//5
write("memory-aggregate.csv",agg)
s=json.loads((out/"summary.json").read_text());s["memory_observations"]=len(rows)
(out/"summary.json").write_text(json.dumps(s,indent=2)+"\n")
print("Validated",len(rows),"fresh-process memory observations")
