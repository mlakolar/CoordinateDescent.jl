import csv
import json
from pathlib import Path
from common import *
root=Path("results/adaptive")
checks=[];rows=[]
for rep in range(1,6):
    d=load_problem(f"inputs/medium-{rep}")
    X,Y,_,_,_=prepare(d)
    w=load_array(f"inputs/medium-{rep}/adaptive_weights.bin",(50,))
    maximum=float(Path(f"inputs/medium-{rep}/adaptive_max.txt").read_text())
    for language,solver in (("julia","cd_sparse_adaptive"),("r","r_srrr_adaptive")):
        folder=root/f"medium-{rep}"/language
        rows.extend(csv.DictReader(open(folder/"adaptive.csv")))
        for file in folder.glob("*.bin"):
            frac=float(file.stem.rsplit("-",1)[1])
            B=load_array(file,(50,100))
            A=load_array(file.with_suffix(".A"),(50,3));V=load_array(file.with_suffix(".V"),(100,3))
            assert np.linalg.norm(A@V.T-B)<1e-7*max(1,np.linalg.norm(B))
            obj,gap,res=sparse_certificate(X,Y,A,V,frac*maximum,w)
            checks.append(dict(rep=rep,solver=solver,frac=frac,objective=obj,residual=res,passed=bool(res<=1e-6)))
with open(root/"independent-certificates.csv","w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
with open(root/"all-timings.csv","w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(timing_rows=len(rows),coefficient_checks=len(checks),certificate_passes=int(sum(c["passed"] for c in checks)),
            profile_reports=len(list(root.glob("*-*/*/profile-*.txt"))))
(root/"summary.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert result==dict(timing_rows=90,coefficient_checks=30,certificate_passes=30,profile_reports=10)
