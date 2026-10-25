"""Independent saved-coefficient verification and small CVXPY reference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy import linalg as la
from common import *

def convex_reference(out):
    rng=np.random.default_rng(810081)
    Z=rng.standard_normal((60,10));Z-=Z.mean(0)
    X=np.sqrt(60)*la.qr(Z,mode="economic")[0]
    Y=rng.standard_normal((60,5));Y-=Y.mean(0)
    C=X.T@Y/60;lam=.2*la.svdvals(C)[0]
    exact=svt(C,lam)
    B=cp.Variable((10,5))
    objective=cp.sum_squares(Y-X@B)/120+lam*cp.norm(B,"nuc")
    problem=cp.Problem(cp.Minimize(objective))
    problem.solve(solver="SCS",eps=1e-9,max_iters=100000)
    cert=nuclear_certificate(X,Y,B.value,lam,1.0)
    error=np.max(np.abs(B.value-exact))
    result=dict(status=problem.status,objective=problem.value,max_coefficient_error=float(error),
                gap=cert[1],residual=cert[2],solver="CVXPY/SCS")
    (Path(out)/"cvxpy-reference.json").write_text(json.dumps(result,indent=2))
    assert error<5e-6 and cert[1]<1e-7 and cert[2]<1e-7,result
    return result

def family(s):
    if s in ("cd_rank","pa_rank","scipy_rrr","r_rrr","r_rrr_qr"):return "rank"
    if s in ("cd_sparse","block_fista","r_srrr"):return "sparse"
    if "component" in s:return "component"
    return "nuclear"

def main():
    parser=argparse.ArgumentParser();parser.add_argument("inputs");parser.add_argument("results")
    parser.add_argument("--cvxpy",action="store_true");args=parser.parse_args()
    out=Path(args.results);out.mkdir(parents=True,exist_ok=True)
    if args.cvxpy:print(json.dumps(convex_reference(out)),flush=True)
    checked=[]
    for file in sorted(out.glob("*-*/*/*.bin")):
        case_id=file.parent.parent.name
        root=Path(args.inputs)/case_id
        if not root.is_dir():continue
        d=load_problem(root)
        solver,frac=file.stem.rsplit("-",1);frac=float(frac);fam=family(solver)
        B=load_array(file,(d["meta"]["p"],d["meta"]["q"]))
        A=V=None
        if fam=="sparse" and file.with_suffix(".A").is_file():
            A=load_array(file.with_suffix(".A"),(d["meta"]["p"],3))
            V=load_array(file.with_suffix(".V"),(d["meta"]["q"],3))
            assert np.linalg.norm(A@V.T-B)<=1e-7*max(1,np.linalg.norm(B))
        obj,gap,res=certificate(d,fam,B,frac,A,V)
        checked.append(dict(case=case_id,solver=solver,family=fam,frac=frac,
                            objective=obj,gap=gap,residual=res,
                            passed=bool(res<=1e-6 and (np.isnan(gap) or gap<=1e-6))))
    with open(out/"independent-certificates.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(checked[0]) if checked else ["case"]);w.writeheader();w.writerows(checked)
    statuses={}
    for file in out.glob("*-*/*/*.csv"):
        if file.name not in ("julia.csv","python.csv","r.csv","r-component.csv"):continue
        for row in csv.DictReader(open(file)):
            key=f"{row['solver']}:{row['status']}"
            statuses[key]=statuses.get(key,0)+1
    result=dict(coefficient_files=len(checked),coefficient_passes=int(sum(r["passed"] for r in checked)),
                statuses=statuses)
    (out/"validation-summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":main()
