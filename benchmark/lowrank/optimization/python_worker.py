import argparse
import cProfile
import json
import os
import pstats
import resource
import time
from pathlib import Path
import numpy as np
from scipy import linalg as la
import pyproximal as px
from pyproximal.optimization.primal import ProximalGradient, ADMM
from threadpoolctl import threadpool_info, threadpool_limits
from common import *

class StopSolve(Exception):
    def __init__(self, B, iterations, reason):
        self.B, self.iterations, self.reason = B.copy(), iterations, reason

class RegressionLoss(px.ProxOperator):
    """Matrix-free/Gram gradient and lazy thin-SVD least-squares prox."""
    def __init__(self, X, Y, H, C):
        super().__init__(hasgrad=True)
        self.X, self.Y, self.gram, self.C = X,Y,H,C
        self.matrix_shape = C.shape
        self.factors = None
    def __call__(self, x):
        B=x.reshape(self.matrix_shape)
        return np.sum((self.X@B-self.Y)**2)/(2*len(self.X))
    def grad(self, x):
        B=x.reshape(self.matrix_shape)
        g = self.gram@B-self.C if self.gram is not None else self.X.T@(self.X@B)/len(self.X)-self.C
        return g.ravel()
    def prox(self, x, tau):
        if self.factors is None:
            U,s,Vt=la.svd(self.X/np.sqrt(len(self.X)),full_matrices=False)
            self.factors=(s*s,Vt)
        s2,Vt=self.factors
        B=x.reshape(self.matrix_shape)
        rhs=B+tau*self.C
        # (I+tau X'X/n)^-1 without a p*q Kronecker matrix.
        ans=rhs-Vt.T@((tau*s2/(1+tau*s2))[:,None]*(Vt@rhs))
        return ans.ravel()

def make_cache(d):
    X,Y,H,C,L=prepare(d)
    return X,Y,H,C,L,RegressionLoss(X,Y,H,C)

def nuclear(solver,d,frac,start=None,cache=None,trace=None):
    X,Y,H,C,L,loss=make_cache(d) if cache is None else cache
    lam=frac*d["meta"]["nuclear_max"]
    B0=np.zeros_like(C) if start is None else start
    penalty=px.Nuclear(C.shape,sigma=lam)
    began=time.perf_counter()
    k=0
    def check(x):
        nonlocal k
        k+=1
        if k%10==0:
            B=x.reshape(C.shape)
            G=loss.grad(x).reshape(C.shape)
            residual=L*np.linalg.norm(B-svt(B-G/L,lam/L))/max(1,np.linalg.norm(C))
            if trace is not None:
                obj,gap,res=nuclear_certificate(X,Y,B,lam,L)
                trace.append((k,time.perf_counter()-began,obj,gap,res))
            if residual<=1e-8:
                raise StopSolve(B,k,"residual")
        if time.perf_counter()-began>=30:
            raise StopSolve(x.reshape(C.shape),k,"timeout")
    try:
        if solver=="py_fista":
            x=ProximalGradient(loss,penalty,B0.ravel().copy(),tau=1/L,niter=20000,
                               acceleration="fista",tol=None,callback=check,show=False)
        else:
            _,z=ADMM(loss,penalty,B0.ravel().copy(),tau=1/L,niter=20000,
                     tol=None,callback=lambda x,z:check(z),callbackz=True,show=False)
            x=z
        return x.reshape(C.shape),k,False,"iteration_limit"
    except StopSolve as stop:
        return stop.B,stop.iterations,stop.reason=="residual",stop.reason

def rank_fit(d):
    X,Y=d["X"]-d["X"].mean(0),d["Y"]-d["Y"].mean(0)
    U,s,Vt=la.svd(X,full_matrices=False)
    keep=s>max(X.shape)*np.finfo(float).eps*s[0]
    U,s,Vt=U[:,keep],s[keep],Vt[keep]
    H,h,Wt=la.svd(U.T@Y,full_matrices=False)
    r=d["meta"]["r"]
    return (Vt.T@(H[:,:r]/s[:,None]))@(h[:r,None]*Wt[:r]),0,True,""

def run_fit(solver,d,frac,mode="fit",trace=None):
    if solver=="scipy_rrr": return rank_fit(d)
    if mode=="path":
        cache=make_cache(d)
        B=np.zeros_like(cache[3]);its=0;ok=True
        for f in np.geomspace(.5,.02,8):
            B,k,flag,reason=nuclear(solver,d,f,B,cache)
            its+=k;ok &= flag
        return B,its,ok,reason
    return nuclear(solver,d,frac,cache=make_cache(d) if mode=="kernel" else None,trace=trace)

def one(d,out,solver,family,frac,round_,phase,mode="fit",cached=None):
    cache=make_cache(d) if mode=="kernel" else None
    began=time.perf_counter()
    if cache is not None:
        B,it,ok,detail=nuclear(solver,d,frac,cache=cache)
    else:
        B,it,ok,detail=run_fit(solver,d,frac,mode)
    secs=time.perf_counter()-began
    row=result_row(d,family,solver,.02 if mode=="path" else frac,round_,secs,it,B,
                   phase=phase,mode=mode,native=ok,detail=detail)
    row["rss_peak_kb"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not ok and detail=="timeout": row["status"]="timeout"
    if round_==1 and mode=="fit":
        save_array(out/f"{solver}-{frac:g}.bin",B)
    return row

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input");parser.add_argument("output")
    parser.add_argument("--phase",default="main")
    parser.add_argument("--memory",default="")
    args=parser.parse_args()
    d=load_problem(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    solvers=["scipy_rrr","py_fista","py_admm"]
    warm=load_problem(Path(args.input).parent/"small-1")
    if args.memory: solvers=[args.memory]
    for s in solvers:run_fit(s,warm,.1)
    baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.memory:
        row=one(d,out,args.memory,"rank" if args.memory=="scipy_rrr" else "nuclear",.1,1,args.phase,"memory")
        row["rss_baseline_kb"]=baseline
        (out/"memory.json").write_text(json.dumps(row))
        return
    if args.phase!="pilot":
        with open(out/"setup.csv","w") as setup:
            setup.write("round,setup_seconds,first_loss_prox_seconds,cached_loss_prox_seconds\n")
            for rd in range(1,4):
                t=time.perf_counter();c=make_cache(d);pre=time.perf_counter()-t
                x=np.zeros(c[3].size)
                t=time.perf_counter();c[5].prox(x,1/c[4]);first=time.perf_counter()-t
                t=time.perf_counter();c[5].prox(x,1/c[4]);warm_prox=time.perf_counter()-t
                setup.write(f"{rd},{pre},{first},{warm_prox}\n")
    f,writer=csv_writer(out/"python.csv")
    repeats=1 if args.phase=="pilot" else 3
    for round_ in range(1,repeats+1):
        for s in np.roll(solvers,(d["meta"]["rep"]+round_)%len(solvers)):
            fam="rank" if s=="scipy_rrr" else "nuclear"
            for frac in ((0,) if fam=="rank" else FRACS):
                try:
                    row=one(d,out,s,fam,frac,round_,args.phase)
                except Exception as err:
                    row=dict(phase=args.phase,language="python",case=d["meta"]["case"],
                             rep=d["meta"]["rep"],family=fam,solver=s,mode="fit",frac=frac,
                             round=round_,status="error",detail=repr(err))
                row["rss_baseline_kb"]=baseline
                writer.writerow(row);f.flush()
                print(json.dumps(row),flush=True)
    if args.phase!="pilot":
        for s in solvers[1:]:
            for mode in ("kernel","path"):
                for round_ in range(1,4):
                    row=one(d,out,s,"nuclear",.1,round_,args.phase,mode)
                    row["rss_baseline_kb"]=baseline;writer.writerow(row);f.flush()
    if args.phase!="pilot" and d["meta"]["case"] in ("medium","responses","wide"):
        for s in solvers:
            prof=cProfile.Profile()
            prof.enable();t=time.perf_counter();calls=0
            while time.perf_counter()-t<.25 or calls==0:
                run_fit(s,d,.1);calls+=1
            prof.disable()
            prof.dump_stats(str(out/f"profile-{s}.pstats"))
            with open(out/f"profile-{s}.txt","w") as report:
                report.write(f"rep={d['meta']['rep']} case={d['meta']['case']} calls={calls}\n")
                pstats.Stats(prof,stream=report).strip_dirs().sort_stats("cumtime").print_stats(60)
        for s in solvers[1:]:
            trace=[];nuclear(s,d,.1,trace=trace)
            with open(out/f"trace-{s}.csv","w") as t:
                t.write("iteration,seconds,objective,gap,residual\n")
                for row in trace:t.write(",".join(map(str,row))+"\n")
    f.close()
    (out/"python-environment.json").write_text(json.dumps(dict(
        numpy=np.__version__,pyproximal=px.__version__,threads=threadpool_info()),indent=2))

if __name__=="__main__":
    with threadpool_limits(limits=1):main()
