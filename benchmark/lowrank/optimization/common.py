"""Shared binary inputs and independent numerical certificates.

All numerical execution of this campaign belongs in a Discovery Slurm job.
"""
import csv
import json
from pathlib import Path
import numpy as np
from scipy import linalg as la

CASES = {
    "small": (300, 50, 10), "medium": (300, 50, 100),
    "responses": (300, 50, 1000), "wide": (200, 400, 100),
    "ill": (500, 100, 100), "deficient": (300, 80, 100),
}
FRACS = (0.5, 0.1, 0.02)
FIELDS = ("phase", "language", "case", "rep", "family", "solver", "mode",
          "frac", "round", "seconds", "iterations", "objective", "gap", "residual",
          "status", "native_converged", "rss_peak_kb", "rss_baseline_kb", "detail")

def save_array(path, x):
    np.asarray(x, dtype="<f8").ravel(order="F").tofile(path)

def load_array(path, shape):
    return np.fromfile(path, dtype="<f8").reshape(shape, order="F").copy(order="F")

def generate(root):
    root = Path(root)
    for ci, (case, (n, p, q)) in enumerate(CASES.items()):
        for rep in range(1, 6):
            dest = root / f"{case}-{rep}"
            dest.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(810000 + 100 * ci + rep)
            X = rng.standard_normal((n, p))
            if case == "ill":
                U, _ = la.qr(X - X.mean(0), mode="economic")
                V, _ = la.qr(rng.standard_normal((p, p)), mode="economic")
                X = (U * (np.sqrt(n) * np.geomspace(1, 1e-4, p))) @ V.T
            elif case == "deficient":
                X[:, 40:] = X[:, :40]
            X += rng.uniform(-0.5, 0.5, p)
            left = np.zeros((p, 3))
            left[:10] = rng.standard_normal((10, 3)) / np.sqrt(3)
            Btrue = left @ rng.standard_normal((3, q))
            Y = X @ Btrue + rng.standard_normal((n, q)) + rng.uniform(-1, 1, q)
            Xc, Yc = X - X.mean(0), Y - Y.mean(0)
            U, s, Vt = la.svd(Xc, full_matrices=False)
            keep = s > max(n, p) * np.finfo(float).eps * s[0]
            U, s, Vt = U[:, keep], s[keep], Vt[keep]
            H, h, Wt = la.svd(U.T @ Yc, full_matrices=False)
            B0 = (Vt.T @ (H[:, :3] / s[:, None])) @ (h[:3, None] * Wt[:3])
            ub, sb, vbt = la.svd(B0, full_matrices=False)
            A0, V0 = ub[:, :3] * sb[:3], vbt[:3].T
            layer = np.outer(ub[:, 0], vbt[0]) * sb[0]
            Ylayer = Yc - Xc @ (B0 - layer)
            C = Xc.T @ Yc / n
            C1 = Xc.T @ Ylayer / n
            ij = np.unravel_index(np.argmax(np.abs(C1)), C1.shape)
            u0, v0 = np.zeros(p), np.zeros(q)
            u0[ij[0]], v0[ij[1]] = 1, 1
            meta = dict(case=case, rep=rep, n=n, p=p, q=q, r=3,
                        nuclear_max=float(la.svdvals(C)[0]),
                        sparse_max=float(np.linalg.norm(C @ V0, axis=1).max()),
                        component_max=float(np.abs(C1).max()),
                        L=float(la.svdvals(Xc)[0] ** 2 / n),
                        reference_loss=float(np.sum((Yc-Xc@B0)**2)/(2*n)),
                        seed=810000+100*ci+rep)
            (dest / "meta.json").write_text(json.dumps(meta, indent=2))
            (dest / "dims.txt").write_text(f"{n}\n{p}\n{q}\n3\n")
            (dest / "scalars.txt").write_text("\n".join(str(meta[k]) for k in
                ("nuclear_max","sparse_max","component_max","L","reference_loss"))+"\n")
            for name, arr in dict(X=X, Y=Y, B0=B0, A0=A0, V0=V0,
                                  layer=Ylayer, u0=u0, v0=v0).items():
                save_array(dest / f"{name}.bin", arr)
    (root / "grid.json").write_text(json.dumps(CASES, indent=2))

def load_problem(root):
    root = Path(root)
    m = json.loads((root / "meta.json").read_text())
    n, p, q, r = (m[k] for k in ("n", "p", "q", "r"))
    d = dict(meta=m)
    for name, shape in dict(X=(n,p), Y=(n,q), B0=(p,q), A0=(p,r),
                            V0=(q,r), layer=(n,q), u0=(p,), v0=(q,)).items():
        d[name] = load_array(root / f"{name}.bin", shape)
    return d

def prepare(d):
    X, Y = d["X"] - d["X"].mean(0), d["Y"] - d["Y"].mean(0)
    n, p = X.shape
    H = X.T @ X / n if p <= n else None
    return X, Y, H, X.T @ Y / n, la.svdvals(X)[0]**2/n

def svt(B, threshold):
    U, s, Vt = la.svd(B, full_matrices=False)
    return (U * np.maximum(s-threshold, 0)) @ Vt

def nuclear_certificate(X, Y, B, lam, L=None):
    n = X.shape[0]
    R = Y-X@B
    G = -X.T@R/n
    L = la.svdvals(X)[0]**2/n if L is None else L
    primal = np.sum(R*R)/(2*n)+lam*la.svdvals(B).sum()
    normG = la.svdvals(G)[0]
    theta = R/n * min(1, lam/max(normG, np.finfo(float).tiny))
    dual = np.sum(Y*theta)-n*np.sum(theta*theta)/2
    gap = max(0, primal-dual)/max(1, abs(primal), abs(dual))
    residual = L*np.linalg.norm(B-svt(B-G/L, lam/L))/max(1,np.linalg.norm(X.T@Y/n))
    return float(primal), float(gap), float(residual)

def sparse_certificate(X, Y, A, V, lam, weights=None):
    n = X.shape[0]
    w = np.ones(A.shape[0]) if weights is None else weights
    E = X@A@V.T-Y
    G = X.T@E@V/n
    norms = np.linalg.norm(A,axis=1)
    active = norms > 1e-12
    errs = np.maximum(0,np.linalg.norm(G,axis=1)-lam*w)
    errs[active] = np.linalg.norm(G[active]+(lam*w[active]/norms[active])[:,None]*A[active],axis=1)
    GV = E.T@X@A/n
    tangent = GV-V@((V.T@GV+GV.T@V)/2)
    scale = max(1,np.linalg.norm(X.T@Y@V/n))
    residual = max(errs.max()/scale,np.linalg.norm(tangent)/max(1,np.linalg.norm(Y.T@X@A/n)),
                   np.linalg.norm(V.T@V-np.eye(V.shape[1])))
    obj = np.sum(E*E)/(2*n)+lam*np.dot(w,norms)
    return float(obj), float("nan"), float(residual)

def lasso_kkt(x, g, lam):
    active = np.abs(x)>1e-12
    e = np.maximum(0,np.abs(g)-lam)
    e[active] = np.abs(g[active]+lam*np.sign(x[active]))
    return e.max()

def component_certificate(X, Y, B, lam):
    n = X.shape[0]
    U,s,Vt = la.svd(B,full_matrices=False)
    if s[0] <= 1e-14:
        return float(np.sum(Y*Y)/(2*n)), float("nan"), 0.0
    u, v = U[:,0], Vt[0]*s[0]
    E = X@np.outer(u,v)-Y
    gu, gv = X.T@E@v/n, E.T@X@u/n
    a = lasso_kkt(u,gu,lam*np.abs(v).sum())/max(1,np.linalg.norm(X.T@Y@v/n))
    b = lasso_kkt(v,gv,lam*np.abs(u).sum())/max(1,np.linalg.norm(Y.T@X@u/n))
    obj = np.sum(E*E)/(2*n)+lam*np.abs(u).sum()*np.abs(v).sum()
    return float(obj), float("nan"), float(max(a,b))

def certificate(d, family, B, frac, A=None, V=None):
    X,Y,_,_,L = prepare(d)
    lam = frac*d["meta"].get(f"{family}_max",0)
    if family == "nuclear":
        return nuclear_certificate(X,Y,B,lam,L)
    if family == "rank":
        obj = np.sum((Y-X@B)**2)/(2*len(X))
        err = abs(obj-d["meta"]["reference_loss"])/max(1,abs(obj))
        rank = np.linalg.matrix_rank(B,tol=max(B.shape)*np.finfo(float).eps*la.svdvals(B)[0])
        return float(obj), float(err), float(max(0,rank-3))
    if family == "sparse":
        if A is None:
            U,s,Vt = la.svd(B,full_matrices=False)
            A,V = U[:,:3]*s[:3],Vt[:3].T
        return sparse_certificate(X,Y,A,V,lam)
    return component_certificate(X,d["layer"],B,lam)

def result_row(d, family, solver, frac, repeat, seconds, iterations, B,
               phase="main", mode="fit", native=True, A=None,V=None,detail=""):
    obj,gap,res = certificate(d,family,B,frac,A,V)
    good = res <= 1e-6 and (np.isnan(gap) or gap <= 1e-6)
    return dict(phase=phase,language="python",case=d["meta"]["case"],rep=d["meta"]["rep"],
                family=family,solver=solver,mode=mode,frac=frac,round=repeat,seconds=seconds,
                iterations=iterations,objective=obj,gap=gap,residual=res,
                status="passed" if good else "accuracy_failed",native_converged=native,detail=detail)

def csv_writer(path):
    f = open(path,"w",newline="")
    writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore")
    writer.writeheader()
    return f,writer

if __name__ == "__main__":
    import sys
    generate(sys.argv[1])
