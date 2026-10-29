"""Convex reference solutions for the dirty multitask model (#90).

Each case draws its own data with NumPy and solves the two-component problem

    minimize over A, S, β₀   Σₜ cₜ ‖yₜ - β₀ₜ - Xₜ(aₜ + sₜ)‖² / (2nₜ)
                              + Σⱼ ωⱼ (λ_shared ‖(s₁ⱼA₁ⱼ, …, s_qⱼA_qⱼ)‖ + λ_specific Σₜ |sₜⱼSₜⱼ|)

with CVXPY and the interior-point solver Clarabel, with A and S as separate variables
(not the single-B reduction used by the package). ‖·‖ is the row ℓ∞ norm
(shared_norm = linf) or the row ℓ2 norm (l2). sₜⱼ is the standard deviation of
column j within task t (divisor nₜ, about the mean with an intercept) with
standardize_x, and one otherwise; where sₜⱼ = 0 both components are constrained to
zero. cₜ = 1 for equal task weights and q nₜ / N for weights by observations. Nothing
here uses HighDimensionalModels.

Run: python -I oracle.py OUTDIR   (versions are written to OUTDIR/versions.csv)
"""

import csv
import os
import sys

import clarabel
import cvxpy as cp
import numpy as np

# The split into A and S is not unique, which makes the tightest interior-point tolerances
# unreachable on some levels; 1e-12, 1e-11 and 1e-10 are tried in turn.
TOLERANCES = (1e-12, 1e-11, 1e-10)


def scales(X, intercept, standardize):
    Xc = X - X.mean(axis=0) if intercept else X
    s = np.sqrt((Xc ** 2).mean(axis=0))
    if not standardize:
        s = (s > 0).astype(float)
    return s


def weights(kind, n):
    q = len(n)
    if kind == "equal":
        return np.ones(q)
    return np.array([q * nt / sum(n) for nt in n], dtype=float)


def solve(Xs, ys, c, S, omega, lb, ls, norm, intercept):
    q, p = len(Xs), Xs[0].shape[1]
    A = cp.Variable((p, q))
    Sv = cp.Variable((p, q))
    b0 = cp.Variable(q) if intercept else np.zeros(q)
    loss = sum(c[t] / (2 * len(ys[t])) * cp.sum_squares(ys[t] - b0[t] - Xs[t] @ (A[:, t] + Sv[:, t]))
               for t in range(q))
    pen = 0
    for j in range(p):
        rowA = cp.multiply(S[j, :], A[j, :])
        rowS = cp.multiply(S[j, :], Sv[j, :])
        pen += omega[j] * (lb * cp.norm(rowA, "inf" if norm == "linf" else 2) + ls * cp.norm(rowS, 1))
    cons = [A[j, t] == 0 for j in range(p) for t in range(q) if S[j, t] == 0]
    cons += [Sv[j, t] == 0 for j in range(p) for t in range(q) if S[j, t] == 0]
    prob = cp.Problem(cp.Minimize(loss + pen), cons)
    for tol in TOLERANCES:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, tol_ktratio=tol, max_iter=500)
        if prob.status == cp.OPTIMAL:
            break
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Clarabel status {prob.status}")
    Av, Sval = A.value, Sv.value
    b0v = b0.value if intercept else np.zeros(q)
    return Av, Sval, b0v, objective(Xs, ys, c, S, omega, lb, ls, norm, Av, Sval, b0v), prob.solver_stats.num_iters, tol


def objective(Xs, ys, c, S, omega, lb, ls, norm, A, Sv, b0):
    q, p = len(Xs), Xs[0].shape[1]
    B = A + Sv
    loss = sum(c[t] * np.sum((ys[t] - b0[t] - Xs[t] @ B[:, t]) ** 2) / (2 * len(ys[t])) for t in range(q))
    pen = 0.0
    for j in range(p):
        ra, rs = S[j, :] * A[j, :], S[j, :] * Sv[j, :]
        na = np.max(np.abs(ra)) if norm == "linf" else np.linalg.norm(ra)
        pen += omega[j] * (lb * na + ls * np.sum(np.abs(rs)))
    return float(loss + pen)


def write_matrix(path, A):
    A = np.atleast_2d(A)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in A:
            w.writerow([repr(float(v)) for v in row])


def draw(rng, n, p, q, shared):
    """Uniform designs (portable) with shared rows of unequal magnitudes and
    task-specific entries."""
    if shared:
        X = rng.uniform(-1, 1, size=(n[0], p)) * np.linspace(0.5, 2.0, p)
        Xs = [X] * q
    else:
        Xs = [rng.uniform(-1, 1, size=(nt, p)) * np.linspace(0.5, 2.0, p) + 0.3 * t for t, nt in enumerate(n)]
    B = np.zeros((p, q))
    B[0, :] = 1.0
    B[1, :] = np.linspace(0.5, 1.5, q)
    B[2, :] = -0.8
    for t in range(q):
        B[3 + t, t] = 1.2
    ys = [1.5 * t - 1 + Xs[t] @ B[:, t] + 0.4 * rng.uniform(-1, 1, size=len(Xs[t][:, 0])) for t in range(q)]
    return Xs, ys


# name, seed, n, p, shared, norm, intercept, standardize, task weights, omega, (λ_shared, λ_specific) pairs
# The ratio r = λ_shared/λ_specific has regimes: r ≤ 1 is row ℓ1/ℓ∞ (ℓ1/ℓ2), r ≥ q (√q) the entrywise
# lasso; q = 3 here, so ratios 1.5 and 2.0 are interior, 2.0 an integer ratio, 3.0 the lasso endpoint.
CASES = [
    ("shared-linf", 1, [40, 40, 40], 9, True, "linf", True, True, "equal", None,
     [(0.30, 0.15), (0.12, 0.08), (0.10, 0.05)]),
    ("tasks-linf", 2, [30, 45, 25], 9, False, "linf", True, True, "observations",
     [1, 0.5, 2, 1, 1, 1.5, 1, 1, 0.8], [(0.24, 0.12), (0.09, 0.06)]),
    ("tasks-linf-endpoints", 3, [35, 28, 40], 8, False, "linf", True, False, "equal", None,
     [(0.06, 0.06), (0.30, 0.10)]),
    ("shared-l2", 4, [40, 40, 40], 9, True, "l2", True, True, "equal", None, [(0.20, 0.14), (0.08, 0.06)]),
    ("tasks-l2-zero-factor", 5, [32, 26, 40], 8, False, "l2", False, True, "equal",
     [1, 1, 0, 1, 1, 1, 1, 1], [(0.15, 0.10)]),
]


def main(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "versions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for k, v in (("python", sys.version.split()[0]), ("numpy", np.__version__), ("cvxpy", cp.__version__),
                     ("clarabel", clarabel.__version__)):
            w.writerow([k, v])
    for (name, seed, n, p, shared, norm, intercept, standardize, wkind, omega, pairs) in CASES:
        rng = np.random.default_rng(seed)
        q = len(n)
        Xs, ys = draw(rng, n, p, q, shared)
        if name == "tasks-linf":
            Xs[1] = Xs[1].copy()
            Xs[1][:, p - 1] = 2.5          # constant in task 2: a dropped coordinate with an intercept
        c = weights(wkind, n)
        S = np.column_stack([scales(X, intercept, standardize) for X in Xs])
        om = np.ones(p) if omega is None else np.asarray(omega, dtype=float)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        if shared:
            write_matrix(os.path.join(d, "X.csv"), Xs[0])
        else:
            for t in range(q):
                write_matrix(os.path.join(d, f"X{t + 1}.csv"), Xs[t])
        for t in range(q):
            write_matrix(os.path.join(d, f"y{t + 1}.csv"), ys[t].reshape(-1, 1))
        write_matrix(os.path.join(d, "penalty_factors.csv"), om.reshape(-1, 1))
        with open(os.path.join(d, "settings.csv"), "w", newline="") as f:
            w = csv.writer(f)
            for k, v in (("tasks", q), ("shared", str(shared).lower()), ("shared_norm", norm),
                         ("intercept", str(intercept).lower()), ("standardize_x", str(standardize).lower()),
                         ("task_weights", wkind), ("levels", len(pairs))):
                w.writerow([k, v])
        for k, (lb, ls) in enumerate(pairs, start=1):
            A, Sv, b0, obj, iters, tol = solve(Xs, ys, c, S, om, lb, ls, norm, intercept)
            write_matrix(os.path.join(d, f"A{k}.csv"), A)
            write_matrix(os.path.join(d, f"S{k}.csv"), Sv)
            write_matrix(os.path.join(d, f"B{k}.csv"), A + Sv)
            write_matrix(os.path.join(d, f"intercept{k}.csv"), b0.reshape(-1, 1))
            with open(os.path.join(d, f"level{k}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["lambda_shared", repr(float(lb))])
                w.writerow(["lambda_specific", repr(float(ls))])
                w.writerow(["objective", repr(float(obj))])
                w.writerow(["iterations", iters])
                w.writerow(["tolerance", repr(tol)])
            B = A + Sv
            print(f"{name} level {k}: ratio {lb / ls:.2f}, objective {obj:.12g}, "
                  f"B entries above 1e-8: {int(np.sum(np.abs(B) > 1e-8))}, A rows above 1e-8: "
                  f"{int(np.sum(np.max(np.abs(A), axis=1) > 1e-8))}, S entries above 1e-8: "
                  f"{int(np.sum(np.abs(Sv) > 1e-8))}, {iters} iterations")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
