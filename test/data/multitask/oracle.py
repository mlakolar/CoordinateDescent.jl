"""Convex reference solutions for the multitask penalties of #89.

Each case draws its own data with NumPy, solves the problem with CVXPY and the
interior-point solver Clarabel, and writes the data, the settings and the solutions
to test/data/multitask/<case>/ as CSV. Nothing here uses HighDimensionalModels: the
column scales, the task weights, the penalty and λmax are computed below from their
definitions in the documentation.

    minimize  Σₜ cₜ ‖yₜ - β₀ₜ - Xₜbₜ‖² / (2nₜ) + λ Σⱼ Pⱼ(s₁ⱼb₁ⱼ, …, s_qⱼb_qⱼ)

with Pⱼ(v) = ωⱼ ‖v‖_∞ for row_norm = linf, and α‖v‖₁ + (1 - α) ωⱼ ‖v‖₂ for the sparse
group (row_norm = l2, alpha > 0): the penalty factors ωⱼ weight the row term only.
sₜⱼ is the standard deviation of column j within task t (divisor nₜ, about the mean
with an intercept and about zero without) with standardize_x, and one otherwise. cₜ = 1 for equal task weights and q nₜ / N for
weights by observations.

Run: python oracle.py OUTDIR   (cvxpy and clarabel; versions are written to
OUTDIR/versions.csv)
"""

import csv
import os
import sys

import clarabel
import cvxpy as cp
import numpy as np

SETTINGS = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, tol_ktratio=1e-10, max_iter=500,
                equilibrate_enable=True)


def scales(X, intercept, standardize):
    if not standardize:
        return np.ones(X.shape[1])
    Xc = X - X.mean(axis=0) if intercept else X
    return np.sqrt((Xc ** 2).mean(axis=0))


def weights(kind, n):
    q = len(n)
    if kind == "equal":
        return np.ones(q)
    return np.array([q * nt / sum(n) for nt in n], dtype=float)


def penalty_row(v, omega, row_norm, alpha):
    """The row penalty Pⱼ as a CVXPY expression."""
    if row_norm == "linf":
        return omega * cp.norm(v, "inf")
    return alpha * cp.norm(v, 1) + (1 - alpha) * omega * cp.norm(v, 2)


def penalty_value(v, omega, row_norm, alpha):
    if row_norm == "linf":
        return omega * np.max(np.abs(v))
    return alpha * np.sum(np.abs(v)) + (1 - alpha) * omega * np.linalg.norm(v)


def gradient_at_zero(Xs, ys, c, S, intercept):
    """∂/∂(sₜⱼbₜⱼ) of the loss at B = 0, with the intercepts at their minimizers."""
    q, p = len(Xs), Xs[0].shape[1]
    G = np.zeros((p, q))
    for t in range(q):
        X, y, n = Xs[t], ys[t], len(ys[t])
        Xc = X - X.mean(axis=0) if intercept else X
        yc = y - y.mean() if intercept else y
        G[:, t] = -c[t] * (Xc.T @ yc) / n / S[:, t]
    return G


def lambda_max(G, omega, row_norm, alpha):
    """The smallest λ at which B = 0 is optimal: the dual norm of each row's gradient."""
    lam = 0.0
    for j in range(G.shape[0]):
        g = np.abs(G[j])
        if row_norm == "linf":
            lam = max(lam, g.sum() / omega[j])
        elif alpha == 0:
            lam = max(lam, np.linalg.norm(g) / omega[j])
        else:
            # zero is optimal while ‖soft(g, αλ)‖₂ ≤ (1 - α)ωⱼλ; the left minus the right
            # decreases in λ, so bisect for the root (at most max|g| / α)
            lo, hi = 0.0, g.max() / alpha
            for _ in range(200):
                mid = (lo + hi) / 2
                if np.linalg.norm(np.maximum(g - alpha * mid, 0)) > (1 - alpha) * omega[j] * mid:
                    lo = mid
                else:
                    hi = mid
            lam = max(lam, hi)
    return lam


def solve(Xs, ys, c, S, omega, lam, row_norm, alpha, intercept):
    q, p = len(Xs), Xs[0].shape[1]
    B = cp.Variable((p, q))
    b0 = cp.Variable(q) if intercept else np.zeros(q)
    loss = sum(c[t] / (2 * len(ys[t])) * cp.sum_squares(ys[t] - b0[t] - Xs[t] @ B[:, t]) for t in range(q))
    pen = sum(penalty_row(cp.multiply(S[j, :], B[j, :]), omega[j], row_norm, alpha) for j in range(p))
    prob = cp.Problem(cp.Minimize(loss + lam * pen))
    prob.solve(solver=cp.CLARABEL, **SETTINGS)
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Clarabel status {prob.status}")
    Bv = B.value
    b0v = b0.value if intercept else np.zeros(q)
    obj = objective(Xs, ys, c, S, omega, lam, row_norm, alpha, Bv, b0v)
    return Bv, b0v, obj, prob.solver_stats.num_iters


def objective(Xs, ys, c, S, omega, lam, row_norm, alpha, B, b0):
    q, p = len(Xs), Xs[0].shape[1]
    loss = sum(c[t] * np.sum((ys[t] - b0[t] - Xs[t] @ B[:, t]) ** 2) / (2 * len(ys[t])) for t in range(q))
    pen = sum(penalty_value(S[j, :] * B[j, :], omega[j], row_norm, alpha) for j in range(p))
    return loss + lam * pen


def write_matrix(path, A):
    A = np.atleast_2d(A)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in A:
            w.writerow([repr(float(v)) for v in row])


def draw(rng, n, p, q, shared, support):
    """Centered uniform designs (portable across languages) and responses."""
    if shared:
        X = rng.uniform(-1, 1, size=(n[0], p)) * np.linspace(0.5, 2.0, p)
        Xs = [X] * q
    else:
        Xs = [rng.uniform(-1, 1, size=(nt, p)) * np.linspace(0.5, 2.0, p) for nt in n]
    Btrue = np.zeros((p, q))
    for (j, t, v) in support:
        Btrue[j, t] = v
    ys = [1.5 * t - 1 + Xs[t] @ Btrue[:, t] + 0.5 * rng.uniform(-1, 1, size=len(Xs[t][:, 0])) for t in range(q)]
    return Xs, ys


SHARED_SUPPORT = [(j, t, v) for j, v in ((0, 1.0), (1, -0.8), (2, 0.6)) for t in range(3)] + [(3, 1, 1.2)]
SPECIFIC_SUPPORT = [(0, 0, 1.0), (0, 1, 0.9), (0, 2, -1.1), (1, 0, 0.8), (4, 1, -1.0), (5, 2, 0.7), (6, 0, 0.5)]

CASES = [
    # name, seed, n, p, shared, support, row_norm, alpha, intercept, standardize, task weights, omega, fractions of λmax
    ("shared-linf", 1, [40, 40, 40], 10, True, SHARED_SUPPORT, "linf", 0.0, True, True, "equal", None, [0.5, 0.15]),
    ("tasks-linf", 2, [30, 45, 25], 10, False, SPECIFIC_SUPPORT, "linf", 0.0, True, True, "observations",
     [1, 0.5, 2, 1, 1, 1.5, 1, 1, 0.8, 1], [0.4, 0.1]),
    ("tasks-linf-raw", 3, [35, 20, 28], 8, False, SPECIFIC_SUPPORT, "linf", 0.0, True, False, "equal", None, [0.3]),
    ("shared-sgl", 4, [40, 40, 40], 10, True, SHARED_SUPPORT, "l2", 0.5, True, True, "equal", None, [0.4, 0.1]),
    # a zero factor leaves the row term of predictor 5 out; its entries keep the ℓ1 term
    ("tasks-sgl", 5, [30, 45, 25], 10, False, SPECIFIC_SUPPORT, "l2", 0.3, True, True, "observations",
     [1, 0.5, 2, 1, 0, 1.5, 1, 1, 0.8, 1], [0.12, 0.03]),
    ("tasks-sgl-noint", 6, [32, 26, 40], 8, False, SPECIFIC_SUPPORT, "l2", 0.7, False, False, "equal", None, [0.25]),
]


def main(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "versions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["python", sys.version.split()[0]])
        w.writerow(["numpy", np.__version__])
        w.writerow(["cvxpy", cp.__version__])
        w.writerow(["clarabel", clarabel.__version__])
    for (name, seed, n, p, shared, support, row_norm, alpha, intercept, standardize, wkind, omega, fracs) in CASES:
        rng = np.random.default_rng(seed)
        q = len(n)
        Xs, ys = draw(rng, n, p, q, shared, support)
        c = weights(wkind, n)
        S = np.column_stack([scales(X, intercept, standardize) for X in Xs])
        om = np.ones(p) if omega is None else np.asarray(omega, dtype=float)
        lmax = lambda_max(gradient_at_zero(Xs, ys, c, S, intercept), om, row_norm, alpha)
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
            for k, v in (("tasks", q), ("shared", str(shared).lower()), ("row_norm", row_norm), ("alpha", repr(float(alpha))),
                         ("intercept", str(intercept).lower()), ("standardize_x", str(standardize).lower()),
                         ("task_weights", wkind), ("lambda_max", repr(float(lmax))), ("levels", len(fracs))):
                w.writerow([k, v])
        for k, fr in enumerate(fracs, start=1):
            lam = fr * lmax
            B, b0, obj, iters = solve(Xs, ys, c, S, om, lam, row_norm, alpha, intercept)
            write_matrix(os.path.join(d, f"B{k}.csv"), B)
            write_matrix(os.path.join(d, f"intercept{k}.csv"), b0.reshape(-1, 1))
            with open(os.path.join(d, f"level{k}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["lambda", repr(float(lam))])
                w.writerow(["objective", repr(float(obj))])
                w.writerow(["iterations", iters])
            print(f"{name} level {k}: λ = {lam:.6g} ({fr} λmax), objective {obj:.12g}, "
                  f"{int(np.sum(np.abs(B) > 1e-8))} entries above 1e-8, {iters} iterations")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
