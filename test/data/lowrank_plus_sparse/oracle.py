"""Convex reference solutions for low-rank-plus-sparse regression (#91).

Each case draws its own data with NumPy and solves

    minimize over L, S, β₀   ‖Y - 1β₀' - X(L + S)‖²_F / (2n) + λ_rank ‖L‖_* + λ_sparse P(S)

with CVXPY and the interior-point solver Clarabel (the nuclear norm through its
semidefinite cone), with L and S as separate variables and the intercepts optimized
explicitly. P(S) = Σⱼ wⱼ Σₜ |Sⱼₜ| for sparsity = entry and Σⱼ wⱼ ‖S[j, :]‖₂ for
sparsity = row. Nothing here uses HighDimensionalModels. The split into L and S need
not be unique; only B = L + S (on full-rank designs), the fitted values and the
objective are identified in general.

Run: python -I oracle.py OUTDIR   (versions are written to OUTDIR/versions.csv)
"""

import csv
import os
import sys

import clarabel
import cvxpy as cp
import numpy as np

TOLERANCES = (1e-10, 1e-9)


def sparsity_value(S, w, sparsity):
    if sparsity == "entry":
        return float(np.sum(w[:, None] * np.abs(S)))
    return float(np.sum(w * np.linalg.norm(S, axis=1)))


def sparsity_expr(S, w, sparsity):
    if sparsity == "entry":
        return cp.sum(cp.multiply(w[:, None] * np.ones(S.shape), cp.abs(S)))
    return sum(w[j] * cp.norm(S[j, :], 2) for j in range(S.shape[0]))


def references(X, Y, w, intercept, sparsity):
    """The thresholds at the zero pair: ‖C₀‖_op and the weighted sparsity dual norm of C₀."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0) if intercept else X
    Yc = Y - Y.mean(axis=0) if intercept else Y
    C = Xc.T @ Yc / n
    rank = np.linalg.norm(C, 2)
    sparse = np.max(np.abs(C) / w[:, None]) if sparsity == "entry" else np.max(np.linalg.norm(C, axis=1) / w)
    return float(rank), float(sparse)


def objective(X, Y, L, S, b0, lr, ls, w, sparsity):
    n = X.shape[0]
    R = Y - b0[None, :] - X @ (L + S)
    return float(np.sum(R ** 2) / (2 * n) + lr * np.sum(np.linalg.svd(L, compute_uv=False))
                 + ls * sparsity_value(S, w, sparsity))


def solve(X, Y, lr, ls, w, intercept, sparsity):
    n, p = X.shape
    q = Y.shape[1]
    L = cp.Variable((p, q))
    S = cp.Variable((p, q))
    b0 = cp.Variable(q) if intercept else np.zeros(q)
    ones = np.ones((n, 1))
    fit = Y - (ones @ cp.reshape(b0, (1, q), order="F") if intercept else 0) - X @ (L + S)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(fit) / (2 * n) + lr * cp.normNuc(L)
                                  + ls * sparsity_expr(S, w, sparsity)))
    for tol in TOLERANCES:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, tol_ktratio=tol, max_iter=1000)
        if prob.status == cp.OPTIMAL:
            break
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Clarabel status {prob.status}")
    b0v = b0.value if intercept else np.zeros(q)
    return L.value, S.value, b0v, objective(X, Y, L.value, S.value, b0v, lr, ls, w, sparsity), \
        prob.solver_stats.num_iters, tol


def write_matrix(path, A):
    A = np.atleast_2d(A)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        for row in A:
            wr.writerow([repr(float(v)) for v in row])


def draw(rng, n, p, q, design):
    """Uniform designs (portable) and a coefficient matrix that is low rank plus sparse."""
    X = rng.uniform(-1, 1, size=(n, p)) * np.linspace(0.6, 1.8, p)
    if design == "duplicated":
        X[:, 1] = X[:, 0]
    Lt = rng.uniform(-1, 1, size=(p, 1)) @ rng.uniform(-1, 1, size=(1, q))
    St = np.zeros((p, q))
    St[0, 0], St[2, 1], St[p - 1, q - 1] = 2.0, -1.5, 1.2
    Y = 0.5 + X @ (Lt + St) + 0.3 * rng.uniform(-1, 1, size=(n, q))
    return X, Y


# name, seed, n, p, q, sparsity, intercept, weights, design, pairs as fractions of the reference thresholds
CASES = [
    ("entry", 21, 30, 8, 5, "entry", True, None, "full", [(0.5, 0.5), (0.3, 0.2), (0.15, 0.1)]),
    ("row", 22, 30, 8, 5, "row", True, [1, 0.5, 2, 1, 1, 1.5, 0.8, 1], "full", [(0.4, 0.4), (0.2, 0.15)]),
    ("entry-noint", 23, 25, 7, 6, "entry", False, None, "full", [(0.3, 0.2)]),
    ("entry-duplicated", 24, 30, 8, 4, "entry", True, None, "duplicated", [(0.3, 0.2)]),
    ("row-wide", 25, 10, 12, 4, "row", True, None, "full", [(0.3, 0.3)]),
]


def main(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "versions.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        for k, v in (("python", sys.version.split()[0]), ("numpy", np.__version__), ("cvxpy", cp.__version__),
                     ("clarabel", clarabel.__version__)):
            wr.writerow([k, v])
    for (name, seed, n, p, q, sparsity, intercept, weights, design, pairs) in CASES:
        rng = np.random.default_rng(seed)
        X, Y = draw(rng, n, p, q, design)
        w = np.ones(p) if weights is None else np.asarray(weights, dtype=float)
        rmax, smax = references(X, Y, w, intercept, sparsity)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        write_matrix(os.path.join(d, "X.csv"), X)
        write_matrix(os.path.join(d, "Y.csv"), Y)
        write_matrix(os.path.join(d, "weights.csv"), w.reshape(-1, 1))
        with open(os.path.join(d, "settings.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            for k, v in (("sparsity", sparsity), ("intercept", str(intercept).lower()), ("design", design),
                         ("lambda_rank_reference", repr(rmax)), ("lambda_sparse_reference", repr(smax)),
                         ("levels", len(pairs))):
                wr.writerow([k, v])
        for k, (fr, fs) in enumerate(pairs, start=1):
            lr, ls = fr * rmax, fs * smax
            L, S, b0, obj, iters, tol = solve(X, Y, lr, ls, w, intercept, sparsity)
            write_matrix(os.path.join(d, f"L{k}.csv"), L)
            write_matrix(os.path.join(d, f"S{k}.csv"), S)
            write_matrix(os.path.join(d, f"B{k}.csv"), L + S)
            write_matrix(os.path.join(d, f"intercept{k}.csv"), b0.reshape(-1, 1))
            with open(os.path.join(d, f"level{k}.csv"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["lambda_rank", repr(float(lr))])
                wr.writerow(["lambda_sparse", repr(float(ls))])
                wr.writerow(["objective", repr(float(obj))])
                wr.writerow(["iterations", iters])
                wr.writerow(["tolerance", repr(tol)])
            sv = np.linalg.svd(L, compute_uv=False)
            print(f"{name} level {k}: objective {obj:.12g}, rank(L) {int(np.sum(sv > 1e-7 * max(1, sv[0])))}, "
                  f"S entries above 1e-7: {int(np.sum(np.abs(S) > 1e-7))}, {iters} iterations at tol {tol}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
