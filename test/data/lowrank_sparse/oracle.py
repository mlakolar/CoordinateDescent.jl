"""Convex reference solutions for regression with a nuclear and a sparsity penalty (#92).

Each case draws its own data with NumPy, solves

    minimize  ‖Y - 1β₀' - XB‖²_F / (2n) + λ_rank ‖B‖_* + λ_sparse P(B)

with CVXPY and the interior-point solver Clarabel (the nuclear norm through its
semidefinite cone), and writes the data, the settings and the solutions to
test/data/lowrank_sparse/<case>/ as CSV. P(B) = Σⱼ wⱼ Σₜ |Bⱼₜ| for sparsity = entry and
Σⱼ wⱼ ‖B[j, :]‖₂ for sparsity = row. Nothing here uses HighDimensionalModels; the
thresholds λmax are computed below from their definitions.

Run: python -I oracle.py OUTDIR   (versions are written to OUTDIR/versions.csv)
"""

import csv
import os
import sys

import clarabel
import cvxpy as cp
import numpy as np

SETTINGS = dict(tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11, tol_ktratio=1e-9, max_iter=1000)


def sparsity_value(B, w, sparsity):
    if sparsity == "entry":
        return float(np.sum(w[:, None] * np.abs(B)))
    return float(np.sum(w * np.linalg.norm(B, axis=1)))


def sparsity_expr(B, w, sparsity):
    if sparsity == "entry":
        return cp.sum(cp.multiply(w[:, None] * np.ones(B.shape), cp.abs(B)))
    return sum(w[j] * cp.norm(B[j, :], 2) for j in range(B.shape[0]))


def thresholds(X, Y, w, intercept, sparsity):
    """λmax of each penalty alone: the dual norm of the gradient at zero."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0) if intercept else X
    Yc = Y - Y.mean(axis=0) if intercept else Y
    C = Xc.T @ Yc / n
    rank = np.linalg.norm(C, 2)
    if sparsity == "entry":
        sparse = np.max(np.abs(C) / w[:, None])
    else:
        sparse = np.max(np.linalg.norm(C, axis=1) / w)
    return float(rank), float(sparse)


def objective(X, Y, B, b0, lr, ls, w, sparsity):
    n = X.shape[0]
    R = Y - b0[None, :] - X @ B
    return float(np.sum(R ** 2) / (2 * n) + lr * np.sum(np.linalg.svd(B, compute_uv=False))
                 + ls * sparsity_value(B, w, sparsity))


def solve(X, Y, lr, ls, w, intercept, sparsity):
    n, p = X.shape
    q = Y.shape[1]
    B = cp.Variable((p, q))
    b0 = cp.Variable(q) if intercept else np.zeros(q)
    ones = np.ones((n, 1))
    fit = Y - (ones @ cp.reshape(b0, (1, q), order="F") if intercept else 0) - X @ B
    prob = cp.Problem(cp.Minimize(cp.sum_squares(fit) / (2 * n) + lr * cp.normNuc(B)
                                  + ls * sparsity_expr(B, w, sparsity)))
    prob.solve(solver=cp.CLARABEL, **SETTINGS)
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Clarabel status {prob.status}")
    b0v = b0.value if intercept else np.zeros(q)
    return B.value, b0v, objective(X, Y, B.value, b0v, lr, ls, w, sparsity), prob.solver_stats.num_iters


def write_matrix(path, A):
    A = np.atleast_2d(A)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        for row in A:
            wr.writerow([repr(float(v)) for v in row])


def draw(rng, n, p, q, rank, nonzero_rows):
    """Uniform designs (portable) and a coefficient matrix that is sparse and of low rank."""
    X = rng.uniform(-1, 1, size=(n, p)) * np.linspace(0.6, 1.8, p)
    U = np.zeros((p, rank))
    U[:nonzero_rows, :] = rng.uniform(-1, 1, size=(nonzero_rows, rank))
    V = rng.uniform(-1, 1, size=(q, rank))
    V[rng.uniform(size=V.shape) < 0.3] = 0.0
    B = 2 * U @ V.T
    Y = 0.5 + X @ B + 0.4 * rng.uniform(-1, 1, size=(n, q))
    return X, Y


CASES = [
    # name, seed, n, p, q, sparsity, intercept, weights, pairs as fractions of (λ_rank max, λ_sparse max)
    ("entry", 11, 30, 8, 5, "entry", True, None, [(0.5, 0.3), (0.2, 0.2), (0.3, 0.05)]),
    ("row", 12, 30, 8, 5, "row", True, [1, 0.5, 2, 1, 1, 1.5, 0.8, 1], [(0.5, 0.3), (0.15, 0.25), (0.3, 0.05)]),
    ("entry-wide", 13, 10, 12, 4, "entry", True, None, [(0.4, 0.2), (0.1, 0.1)]),
    ("row-noint", 14, 25, 7, 6, "row", False, None, [(0.3, 0.2), (0.1, 0.05)]),
]


def main(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "versions.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        for k, v in (("python", sys.version.split()[0]), ("numpy", np.__version__), ("cvxpy", cp.__version__),
                     ("clarabel", clarabel.__version__)):
            wr.writerow([k, v])
    for (name, seed, n, p, q, sparsity, intercept, weights, pairs) in CASES:
        rng = np.random.default_rng(seed)
        X, Y = draw(rng, n, p, q, 2, max(2, p // 2))
        w = np.ones(p) if weights is None else np.asarray(weights, dtype=float)
        rmax, smax = thresholds(X, Y, w, intercept, sparsity)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        write_matrix(os.path.join(d, "X.csv"), X)
        write_matrix(os.path.join(d, "Y.csv"), Y)
        write_matrix(os.path.join(d, "weights.csv"), w.reshape(-1, 1))
        with open(os.path.join(d, "settings.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            for k, v in (("sparsity", sparsity), ("intercept", str(intercept).lower()),
                         ("lambda_rank_max", repr(rmax)), ("lambda_sparse_max", repr(smax)), ("levels", len(pairs))):
                wr.writerow([k, v])
        for k, (fr, fs) in enumerate(pairs, start=1):
            lr, ls = fr * rmax, fs * smax
            B, b0, obj, iters = solve(X, Y, lr, ls, w, intercept, sparsity)
            write_matrix(os.path.join(d, f"B{k}.csv"), B)
            write_matrix(os.path.join(d, f"intercept{k}.csv"), b0.reshape(-1, 1))
            with open(os.path.join(d, f"level{k}.csv"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["lambda_rank", repr(float(lr))])
                wr.writerow(["lambda_sparse", repr(float(ls))])
                wr.writerow(["objective", repr(float(obj))])
                wr.writerow(["iterations", iters])
            sv = np.linalg.svd(B, compute_uv=False)
            print(f"{name} level {k}: objective {obj:.12g}, entries above 1e-7: {int(np.sum(np.abs(B) > 1e-7))}, "
                  f"singular values above 1e-7: {int(np.sum(sv > 1e-7))}, {iters} iterations")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
