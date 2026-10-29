"""Convex reference solutions for joint-sparse transfer multitask learning (#93).

Domain 0 is the target and domains 1..L the sources. Domain l has n_l observations of
K responses; its design is one matrix X_l, or one matrix X_lk per response. With
N = sum_l n_l, each case solves the fused problem in the original coordinates

    minimize over B_0..B_L, b_0..b_L
        (1/(2NK)) sum_l sum_k ||y_lk - b_lk - X_lk B_l[:, k]||^2
        + lam0 sum_j ||B_0[j, :]||_2 + lam0 sum_{l>=1} a_l sum_j ||B_l[j, :] - B_0[j, :]||_2

with unpenalized intercepts b_lk (or none), and then the correction on the target

    minimize over D, d   (1/(2 n_0 K)) sum_k ||y_0k - d_k - X_0k (W[:, k] + D[:, k])||^2
                          + lamt sum_j ||D[j, :]||_2,

where W = sum_l (n_l/N) B_l is the fused estimate of the reference solution. The
contrast weights are a_l = 8 sqrt(n_l/N) (sqrt), 8 n_l/N (linear) or given values.
lambda_max is max(max_j ||sum_l S_lj||_2, max_{l>=1, j} ||S_lj||_2 / a_l), with
S_ljk = <x_lkj, r_lk>/(NK) on centered data at zero coefficients; lamt_max is the
correction's largest gradient row norm at D = 0.

The problems are solved with CVXPY and the interior-point solver Clarabel, with every
B_l a separate variable (not the package's reparametrization). Nothing here uses
HighDimensionalModels.

Run: python -I oracle.py OUTDIR   (versions are written to OUTDIR/versions.csv)
"""

import csv
import os
import sys

import clarabel
import cvxpy as cp
import numpy as np

TOLERANCES = (1e-12, 1e-11, 1e-10)


def designs(Xs, l, k):
    X = Xs[l]
    return X[k] if isinstance(X, list) else X


def center(A, intercept):
    return A - A.mean(axis=0) if intercept else A


def contrast_weights(kind, n):
    N = sum(n)
    if kind == "sqrt":
        return np.array([8 * np.sqrt(nl / N) for nl in n[1:]])
    if kind == "linear":
        return np.array([8 * nl / N for nl in n[1:]])
    return np.asarray(kind, dtype=float)


def lambda_max(Xs, Ys, n, K, a, intercept):
    N, L = sum(n), len(n) - 1
    S = [np.column_stack([center(designs(Xs, l, k), intercept).T @ center(Ys[l][:, k], intercept)
                          for k in range(K)]) / (N * K) for l in range(L + 1)]
    total = sum(S)
    m = np.max(np.linalg.norm(total, axis=1))
    for l in range(1, L + 1):
        m = max(m, np.max(np.linalg.norm(S[l], axis=1)) / a[l - 1])
    return float(m)


def solve(prob):
    for tol in TOLERANCES:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol, tol_ktratio=tol,
                   max_iter=500)
        if prob.status == cp.OPTIMAL:
            return prob.solver_stats.num_iters, tol
    raise RuntimeError(f"Clarabel status {prob.status}")


def fused(Xs, Ys, n, K, a, lam0, intercept):
    N, L, p = sum(n), len(n) - 1, designs(Xs, 0, 0).shape[1]
    B = [cp.Variable((p, K)) for _ in range(L + 1)]
    b = [cp.Variable(K) for _ in range(L + 1)] if intercept else None
    loss = 0
    for l in range(L + 1):
        for k in range(K):
            r = Ys[l][:, k] - designs(Xs, l, k) @ B[l][:, k]
            if intercept:
                r = r - b[l][k]
            loss += cp.sum_squares(r)
    pen = lam0 * cp.mixed_norm(B[0], 2, 1)
    for l in range(1, L + 1):
        pen += lam0 * a[l - 1] * cp.mixed_norm(B[l] - B[0], 2, 1)
    iters, tol = solve(cp.Problem(cp.Minimize(loss / (2 * N * K) + pen)))
    Bv = [Bl.value for Bl in B]
    bv = [bl.value for bl in b] if intercept else [np.zeros(K) for _ in range(L + 1)]
    return Bv, bv, fused_objective(Xs, Ys, n, K, a, lam0, Bv, bv), iters, tol


def fused_objective(Xs, Ys, n, K, a, lam0, B, b):
    N, L = sum(n), len(n) - 1
    loss = sum(np.sum((Ys[l][:, k] - b[l][k] - designs(Xs, l, k) @ B[l][:, k]) ** 2)
               for l in range(L + 1) for k in range(K))
    pen = lam0 * np.sum(np.linalg.norm(B[0], axis=1))
    for l in range(1, L + 1):
        pen += lam0 * a[l - 1] * np.sum(np.linalg.norm(B[l] - B[0], axis=1))
    return float(loss / (2 * N * K) + pen)


def correction(Xs, Ys, n, K, W, intercept, fraction):
    n0, p = n[0], W.shape[0]
    R = [center(Ys[0][:, k] - designs(Xs, 0, k) @ W[:, k], intercept) for k in range(K)]
    G = np.column_stack([center(designs(Xs, 0, k), intercept).T @ R[k] for k in range(K)]) / (n0 * K)
    tmax = float(np.max(np.linalg.norm(G, axis=1)))
    lamt = fraction * tmax
    D = cp.Variable((p, K))
    d = cp.Variable(K) if intercept else np.zeros(K)
    loss = sum(cp.sum_squares(Ys[0][:, k] - d[k] - designs(Xs, 0, k) @ (W[:, k] + D[:, k])) for k in range(K))
    iters, tol = solve(cp.Problem(cp.Minimize(loss / (2 * n0 * K) + lamt * cp.mixed_norm(D, 2, 1))))
    Dv, dv = D.value, (d.value if intercept else np.zeros(K))
    obj = sum(np.sum((Ys[0][:, k] - dv[k] - designs(Xs, 0, k) @ (W[:, k] + Dv[:, k])) ** 2) for k in range(K))
    obj = float(obj / (2 * n0 * K) + lamt * np.sum(np.linalg.norm(Dv, axis=1)))
    return Dv, dv, lamt, tmax, obj, iters, tol


def write_matrix(path, A):
    A = np.atleast_2d(A)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in A:
            w.writerow([repr(float(v)) for v in row])


def draw(rng, n, p, K, task_designs, shifts, scales=None, means=None):
    """Uniform designs (portable). The target's coefficients share three rows with
    the sources; shifts[l] adds source-specific rows."""
    L = len(n) - 1
    base = np.zeros((p, K))
    base[0, :] = 1.0
    base[1, :] = np.linspace(0.5, 1.5, K)
    base[2, :] = -0.8
    Xs, Ys = [], []
    for l in range(L + 1):
        col = np.linspace(0.5, 2.0, p) * (1.0 if scales is None else np.asarray(scales[l]))
        def one():
            return rng.uniform(-1, 1, size=(n[l], p)) * col + 0.2 * l
        X = [one() for _ in range(K)] if task_designs else one()
        Bl = base + (shifts[l] if l in shifts else 0)
        mu = np.zeros(K) if means is None else np.asarray(means[l], dtype=float)
        Y = np.column_stack([mu[k] + (X[k] if task_designs else X) @ Bl[:, k] + 0.4 * rng.uniform(-1, 1, size=n[l])
                             for k in range(K)])
        Xs.append(X)
        Ys.append(Y)
    return Xs, Ys


def row(p, K, j, v):
    S = np.zeros((p, K))
    S[j, :] = v
    return S


# name, seed, n (target first), p, K, task designs, intercept, contrast weights, levels as
# fractions of lambda_max, identified, extra draw arguments
CASES = [
    ("unequal-sizes", 1, [25, 40, 60], 7, 3, False, True, "sqrt", (0.4, 0.08, 0.01), True,
     dict(shifts={1: row(7, 3, 4, 1.8), 2: row(7, 3, 5, -1.6)})),
    ("task-designs", 2, [20, 30, 30], 6, 2, True, False, "linear", (0.3, 0.06), True,
     dict(shifts={2: row(6, 2, 3, 0.8)})),
    ("target-constant", 3, [20, 40], 6, 2, False, True, "sqrt", (0.3, 0.01), False,
     dict(shifts={1: row(6, 2, 4, 1.8)})),
    ("curvatures", 4, [30, 35, 45], 6, 3, False, True, [2.0, 5.0], (0.35, 0.07), True,
     dict(shifts={1: row(6, 3, 3, 0.9)}, scales=[np.ones(6), np.linspace(2.0, 0.3, 6), np.linspace(0.4, 3.0, 6)])),
    ("domain-means", 5, [24, 36, 36], 6, 2, False, True, "sqrt", (0.3, 0.06), True,
     dict(shifts={2: row(6, 2, 4, -0.6)}, means=[[2.0, -1.0], [-3.0, 0.5], [6.0, 4.0]])),
    ("cancellation", 6, [20, 40, 40], 6, 2, False, True, "sqrt", (0.2, 0.03), True,
     dict(shifts={1: row(6, 2, 3, 1.2), 2: row(6, 2, 3, -1.2)})),
]

CORRECTION_FRACTION = 0.3


def main(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "versions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for k, v in (("python", sys.version.split()[0]), ("numpy", np.__version__), ("cvxpy", cp.__version__),
                     ("clarabel", clarabel.__version__)):
            w.writerow([k, v])
    for (name, seed, n, p, K, task_designs, intercept, kind, fractions, identified, extra) in CASES:
        rng = np.random.default_rng(seed)
        L = len(n) - 1
        Xs, Ys = draw(rng, n, p, K, task_designs, **extra)
        if name == "target-constant":
            Xs[0] = Xs[0].copy()
            Xs[0][:, p - 1] = 1.7          # constant in the target only
        a = contrast_weights(kind, n)
        lmax = lambda_max(Xs, Ys, n, K, a, intercept)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        for l in range(L + 1):
            if task_designs:
                for k in range(K):
                    write_matrix(os.path.join(d, f"X{l}_{k + 1}.csv"), Xs[l][k])
            else:
                write_matrix(os.path.join(d, f"X{l}.csv"), Xs[l])
            write_matrix(os.path.join(d, f"Y{l}.csv"), Ys[l])
        write_matrix(os.path.join(d, "contrast_weights.csv"), a.reshape(-1, 1))
        with open(os.path.join(d, "settings.csv"), "w", newline="") as f:
            w = csv.writer(f)
            for k, v in (("sources", L), ("responses", K), ("predictors", p), ("nobs", ";".join(map(str, n))),
                         ("task_designs", str(task_designs).lower()), ("intercept", str(intercept).lower()),
                         ("contrast_weights", kind if isinstance(kind, str) else "vector"),
                         ("lambda_max", repr(lmax)), ("levels", len(fractions)),
                         ("identified", str(identified).lower())):
                w.writerow([k, v])
        for i, frac in enumerate(fractions, start=1):
            lam0 = frac * lmax
            B, b, obj, iters, tol = fused(Xs, Ys, n, K, a, lam0, intercept)
            W = sum(n[l] / sum(n) * B[l] for l in range(L + 1))
            for l in range(L + 1):
                write_matrix(os.path.join(d, f"B{l}_{i}.csv"), B[l])
                write_matrix(os.path.join(d, f"intercept{l}_{i}.csv"), b[l].reshape(1, -1))
            write_matrix(os.path.join(d, f"W{i}.csv"), W)
            Dv, dv, lamt, tmax, cobj, citers, ctol = correction(Xs, Ys, n, K, W, intercept, CORRECTION_FRACTION)
            write_matrix(os.path.join(d, f"D{i}.csv"), Dv)
            write_matrix(os.path.join(d, f"correction_intercept{i}.csv"), dv.reshape(1, -1))
            with open(os.path.join(d, f"level{i}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                for k, v in (("lambda0", repr(float(lam0))), ("objective", repr(obj)), ("iterations", iters),
                             ("tolerance", repr(tol)), ("lambda_tilde", repr(float(lamt))),
                             ("lambda_tilde_max", repr(tmax)), ("correction_objective", repr(cobj)),
                             ("correction_iterations", citers), ("correction_tolerance", repr(ctol))):
                    w.writerow([k, v])
            rows = [int(np.sum(np.linalg.norm(B[l] - (B[0] if l else 0), axis=1) > 1e-8)) for l in range(L + 1)]
            print(f"{name} level {i}: objective {obj:.12g} (tol {tol}), nonzero rows of B0 and contrasts {rows}, "
                  f"W rows {int(np.sum(np.linalg.norm(W, axis=1) > 1e-8))}, correction rows "
                  f"{int(np.sum(np.linalg.norm(Dv, axis=1) > 1e-8))} (tol {ctol})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
