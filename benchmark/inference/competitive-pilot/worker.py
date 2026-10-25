"""One independent process for one EconML fit (Discovery only)."""
import csv
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
import numpy as np
from scipy.stats import norm
from econml.sklearn_extensions.linear_model import DebiasedLasso

engine, mode, scenario, rep = sys.argv[1:]
dest = Path("data") / scenario / rep
out = Path("outputs") / f"{engine}-{mode}-{scenario}-{rep}"
config = next(csv.DictReader(open(dest / "config.csv")))
X = np.loadtxt(dest / "X.csv", delimiter=",")
y = np.loadtxt(dest / "y.csv", delimiter=",")
targets = np.loadtxt(dest / "targets.csv", dtype=int) - 1
seed = 770000 + int(rep)
record = dict(engine=engine, mode=mode, scenario=scenario, rep=int(rep), status="error")

def fit(A, b):
    kw = dict(random_state=seed, n_jobs=1)
    if mode == "matched":
        kw.update(alpha=float(config["lambda_main"]), alpha_cov=float(config["lambda_node"]),
                  max_iter=100000, tol=1e-7)
    return DebiasedLasso(**kw).fit(A, b)

try:
    # Small, independent warm-up excludes import/first-use initialization.
    rng = np.random.default_rng(177)
    fit(rng.normal(size=(50, 12)), rng.normal(size=50))
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        started = time.perf_counter()
        model = fit(X, y)
        elapsed = time.perf_counter() - started
    low, high = model.coef__interval(alpha=.05)
    se = np.asarray(model.coef_stderr_)
    bhat = np.asarray(model.coef_)
    pvalues = 2 * norm.sf(np.abs(bhat / se))
    np.savetxt(str(out) + ".csv", np.column_stack((targets+1, bhat[targets], se[targets],
               low[targets], high[targets], pvalues[targets])), delimiter=",",
               header="target,estimate,se,lower,upper,pvalue", comments="")
    record.update(status="ok", fit_seconds=elapsed,
                  warnings=[str(w.message) for w in ws],
                  warning_categories=[w.category.__name__ for w in ws],
                  alpha=(None if model.selected_alpha_ is None else float(model.selected_alpha_)),
                  actual_inferred_targets=X.shape[1])
except Exception as e:
    record.update(error=str(e), traceback=traceback.format_exc())
Path(str(out) + ".json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
