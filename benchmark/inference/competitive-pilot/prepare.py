"""Generate shared pilot data on a Discovery compute node."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
cases = []
tasks = []
for family in ("gaussian", "binomial"):
    for label, n, p, rho, s in (("moderate", 200, 500, .3, 5),
                               ("correlated", 100, 500, .9, 15)):
        scenario = f"{family}-{label}"
        config = dict(scenario=scenario, family=family, n=n, p=p, rho=rho, s=s,
                      targets=list(range(5)) + list(range(s, s + 5)),
                      lambda_main=.1 * np.sqrt(2 * np.log(p) / n) if family == "binomial"
                                  else .5 * np.sqrt(2 * np.log(p) / n),
                      lambda_node=np.sqrt(2 * np.log(p) / n),
                      reps=20, noise_sd=1, intercept=.2)
        cases.append(config)
        for rep in range(1, 21):
            # Same covariates across response families; independent across replications.
            seed = 77260000 + (1000 if label == "correlated" else 0) + rep
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((n, p))
            for j in range(1, p):
                X[:, j] = rho * X[:, j-1] + np.sqrt(1-rho*rho) * X[:, j]
            X -= X.mean(axis=0)
            X /= np.sqrt(np.mean(X*X, axis=0))
            beta = np.zeros(p)
            beta[:s] = (.7 if family == "binomial" else .5) * (-1.) ** np.arange(s)
            eta = .2 + X @ beta
            if family == "gaussian":
                y = eta + rng.standard_normal(n)
            else:
                prob = 1 / (1 + np.exp(-eta))
                y = (rng.random(n) < prob).astype(float)
            dest = ROOT / "data" / scenario / str(rep)
            dest.mkdir(parents=True, exist_ok=True)
            np.savetxt(dest / "X.csv", X, delimiter=",")
            np.savetxt(dest / "y.csv", y, delimiter=",")
            np.savetxt(dest / "beta.csv", beta, delimiter=",")
            np.savetxt(dest / "targets.csv", np.array(config["targets"]) + 1, fmt="%d")
            with open(dest / "config.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[k for k in config if k != "targets"])
                w.writeheader(); w.writerow({k: v for k, v in config.items() if k != "targets"})
            for engine in (("ours", "oursqp", "hdi", "econml", "sslasso", "SIHR")
                           if family == "gaussian" else ("ours", "hdi", "SIHR")):
                for mode in ("default", "matched"):
                    # Native hdi GLM transforms the data internally, without a
                    # public hook for matched initial Fisher weights. Keep its
                    # native default, and record matched GLM as unsupported.
                    tasks.append((engine, mode, scenario, str(rep)))
with open("cases.json", "w") as f:
    json.dump(cases, f, indent=2)
with open("tasks.tsv", "w") as f:
    for row in tasks:
        f.write("\t".join(row) + "\n")
with open("data.sha256", "w") as f:
    for path in sorted((ROOT / "data").glob("*/*/*.csv")):
        f.write(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path.relative_to(ROOT)) + "\n")
print(f"DATA_COMPLETE: {len(cases) * 20} datasets, {len(tasks)} tasks", flush=True)
