# Discovery competitive pilot

User-authorized scope: 20 independent datasets per scenario; estimate the cost
of a later 1,000-dataset comparison. Do not submit the full comparison.

Four scenarios cross Gaussian/logistic outcomes with `(n,p,rho,s)` equal to
`(200,500,.3,5)` and `(100,500,.9,15)`. Covariates are Gaussian AR(1), centered
and scaled to population SD one before responses are generated. The nonzero
coefficients alternate signs, with magnitudes .5 (Gaussian) or .7 (logistic).
Gaussian noise SD is one, intercept .2. Fresh covariates and responses are
generated each replication. All implementations consume the same CSV inputs.

Inference requests ten predeclared slopes: first five active and first five null.
hdi, EconML and authors' sslasso compute all p slopes internally; Julia and SIHR
support target subsets. Performance is for this inference task, not an isolated
equivalent solver kernel. The pilot does not establish all-p SIHR/QP scaling.

Default means each package's statistical defaults, with RNG and CPU controls.
Julia logistic requires an explicit lambda, supplied from the matched rule and
reported as such. Matched means common nominal penalties on each method's
working design: Gaussian initial .5 sqrt(2 log(p)/n), logistic initial .1 times
that universal level, projection at the universal level. An initial interface
canary at .5 times the projection level exposed an infeasible QP in the strongly
correlated design; that attempt is retained separately and excluded from the
final pilot. The final projection level is fixed before the 20-dataset pilot.
Methods use
different projection objectives and noise/variance estimators. For SIHR matched,
rescale=1 and prob.filter=0. Native hdi matched GLM is unsupported because its
internal Fisher preprocessing lacks a public matching hook. EconML and sslasso
are Gaussian-only. No output is imputed for unsupported combinations.

Original sslasso contains an undefined `Las` variable in its explicit-lambda
helper, and its diagonal QP fast path uses the branch threshold `mu0` where the
supplied `mu` is required. Matched mode replaces the helper with exact-lambda
glmnet and corrects this one fast-path expression. The iterative inverse solver,
noise estimator, interval inflation and testing code are retained. Default mode
uses the original code. The intermediate matched results with the original
nonoptimal fast path are retained separately; only affected matched sslasso
fits are repeated after the main pilot. Post-timing feasibility and primal-dual
gap diagnostics check the corrected reference on the requested targets.

Each fit runs in its own single-CPU process under an exclusive srun step, with
BLAS/OpenMP threads set to one. Small independent warm-up precedes timed fitting.
Linux child-resource usage records end-to-end process time and peak RSS. Errors/timeouts/warnings
are preserved and summarized. Final fits have a 600-second process timeout;
the initial interface canary used 240 seconds.

Use GNU Parallel in one Slurm allocation; no local numerical execution. Reuse
the validated Julia source/environment from run 20261025-cd77-inference and the
run-local R/Python environment. Exact versions, submitted scripts, dataset
hashes, output records and Slurm accounting belong in the run folder/registry.

Cost estimates multiply measured per-configuration means by 1,000, with warmed
fit and fresh-process costs reported separately. Missing/failed configurations
make totals incomplete; target count, dimensions and tuning must accompany any
extrapolation. Twenty datasets provide throughput information, not a definitive
ranking of confidence calibration or power.
