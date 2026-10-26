# Issue106 adapter investigation, before implementation

2026-10-26. User campaign compute-as-needed authorization applies. No106 code,
worktree, environment mutation, jobs or evaluation inputs exist at this checkpoint.
Current remote master295fbf5420d6384a1ca261023367ee86d0a40b8c was rechecked.
107's main/runtime pilot gate remains the active task;106 precedes108.

Primary sources inspected:
- https://github.com/biona001/Knockoffs.jl/blob/v3.1.4/src/utilities.jl
- https://github.com/biona001/Knockoffs.jl/blob/v3.1.4/src/modelX.jl
- https://github.com/biona001/Knockoffs.jl/blob/v3.1.4/Project.toml
- https://github.com/biona001/Knockoffs.jl/blob/v3.1.4/LICENSE
- https://github.com/mlakolar/HighDimensionalModels.jl/issues/106

Knockoffs3.1.4 UUID878bf26d-0c49-448a-9df5-b057c815d613, MIT license, is the
existing historical comparator. Its solve_s(Symmetric(Sigma),strategy) maps to
correlation scale, calls solve_MVR / solve_max_entropy and maps back. Both
optimizers return only s, without a convergence flag or iteration count. Their
niter100/tol1e-6/lambda_min1e-6 defaults cannot be advertised as certified
convergence just because the function returns. A package extension must either
independently certify the returned original objective or explicitly reject an
uncertified optimizer result. Do not copy the optimizer wholesale or silently
fall back to another strategy. Pin an actual commit and installed source hashes
before running comparisons; a tag and this planning note are not provenance.

The inspected single-knockoff external condition() uses row normals times a
lower Cholesky factor. This retains the historical orientation problem; use
only its diagonal optimizer with our independently validated sampler. Any
external full sampler must pass covariance/root identities before statistical
rankings. The multiple-knockoff branch is outside this issue's implementation.

Proposed route to specify before coding: weakdep/extension for the established
diagonal optimizer, with an explicit owned diagonal result and actual strategy,
backend/version/options, independent feasibility and optimality diagnostics.
Keep model type Gaussian/fixed/provided separate from diagonal strategy. A
sentinel absent strategy should resolve to equicorrelation if s is absent and
to user if s is supplied, allowing explicit conflicting combinations to fail.
Forward noise/strategy/diagonal/optimizer options as named construction options,
never CDOptions. Validate conflicts and solver options before expensive work.
Keep low-level matrix-returning samplers and old result constructors compatible.
Equicorrelation remains the compatibility default until scoped evidence supports
a separate change. Cache ownership/fast factorization are issue108, not106.

Independent original correlation-scale objectives to verify before adoption:
for D=diag(s), M=2R-D, joint covariance is orthogonally equivalent to diag(M,D).
MVR minimizes tr(inv(M))+sum(1/s), gradient diag(inv(M)^2)-1/s^2.
Maximum entropy maximizes logdet(M)+sum(log(s)); its negative gradient is
 diag(inv(M))-1/s. These interior formulas require positive s and SPD M.
Independently check objective, stationarity and feasibility rather than relying
on iteration equality. Boundary cases need appropriate feasible/KKT treatment.
Investigate the optimizer's lambda_min perturbation on the ORIGINAL objective;
never use it to weaken our scale-relative joint/conditional PSD validation.
Float32 output rounding must also be validated after original-unit mapping.

Remaining work: freeze API/protocol and correct objective certificates; prepare
isolated source/env; release/LTS construction/forwarding/failure tests; reference
Knockoffs and applicable R objective comparisons; >=1000 datasets/main scenario
across7 processes with common X/y/innovations and fixed lasso statistic/penalty;
signs/global-null/covariance-stress cases;7-process runtime profiles separating
optimizer/sampling/filter cost. All numerical work under Discovery Slurm. No
implementation recommendation or default/statistical guarantee is established.

## Prototype API fixed before implementation

`knockoff_diagonal(Sigma; strategy=nothing, s=nothing, construction_options=(;))`
returns an owned `KnockoffDiagonal` with s, actual strategy and diagnostics.
The same named strategy/s/options are accepted by Gaussian/fixed constructors
and forwarded through `knockoff_filter`. Missing strategy resolves to
:equicorrelated if s is absent, :user if supplied. Explicit conflicts reject;
:user requires s, other strategies forbid s. Choices :mvr/:maxent load only
through a Knockoffs weakdep extension. Loading it never delegates sampling.
Provided Xk forbids strategy/s/noise/construction options, while method remains
:gaussian/:fixed/:provided. Construction options never become CDOptions.

Knockoffs3.1.4 annotated tag resolves to commit
d23942e9af9103f93be9a20e47ae6f7cad685aa5. Backend defaults to investigate in
prototype canary: niter1000/tol1e-10/lambda_min1e-10/robustfalse, with original
objective componentwise relative stationarity <=max(1e-6,100eps(input T)).
These are provisional pre-evaluation optimizer controls, not adopted defaults
or proof of statistical superiority. Backend actual iterations/status are
unavailable and must be labeled as such. Independent optimality/feasibility
failure raises an error; no silent fallback, PSD projection or solver copying.
Optimize in Float64 on correlation scale, map to original covariance units,
then verify the actual rounded diagonal. Equicorrelation preserves existing
arithmetic/type behavior exactly, including Float32's existing promotion.

Existing matrix-returning constructors remain matrix-returning. Result gains
construction metadata including actual diagonal/strategy/certification, with
old positional constructors preserved as unspecified metadata. A standalone
diagonal result is passed to samplers explicitly as s=result.s; this is not
an owned covariance/factor cache. Changing Sigma requires correct validation.
Issue108 owns caching/fast factorization, not this change.

First release/LTS canary covers old golden/regression checks plus selector
conflicts, forwarding, missing extension, input/RNG preservation, old result
constructors, Float32/64 and scale-relative boundary/failure cases. Extension
checks independently reconstruct objectives/gradients/covariance/root/Gram
identities; optional backend failure is explicit. Main protocol and fresh
>=1000/scenario statistical/runtime cohorts are frozen only after the canary.
