# Explicit Gaussian SLOPE calibration: frozen prototype contract

Issue107. Baseline master295fbf5420d6384a1ca261023367ee86d0a40b8c, verified
through the GitHub branch API before creating this isolated worktree. The user
authorized the whole issue campaign and compute as needed; this supersedes the
issue's historical note that opening an issue alone did not authorize jobs.

## Formula and API chosen before simulation

Public `slope_gaussian(p; n, q=.1, sigma=1, intercept=false)` uses the published
one-based denominator m-i, with m=n-intercept. Dimensionless BH quantiles use
erfcinv to avoid cancellation of 1-small_probability. Accumulate squared adjusted
weights until the first nondecrease or a nonpositive denominator, then flatten
the remaining tail at the preceding value. Scale the entire sequence by
sigma/sqrt(n), not sqrt(m), to match the package loss /n and column mean-square
normalization. This explicitly defined first-nondecrease convention avoids
claiming an unproved identity to every global-minimum/cumulative-minimum rule.
Require m>=2 when p>1, m>=1 when p=1. Sigma0 gives zero weights; automatic paths
still need a nonzero base sequence unless explicit factors are supplied.

`calibration=:bh/:gaussian/:user` is explicit in fits and paths. Explicit weights
win over either automatic rule and leave q/sigma unused. Metadata identifies the
actual and requested rule, supplied-sigma or user-weight convention, dimensions,
standardization and flattening. Existing coefficient accessors and old positional
result constructors are preserved. There is no new noise estimator or solver.
BH remains the compatibility default pending evidence; Gaussian adjustment does
not imply a correlated-design, estimated-sigma or path-selection FDR guarantee.

## Pinned primary references

- [Bogdan et al.2015, equations26–27 and w(k)](https://arxiv.org/html/1407.3824):
  w(k)=1/(n-k-1), so w(i-1)=1/(n-i). The derivation is approximate for Gaussian
  designs; the paper separately proves the orthogonal known-noise BH theorem.
- [SLOPE.jl1.3.1, commit14d2192879c72180a0fb92c661c61a064eedf572](https://github.com/jolars/SLOPE.jl/blob/14d2192879c72180a0fb92c661c61a064eedf572/src/regweights.jl):
  one-based denominator max(1,n-i-1), raw recurrence followed by cumulative minimum.
- [R SLOPE2.1.1, commit243741cb1b4161398d757adfdddc0caa472b9303](https://github.com/jolars/SLOPE/blob/243741cb1b4161398d757adfdddc0caa472b9303/src/slope/regularization_sequence.cpp):
  zero-based i=1 corresponds to second weight and denominator max(1,n-2).
  In one-based notation its denominator is max(1,n-i). Native Julia/R weight
  differences are deliberate comparator outcomes, not solver errors.

## Validation and planned evidence

First run release/LTS selection/sparse/golden plus new boundary/recurrence,
Float32/64, propagation, explicit-weight precedence, old constructor and sparse
storage/input-preservation tests. All numerical work, even these checks, runs
under Slurm. Source and environments are immutable for that allocation.

Before the evaluation cohort, freeze a complete simulation protocol and harness:
at least1000 independent datasets/main scenario across at least7 processes,
shared raw X/y and known sigma, iid/AR.4/AR.8/weak/wide/global-null plus exact
orthogonal controls; both same-sign and alternating signals. Estimated-noise and
misspecification studies are separate stress tests. Preregister dimensions,
signals, sequence rules and selection cutoffs before generation, and do not tune
the sequence on evaluation outcomes. Compare native/Julia/R at common weights
first, then each native weight convention separately. Independently certify all
measured Float64 fits with full relative proximal residual<=1e-7, including
intercept stationarity, objectives and selections. Preserve failures.

Runtime/profile evidence needs7 independent processes, multiple independent
inputs,3 timing rounds/input, rotated method order, fixed EPYC7513 and1 thread.
Separate weight construction and complete fit/path cost, and keep all validation
and profiling outside timers. Report FDR/power, false/true selections, paired
differences and Monte Carlo uncertainty; adverse correlated/wide results remain
part of the conclusion. This prototype is not ready for human review until that
evidence and complete hosted CI pass. No default-change or universal gain claim.
