# Prototype correctness, not statistical adoption evidence

Discovery Slurm13454092 completed0:0 in2m09s, two1CPU workers, Julia1.10.12
and1.13.2, fixed EPYC7513, one Julia/BLAS thread each. Sourcece00fa70d5913cd327c99f225997af135cd5bc75.
Pinned environments reuse the campaign PB/PS/KS dependencies, with only the HDM
path changed to the isolated source. Existing selection/sparse/R-golden checks
and493 new Gaussian calibration checks pass on EACH version. The adjacent
archive retains complete logs and actual source/CPU/thread hashes.

New checks cover independent two-term quantile/recurrence values, nonincreasing
finite nonnegative weights, scaling/zero sigma, small/wide dimensions,
intercept dimension/scaling, Float32/64 dense/sparse fit/path propagation,
constant/offset columns, explicit-weight precedence, metadata, legacy result
constructors and input preservation. Existing golden checks compare common
explicit weights with R. They do not establish native Gaussian weight identity.

The statistical cohort, common-weight Gaussian comparator checks, independent
full-fit/intercept certificates, runtime/profile replications, and justified
final defaults are still required by DESIGN.md. No statistical improvement,
general FDR guarantee or default change follows from this canary. Keep draft.
