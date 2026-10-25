# Discovery results, 25 October 2026

Run `20261025T0910Z-cd81`, Julia 1.13.2, AMD EPYC 7513 (`znver3`), one
Julia thread and one BLAS thread per worker. Five independent data seeds,
three interleaved timing rounds per seed. Each cell below is the median of
the five within-seed timing medians; the range is over those five medians.
Allocated MB means total allocation per call, not resident memory.

| q | method | median seconds | replication range | median allocated MB |
|---:|---|---:|---:|---:|
| 10 | rank | 0.0015841 | 0.0015347–0.0018427 | 1.04 |
| 10 | nuclear | 0.0019695 | 0.0019287–0.0020193 | 1.36 |
| 10 | nuclear_path | 0.0056229 | 0.0053105–0.0060891 | 4.25 |
| 10 | sparse | 0.0016987 | 0.0015665–0.0017577 | 2.43 |
| 10 | sparse_svd | 0.0059659 | 0.0058509–0.0063996 | 12 |
| 100 | rank | 0.0024593 | 0.0023711–0.0024616 | 2.43 |
| 100 | nuclear | 0.021012 | 0.019157–0.02209 | 8.5 |
| 100 | nuclear_path | 0.082267 | 0.076341–0.084247 | 31.9 |
| 100 | sparse | 0.0029283 | 0.0028634–0.0029442 | 4.25 |
| 100 | sparse_svd | 0.026572 | 0.024814–0.028568 | 48.9 |
| 1000 | rank | 0.008428 | 0.0079054–0.0086167 | 15.1 |
| 1000 | nuclear | 0.11349 | 0.1089–0.12084 | 81.6 |
| 1000 | nuclear_path | 0.44675 | 0.40847–0.45413 | 293 |
| 1000 | sparse | 0.012302 | 0.011259–0.012616 | 22 |
| 1000 | sparse_svd | 0.19156 | 0.17561–0.19665 | 419 |

All 225 measurements converged. Five independent data replications, three rounds each.
Profiles: 75 nonempty text reports (five data replications for each method and response size).

The nuclear path has five penalty levels. Sparse-SVD regression uses eight
candidate penalties per component. All methods include centering and setup.
Compilation is warmed up before timing. Profiles use ten subsequent calls
per method/size/seed, and are excluded from timing.

At q = 1000, ProximalBase's LAPACK SVD frame accounts for 63.3% of self
samples for nuclear regression and 62.5% for its path. Sparse-SVD regression
spends 27.5% of self samples in `GenericMemory` allocation and 15.7% in
LAPACK SVD. These are descriptive sampling shares over five independent
replications; see `profiles-summary.txt` for counts and `profiles.txt` for
all 75 complete reports. Short q = 10 fits have fewer samples. Every profile
has samples (12–4555 snapshots per report).

Validation on Discovery:

- Julia 1.10: 2,337 prefix checks and 303 resumed remaining checks passed,
  including Aqua; the interruption was a missing direct dependency in the
  custom run environment. No numerical test failed.
- Julia 1.13: the complete package suite passed 2,640/2,640 checks.
- The final low-rank API and golden suite separately passed 153/153 checks
  on both Julia versions (including AIC/BIC normalization).
- The independent oracle passed 47/47 checks. Identifiable RRR coefficients
  agree to 6.7e-15, row-sparse coefficients to 2.9e-11 and selected RSSVD
  coefficients to 3.3e-9. Wide-design fitted values and exact degrees of
  freedom also agree. See the fixture README for the nonunique coefficient
  convention and rrpack 0.1-14 BIC/BICP-label workaround.
- The documentation build passed in a separate final docs job.

Measurements are from job 13423245: all five worker steps completed with
exit 0. That allocation exited nonzero because its preceding documentation
build found a relative link outside Documenter's build tree. The link was
fixed and documentation was validated separately by job 13423277; measurements
were retained and were not repeated. Earlier harness failures and their
receipts remain in the run registry rather than being discarded.

`metadata.json` records source hashes, dependency revisions, job IDs, and the
local/remote run and registry paths. `rep*.csv` contains all 225 observations.
The Julia source hashes of the measured solvers and driver match implementation
commit `70951d9`; later changes only add reporting, fix the docs link, and make
the standalone benchmark environment resolve its unregistered dependencies.
