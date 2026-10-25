# Run provenance

Run `20261025-cd81-optcomp` used the approved Python/Julia/R optimization-only
scope, five independent inputs per case, and no MATLAB. The task registry is
`.slurm/runs/20261025-cd81-optcomp.json` in the parent juliaPkg workspace.
The retained remote root is
`/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp`.

The measured package is commit `9b7edcf315f173e862b049519a9731d3c8f02925`
(implementation commit `70951d9544cacb7c7421c061a595bc067a07cd17`).
ProximalBase is `0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a`;
ProximalSolvers is `9041e236f3de8ffdc02103e5a505993463e00bb1`.
No unrelated changes in the parent workspaces entered the numerical environment.

| Job | Purpose | Outcome |
|---|---|---|
| 13423556 | Environment preparation | Completed, 242 s |
| 13423582 | Initial pilot | Failed, 16 s; Python reserved property and Julia parse error corrected |
| 13423591 | Corrected pilot | 53 s; numerical fits/reference completed; summary JSON serialization failed |
| 13423647 | Factor certification | Completed, 43 s; actual-factor checks avoid arbitrary null-space completions |
| 13423662 | FISTA stopping calibration | Completed, 11 s; stricter stopping meets independent gap target |
| 13423672 | Initial main launcher | 113 s; valid isolated large R runs, missing ordinary workers because PARALLEL environment variable was reserved |
| 13423683 | Corrected main launcher | Completed, 1299 s; full original Python/Julia/R coverage |
| 13423704 | Adaptive weights | Completed, 33 s |
| 13423727 | R precision and QR route | Completed, 174 s; all R timings/profiles/memory repeated, old records retained |
| 13423905 | R adaptive precision and coefficient-only rank | Completed, 31 s |
| 13423914 | Reporting aggregation | Completed, 1 s |
| 13423932 | Shared-loss-target rank projection | Completed, 32 s; final aggregate |

These non-overlapping allocations used 34.13 running-union minutes and
1.969 allocated CPU-hours, including failed attempts and environment preparation.
Peak requested resources were 4 CPUs/12 GB. Ordinary steps used 1 CPU/2 GB;
large R component steps were serialized at 1 CPU/8 GB. The prior implementation
campaign had used 30.4 minutes and 1.976 CPU-hours, giving task totals of
64.53 running-union minutes and 3.945 CPU-hours. The comparison stayed within
its separately approved 60-minute ceiling; recovery reserve was unnecessary.
Accounting timestamps use the scheduler's local time; elapsed fields determine cost.

The original main Python/Julia workers are from source-v6.tar at the remote
root. R measurements are from the separately staged higher-resolution worker.
Source snapshots v1–v6 and superseded measurements remain there for diagnosis.
The exact currently committed numerical worker bytes match measured hashes:

| Measured remote name | Committed harness name |
|---|---|
| code/julia_worker.jl | julia_worker.jl |
| code/python_worker.py | python_worker.py |
| code/common.py | common.py |
| code/r_worker_hires.R | r_worker.R |
| code/adaptive.jl | adaptive.jl |
| code/adaptive_hires.R | adaptive.R |
| code/rank_core.jl | rank_core.jl |
| code/rank_matched.jl | rank_matched.jl |
| code/finalize.py | finalize.py |

`measured-source-sha256.txt` verifies those bytes. The other hash files record
the primary and supplemental submissions. Launcher/reporting adjustments can
postdate a measured worker; their role is distinguished from numerical code.
JuliaProject.toml/JuliaManifest.toml refer to frozen sources in the prior run,
not a changing parent checkout. Registered dependencies are pinned by the
manifest; Python versions are in python-freeze.txt. All Julia imports and
precompilation occurred in Slurm allocations.

Raw inputs, first-round coefficients/factors, prior failed/capped outcomes,
and job logs stay in the run directory. Compact committed artifacts include
all accepted timing rows and independent checks; profile and diagnostic
archives combine hundreds of small reports for practical transfer. They do
not replace the matrices or the recreation recipe.

Committed CSVs normalize line endings to LF and printed text removes trailing
whitespace; data fields and numerical worker bytes are unchanged. The local
transfer receipt and remote run retain the original reporting bytes.
