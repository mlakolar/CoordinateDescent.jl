# Nuclear PANOC controlled calibration

One independent new dataset per design, rep0 excluded from a later20-input holdout. Three warmed rounds per workload, fixed EPYC7513/single-thread stack. All336 original SciPy primal-dual gap and proximal-gradient certificates pass1e-6; corrected timing outputs are byte-identical to those certified arrays. No production optimization or solver-default change has been adopted.

Initial pilot13450595 completed3m00s. Two harness defects were found before drawing speed conclusions: native/APG public controls prepared data twice (12 affected controls), and repeated local named fit definitions made cached labels call the public closure (96 controls). Targeted corrections13450643(39s) and13450661(42s) reran only those108 controls on the same inputs. Preserve initial/v2/v3 tables, and do not count corrections as new inputs/certificates. The84 other public controls and all paths remain valid.

| Case | Native public ms | Controlled PB95/native | PO95/PB95 | Gram/PB95 | Shared-residual/PB95 | Final-reuse/PB95 | Combined/native |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 2.0019 | 0.982 | 1.038 | 0.787 | 1.092 | 0.992 | 0.781 |
| medium | 16.5089 | 0.962 | 1.022 | 0.868 | 1.080 | 0.988 | 0.821 |
| responses | 109.4228 | 0.872 | 1.086 | 0.749 | 1.100 | 0.961 | 0.649 |
| wide | 209.2860 | 0.959 | 1.036 | 0.920 | 0.920 | 0.989 | 0.882 |
| ill | 72.6690 | 0.987 | 1.019 | 0.836 | 1.109 | 0.992 | 0.829 |
| deficient | 21.5546 | 1.020 | 0.989 | 0.841 | 1.059 | 0.966 | 0.845 |

Ratios from one input are diagnostic signals, not repeatable improvement evidence. The compiled controlled PB95 wrapper can differ from the native extension despite equivalent mathematics/outputs; use controlled one-factor ratios for attribution and actual before/after production APIs for adoption. The many-response native/control difference remains visible rather than being assigned entirely to prox/diagnostics. Cached controls now explicitly exclude centering/gradient setup, with an allocation witness and exact B equality.

The Gram/shared-residual proposal merits a guarded production prototype and20-input test. A quadratic Gram loss value can cancel near a tiny residual loss, so the calibration expression alone is insufficient for adoption; examine stable direct-residual fallback and Float32/64 near-fit behavior. Stable shared residual work helps wide designs but is adverse on tall designs, so a universal residual-gradient switch is unsupported. Final residual/singular-value reuse gives smaller effects; preserve rank threshold/support/intercept/DF/objective semantics. Quadratic-trait call counts, cadence sensitivity, unsafe boundary-step controls and all profiles are retained. Keep certificate tolerance1e-8/cadence10/standard0.95/L and APG default until broader evidence justifies a change.

Every8-level calibration path point is independently certified, including paths with nonuniform timing results. Complete20-input cold strong/middle/weak and path evidence, independent certificates, regression controls, all-target fresh-process memory/profiles, and optional-extension/Float32/64/zero/iteration-limit/CV tests remain required by issue103. No inference or universal solver ranking is established.

Available corrected public CPU profiles: 96; empty: 0. Short calibration profiles can be empty; future20-input target profiles must use bounded repeated sampling outside timers. Raw coefficient matrices and binary inputs remain on Discovery. This reporting delivery contains original/corrected tables, certificates, call counts, profiles, logs and frozen manifests.
