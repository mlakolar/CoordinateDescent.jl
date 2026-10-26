# Morning report, 2026-10-26 Dubai

**EMPIRICAL. Gaussian inference now has a defensible problem-specific automatic recipe. Logistic and QP inference do not yet have a broadly calibrated default.** The evidence does not support a claim of95% coverage in every design or for every coefficient. The PR remains draft.

## Corrected independent Gaussian validation

A fresh validation with disjoint base1000000000 tested500 datasets in each of ten prespecified designs:5,000 datasets and50,000 marginal coefficient intervals. Source/protocol3eefe3b was pushed before generation. All fits were complete, finite and converged, with zero fit errors. Every prospective pooled engineering gate passed. Pooled active coverage was94.80–97.12%, with dataset MCSE0.36–0.45percentage points. Type I error was2.08–5.68%, with MCSE0.30–0.48points.

|Design|Coverage ± MCSE (%)|Type I (%)|Power (%)|Active interval length|
|---|---:|---:|---:|---:|
|Moderate|97.12 ±0.36|2.08|96.60|.5193|
|Correlated|96.84 ±0.39|3.76|13.80|1.9649|
|Benign|94.80 ±0.45|4.60|100.00|.2193|
|Random support|95.56 ±0.40|2.68|97.60|.5029|
|Weak signals|95.88 ±0.38|3.32|31.48|.4940|
|Larger n|95.64 ±0.42|5.68|100.00|.2336|
|Independent predictors|95.96 ±0.39|2.76|98.64|.4673|
|Heteroscedastic|96.16 ±0.38|3.56|99.84|.4006|
|Tested-predictor heteroscedasticity|96.24 ±0.39|3.00|99.76|.4236|
|t3errors|96.40 ±0.38|2.72|99.40|.3947|

Six active and four null target roles trigger the unadjusted Monte Carlo tolerance flags. The lowest observed active-role coverage is93.2%; benign null role10has7.2% rejection±1.16points. These flags are retained. Pooled calibration does not certify every marginal role. Correlated and weak-signal power is low, and correlated intervals are long. No joint or bootstrap coverage claim follows from this marginal experiment.

The automatic Gaussian recipe uses five-fold/40-levelCV-min, a regularized initializer, restrictedLDPE with four X-only selected unpenalized controls, a50%projection-SD budget,60projection levels to.001of conditional maximum, bias target sqrt(2log(p)), relaxation.1, and selected-designHC3 influences. Solver opt_tol is1e-8 and gaptol0. Expert initializers, penalties, projections, variance policies and preprocessing remain explicit controls. Naive selected-model refitting repeatedly worsened development coverage, so it is not the Gaussian automatic default. Refitting remains available.

Job13455264 took562 seconds with64CPUs/128GiB allocated. Actual worker CPU was7.38 hours; maximum individual-worker RSS was.891GiB. Warm ten-target fits range from.055 seconds in the benign design to8.56 seconds in tested-predictor heteroscedasticity. Compilation and process startup are separately scoped. The short13-second cancelled allocation attempt is retained in accounting.

[All ten designs, uncertainty, roles, resources and checks](results/discovery-20261026-gaussian-disjoint500/RESULTS.md).

## Two reproducibility corrections

The old stage bases were100000apart while expanded design offsets span200000. Some cross-design/stage seeds therefore overlap. This qualifies earlier claims that the entire confirmation collection was independent of development. Per-design estimates, failures and historical results remain preserved. The replacement validation uses a widely separated block; no tuning changed against its outcomes. [Metadata audit](results/discovery-20261026-seed-stream-audit/RESULTS.md).

The old projection RNG indexed seeds by requested target position and vector length. Asking for ten targets versus all slopes could change their numerical paths. Commit e13b40f assigns seeds by physical predictor.220 replicated subset/order/thread checks passed, along with the inference/golden/tuning/LDPE suites. A60 dataset/600 target bridge to saved full-all-target fits agreed within8.9e-16for pvalues,4.5e-16for estimates, and exactly forSE/projection penalties. The full ordered all-target stream is unchanged. Historical subset results remain separately labeled.

## Speed and competitors

The new automatic policy was timed over20 matched datasets in each of three designs with all slopes requested. Its warmed automatic fit includes its own tuning.

|Design|Slopes|Mean fit ± MCSE (s)|Separate initial CV mean (s)|Maximum worker RSS (GiB)|
|---|---:|---:|---:|---:|
|Moderate|500|229.072 ±2.461|.353|.896|
|Correlated|500|98.657 ±.769|.182|.875|
|Benign|100|1.031 ±.012|.027|.882|

There is no general automatic-policy speed advantage. Native hdi averaged204.595/57.035/8.591 seconds in these designs; EconML60.249/24.109/.952; sslasso46.214/17.507/.143. Those native statistical recipes have different calibration and solver-warning/certificate outcomes. SIHR and hdm are distinct ten-target pipelines. The historical20 dataset/140 fit matched-penalty engine experiment remains separate and cannot substantiate an automatic-policy speedup. Its EconML intercept-degrees-of-freedom scaling and QP certificates remain documented.

[Equal-target timing](results/discovery-20261026-all-target-time20/RESULTS.md) · [Native20pilot with all strict failures and MC errors](results/discovery-20261026-gaussian-native20/RESULTS.md).

The frozen native comparison of hdi,EconML,sslasso,SIHR and hdm at1,000 datasets in each of three Gaussian core designs is still running as13452003. It reuses the first20 fits per case and executes14,700 newfits on exactly hash-matched92300000inputs. Analysis13454095 is queued after it. The resource estimate was142consumedCPU-hours and3.4 planning wall-hours on64CPUs/128GiB, including50%margin but excluding input preparation, queueing and analysis. This covers three native core designs, not the complete ten-design defaults-and-matched campaign. The earlier own ten-design1,000arm is historical evidence and is explicitly qualified by the provenance corrections above.

## Logistic findings and direction

Native methods do not achieve nominal coverage uniformly in the current difficult regimes. The100 dataset native study retained every planned method/case group, including failures. SIHR active coverage was57.4%moderate and81.0%random; post-hdm84.2% and87.4%. The documented SIHR positive control gave95.5%, and benign/post-hdm gave95.2%. These settings show useful regime dependence, not a universal refitting solution.

The official DoubleMLLPLR sparse-learner20pilot completed240 datasets/480score workers. It returned839of960finite target records and121 target errors. Native score/sandwichSE parity was2.22e-16. Moderate active coverage was45%/55% for the two scores. Its favorable Liu n1000,p201positive control covered the active coefficient in20/20 datasets, but that result has a95%Wilson Monte Carlo lower bound of83.9%; it is not evidence of established95% coverage. DoubleML requires explicit nuisance learners. These are declared learners with native nested5x5cross-fitting and two requested roles, not universal package defaults or a paper-bootstrap replica. Consumed CPU was51.39 hours; maximum worker RSS.2515GiB. [Complete pilot](results/discovery-20261026-doubleml20/RESULTS.md).

The favorable Liu control is being expanded to100development datasets with unchanged algorithms and learner choices, reusing the first20outcomes. This is selected development expansion, not fresh confirmation. No logistic default changes from it. Native calibrated-lossRCAL also has a real computational limitation: its unchanged canary recovery timed out after two hours, leaving eight n800 target markers missing; partial results are preserved.

The calibrated exponential-loss audit found196 numerical recession witnesses among1,000LPs on20 datasets. Some proposed CV levels make that loss unbounded. Increasing iteration limits alone cannot repair those levels. A guarded Julia prototype rejects such levels and requires trainingKKT checks. All40 selected nuisance fits on20 datasets converged; coefficients agree withRCAL within6.2e-5and objectives within5.9e-10.38of40CVindices match. The maximum common-fold loss discrepancy is.1127, relative.00118. This is promising numerical development, not exactCVparity or inference calibration. [Checks](results/discovery-20261026-calibration-bounded20/RESULTS.md) · [Saved-fold audit](results/discovery-20261026-calibration-cv-saved-audit/checks.json).

The next substantive logistic work is a checked calibrated-nuisance pipeline and a faithful published-regime replication, including the published bootstrap interval construction. Compare Wald and bootstrap inference with the same conditional-coefficient estimand. Resolve nuisanceKKT/CV discrepancies before substituting the Julia backend. Sample-split/refined-GLM inference is another principled candidate. Bias-reduced/profile-likelihood post-selection refits remain an additional experiment; their selection step still needs validation. The native logistf installation is blocked by missing compiler headers in the R container, after the originalCRAN transport failure; no Firth coverage claim has been made.

Separate defaults by method and family remain appropriate. Gaussian has finite-design supporting evidence; logistic,QP options andPoisson remain experimental. No arbitrary SE inflation, truth-based tuning or deletion of failed regimes was used.

## Reproduction and implementation status

All numerical work ran on Discovery compute nodes. Source snapshots, raw coefficients, diagnostics, warnings, failures, joblogs and resource receipts are retained locally under the project run directory and on Discovery. Compact reporting inputs and recipes are in this branch. Exact-job accounting is cumulative in `.slurm/runs/20261026-cd77-defaults`, not a reset budget. The package source, target-stream correction and correctness tests are pushed to draftPR86. Hosted CI is reported separately from Discovery statistical validation. Nothing is merged automatically.
