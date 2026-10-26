# Morning report, 2026-10-26 Dubai

**EMPIRICAL. Gaussian inference now has a defensible problem-specific automatic recipe. Logistic and QP inference do not yet have a broadly calibrated default.** The evidence does not support a claim of95% coverage in every design or for every coefficient. The PR remains draft.

The published logistic positive control now gives near-nominal coverage for both our existing Julia default (94.8%) and the official DoubleML instrument score (95.8%) on the same500fresh datasets. A new experimental partialled-QP/full-scoreHC3 recipe also passed the ten-design pooled confirmation gates. Its production integration remains outstanding.

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

The three-core native comparison is complete:1,000datasets/design,15,000native fits,all finite intervals,11,728strict-valid fits. Current-source Julia adds3,000paired fits/30,000intervals with exact input-hash matches and zero fit errors. Moderate active coverage is96.18%for Julia,78.40%hdi,70.42%EconML,76.32%raw native sslasso,71.84%SIHR,and81.30%hdm. In the correlated design,Julia gives96.48%coverage/15.26%power/1.9648length;hdm94.32%/34.02%/1.2647. This shows the coverage/length/power tradeoff. Native sslasso has no strictly certified fits in the two high-dimensional cases;EconML has exposed convergence warnings,which may concern discardedCVlevels. Conditional native coverage must be read together with strict availability. Nothing was silently patched or dropped.

Native job13452003 took7,708seconds on64CPUs/128GiB and consumed135.67workerCPU-hours,maxindividualworkerRSS.776GiB. Current-source own refresh13455533 took172seconds,2.29CPU-hours,maxworkerRSS.913GiB. The original forecast142CPU-hours/3.4planning wall-hours was conservative. An old seven-task planned-count typo in the native analyzer was corrected to15,000with its original checks retained. [Full native comparison, paired MC uncertainty and timing/memory](results/discovery-20261026-gaussian-core-comparison1000/RESULTS.md). Seven native stress-design arms and refreshed matched-statistical-recipe campaigns are not complete;this is a three-core native comparison,not the entire proposed campaign.

## Logistic findings and direction

Native methods do not achieve nominal coverage uniformly in the current difficult regimes. The100 dataset native study retained every planned method/case group, including failures. SIHR active coverage was57.4%moderate and81.0%random; post-hdm84.2% and87.4%. The documented SIHR positive control gave95.5%, and benign/post-hdm gave95.2%. These settings show useful regime dependence, not a universal refitting solution.

The official DoubleMLLPLR sparse-learner20pilot completed240 datasets/480score workers. It returned839of960finite target records and121 target errors. Native score/sandwichSE parity was2.22e-16. Moderate active coverage was45%/55% for the two scores. Its favorable Liu n1000,p201positive control covered the active coefficient in20/20 datasets, but that result has a95%Wilson Monte Carlo lower bound of83.9%; it is not evidence of established95% coverage. DoubleML requires explicit nuisance learners. These are declared learners with native nested5x5cross-fitting and two requested roles, not universal package defaults or a paper-bootstrap replica. Consumed CPU was51.39 hours; maximum worker RSS.2515GiB. [Complete pilot](results/discovery-20261026-doubleml20/RESULTS.md).

The favorable Liu control was expanded to100 development datasets, then the unchanged primary instrument score and secondary nuisance_space score were frozen in c114663 before500 new datasets at disjoint Rbase1200000000. The primary instrument score passed its prospective gate:95.8%active coverage±.90percentage points,4.0%typeI±.88points,100%power,active interval length.3139,500/500complete. Secondary nuisance_space gave95.4%active coverage and3.6%typeI. NativeSEparity was4.16e-17. This consumed4.064CPU-hours, with maximum individual-workerRSS.239GiB and350seconds elapsed on64CPUs/64GiB. Warnings were150000retained FutureWarnings, with no convergence/fit errors. [Independent reference confirmation](results/discovery-20261026-liu-doubleml-confirmation500/RESULTS.md).

The existing Julia logistic default on the same500 fresh inputs gave94.8%active coverage±.99points,4.2%typeI±.90points,100%power and active length.2970. The unchanged restricted-LDPE expert recipe gave96.6%coverage±.81points,5.4%typeI±1.01points and active length.3449. Both completed500/500datasets. This identifies a credible favorable regime for our own method as well as the literature reference. It does not repair the failed n200,p500regimes or justify changing the broad logistic default. Native calibrated-lossRCAL also has a real computational limitation: its unchanged canary recovery timed out after two hours, leaving eight n800 target markers missing; partial results are preserved.

The calibrated exponential-loss audit found196 numerical recession witnesses among1,000LPs on20 datasets. Some proposed CV levels make that loss unbounded. Increasing iteration limits alone cannot repair those levels. A guarded Julia prototype rejects such levels and requires trainingKKT checks. All40 selected nuisance fits on20 datasets converged; coefficients agree withRCAL within6.2e-5and objectives within5.9e-10.38of40CVindices match. The maximum common-fold loss discrepancy is.1127, relative.00118. This is promising numerical development, not exactCVparity or inference calibration. [Checks](results/discovery-20261026-calibration-bounded20/RESULTS.md) · [Saved-fold audit](results/discovery-20261026-calibration-cv-saved-audit/checks.json).

The next substantive logistic work is a prespecified regime ladder between the successful n1000,p201control and failed n200,p500designs,with sample size,sparsity,correlation and signal varied separately. Compare the existing Julia recipes and native DoubleML on matched inputs before introducing more tuning. A checked calibrated-nuisance pipeline and the published bootstrap interval construction remain methodological alternatives. Compare Wald and bootstrap inference with the same conditional-coefficient estimand. Resolve nuisanceKKT/CV discrepancies before substituting the Julia backend. Sample-split/refined-GLM inference is another principled candidate. Bias-reduced/profile-likelihood post-selection refits remain an additional experiment; their selection step still needs validation. The native logistf installation is blocked by missing compiler headers in the R container, after the originalCRAN transport failure; no Firth coverage claim has been made.

Separate defaults by method and family remain appropriate. Gaussian has finite-design supporting evidence; logistic,QP options andPoisson remain experimental. No arbitrary SE inflation, truth-based tuning or deletion of failed regimes was used.

## QP candidate: independent confirmation completed

A distinct experimental Gaussian partialled-QP score estimator now has promising independent evidence. CVmin regularized initialization,fourX-only partialled controls,certified adaptiveQP search,fully solved target/intercept moments,and standard selected-initial-designHC3 were frozen in67236f1before500fresh datasets per each tenGaussian designs at disjoint base1300000000. Primary pooled gates passed10/10,with zero fit errors and100,000complete coefficient records for primary/secondary variants. Primary active coverage ranges94.32–98.04%and typeI1.20–5.24%. Moderate gives95.00%coverage,99.12%power,length.4267;weak signals96.16%coverage,47.04%power,length.4027. Correlated coverage98.04%,power10.24%,length1.8373 remains conservative.

Twelve primary-role Monte Carlo flags remain.40,000deeper-search warning receipts are retained;selectedprojections are certified,and certificates apply to the residualized workingQP. This is not original JM/sslasso or a productionQPdefault. It is a research prototype requiring packageAPIintegration,correctness/parity tests and expert controls before promotion. No1,000repQPcampaign has been launched. Confirmation13455517 consumed2.774CPU-hours,maximumworkerRSS1.047GiB,350seconds elapsed on64CPUs/128GiB. [Independent QP confirmation](results/discovery-20261026-qphc3-confirmation500/RESULTS.md).

## Reproduction and implementation status

All numerical work ran on Discovery compute nodes. Source snapshots, raw coefficients, diagnostics, warnings, failures, joblogs and resource receipts are retained locally under the project run directory and on Discovery. Compact reporting inputs and recipes are in this branch. Exact-job accounting is cumulative in `.slurm/runs/20261026-cd77-defaults`, not a reset budget. The package source,target-stream correction and correctness tests are pushed to draftPR86. All five hosted correctness jobs and documentation passed on3eefe3b,which includes the production stream fix. Later commits add research protocols/reports; their hosted checks are separate and may still be running. Nothing is merged automatically. The three-core native Gaussian comparison is complete. The seven extra stress-design native arms,refreshed matched-statistical campaigns and production integration of the new QPscore estimator remain outstanding. No additional large campaign was launched after the morning report.
