# Interpretation and next direction

EMPIRICAL. The projection-grid endpoint is not the main explanation for the remaining logistic undercoverage. The logistic one-step correction has a material nonlinear remainder. These findings motivate joint work on the initializer and nonlinear correction, with separate policies by problem. No production defaults changed and no full competitor campaign was launched.

## Findings

The current experimental restricted-LDPE recipe produced active coverage 92.5% +/-0.9pp (moderate),97.2% +/-0.5pp (correlated),91.8% +/-1.0pp (benign),94.2% +/-0.8pp (random support). These are200 fresh development datasets per case; uncertainties are dataset-level Monte Carlo standard errors, not confidence interval half-widths. This development study does not replace the earlier independently frozen holdout. The new seed results are not evidence of a policy improvement: the recipe is unchanged.

| Logistic design | Active coverage | Null type I | Power | Mean active interval length |
|---|---:|---:|---:|---:|
| Moderate |92.5% +/-0.9pp|3.0% +/-0.5pp|61.8% +/-1.7pp|0.9518|
| Correlated |97.2% +/-0.5pp|5.1% +/-0.8pp|10.7% +/-1.1pp|3.5071|
| Benign |91.8% +/-1.0pp|4.5% +/-0.7pp|100%|0.4730|
| Random support |94.2% +/-0.8pp|2.3% +/-0.5pp|60.0% +/-1.4pp|1.0141|

The moderate logistic signed standardized error decomposes into stochastic +0.019, linear -0.117, nonlinear -0.535. The random-support counterparts are +0.069,-0.075,-0.539. The nonlinear component is larger in magnitude than the linear component in these designs. This is an exact algebraic decomposition of the implemented estimator, followed by empirical averaging. It is not a theorem about a new estimator or an assertion that the components are independent.

In the benign logistic case, linear error is almost zero, but the nonlinear component is -0.466 and the stochastic component has mean +0.343 with estimated response-dependent scores. These partially cancel. Active coverage ranges from 89.0% to93.5% across the five fixed active targets. Moderate coverage ranges 91.0% to95.0%. This is not merely a pooled-target artifact. Each fixed target has200 replications, so individual-target MC errors are larger than pooled dataset-level errors; see targets.json. Random-support results use active/null slots, not inconsistent pooling of coordinate IDs.

Oracle controls must be interpreted together. Using true coefficients/intercept with true Fisher weights gave 94.7--95.8% active coverage across the four logistic designs. Using true coefficients with estimated weights gave 93.2--98.0%. Replacing only the weights, while retaining the regularized initial estimate, made moderate/random coverage 60.3%/73.0%; their signed nonlinear components became+1.751/+1.477. The oracle weighting intervention is not an implementable repair and does not isolate a universally beneficial change. A more accurate Hessian alone is not supported as a solution by these results. The estimated scores depend on the response, so the stochastic term need not have mean zero unconditionally.

Densifying the grid barely changed estimates and SE. Extending the lower floor from .001 to 1e-7 removed most high-dimensional endpoint selections, but did not improve logistic coverage: moderate92.5% to 92.3%, correlated97.2% to 96.9%, random94.2% to 94.2%. Mean absolute SE changes were about 3.35%,1.93%,2.58%, respectively. Paired coverage changes were-0.20+/-0.32pp,-0.30+/-0.22pp,0.00+/-0.20pp. In benign logistic, the extended path and exact inverse both gave 91.7%, almost unchanged from91.8%; mean absolute estimate changes were only 0.00146 of a base SE. A boundary flag is therefore a useful stability warning, not itself evidence that extension will repair coverage. Tiny penalties can still select a different stage-1 reference and are not a uniform accuracy improvement.

Gaussian known-noise controls behaved differently: moderate96.2%+/-0.9pp and benign95.6%+/-0.9pp, with negligible nonlinear remainder. The deeper grid changed moderate coverage by +0.20+/-0.35pp and benign by 0pp. These controls isolate projection behavior and do not revalidate estimated-noise defaults or extend to all Gaussian designs.

## Recommended next experiment (not launched)

Keep the Gaussian restricted-LDPE rule as the leading experimental Gaussian candidate. For logistic regression, compare a small set of implementable joint changes to initialization and correction. Prioritize a carefully specified link-specific weighted correction and a sample-split/refined-debiasing alternative. Test them first on development data, including weak signals and varying p/n. Do not assume repeated Newton steps or sample splitting guarantees coverage. Freeze the exact formula, nuisance treatment, regularization and variance rule before new independent holdout seeds. Existing oracle controls cannot be used as production tuning rules, and the present development data cannot become an untouched holdout.

Treat QP separately; this diagnostic study did not address its universal-mu calibration failure. Poisson and simultaneous inference are also outside this study. Different defaults by family/projection are appropriate possibilities. No default promotion is justified by convergence alone. A full comparison should wait until the selected family-specific policies pass a prespecified coverage/type-I gate and their interval lengths and power are acceptable. Fixed-penalty speed advantages from the earlier competitor benchmark remain separate from these diagnostic costs.

## Execution and reproducibility

Discovery main job13445380 completed in 58m54s on 8CPUs/16GiB; canary13445369 took 89s; saved-output aggregation13446265 took 7s on 1 CPU/4GiB. All 1000 completion markers and 154000 unique target records were present, with zero fit-error files and all projection/initializer convergence flags true. Maximum decomposition identity error 1.9984e-15; exact benign linear -bias cancellation 1.1102e-15; Gaussian nonlinear maximum 1.0686e-16. All 48 worker receipts report exit 0; maximum whole-worker RSS 1126.5MiB. Worker CPU time was 27898s and summed wall time 28072.7s. This is a diagnostic workload with many oracle/grid interventions and compilation, not a production fit-speed benchmark or a new full-campaign cost estimate.

The scientific runner is study.jl at commit 3faa3f22011f49d7f08e2bed2aec2bf22fb7d391 (SHA256 24040c404920d655fc1b541279c8184ba1b58d0c5191e6e7fb4652a6a0fdde7c). Fitting source is the prior immutable source-v5/env-v5, package commit f3c60f1012f940c59fba79985531bc80ddc45579; fitting code was not changed. Main namespace 91700000 and canary 91800000 are disjoint from the earlier 91500000/91600000 namespaces. Six datasets in canary are additional to the 1000 main datasets. Corresponding Gaussian/logistic cases share covariates; independence is within a case, not between response families.

For recreation, use the scripts and full policy in ../../README.md with main-tasks.tsv and the Slurm launchers stored alongside this report. Run canary.sbatch, then main.sbatch afterok the canary, then detail.sbatch afterok main. Launchers use the preserved absolute source/environment and run roots; adapt only those paths for a new recreation. analyze.py runs after the main workers on the compute allocation; detail-v1.py (the checked-in detail.py) reads saved target records. Python aggregation and all Julia fitting were performed on Discovery compute nodes. The laptop performed file/metadata/report preparation only.

Raw 154000-row coefficients (approximately 60MiB), original generated report, logs, receipts, launchers and exact accounting are retained locally under /Users/mladen.kolar/projects/juliaPkg/.slurm/runs/20261025-cd77-decomposition and remotely under /home1/mkolar/juliaPkg/runs/20261025-cd77-decomposition. Compact summaries, per-target records and checks accompany this report. No refits or recoveries were needed. Raw score vectors were not saved; stability is measured through estimates and SE, not vector norms. The prior full-campaign warm-cost projection remains conditional and was not recalculated from this diagnostic workload.

---

# Discovery error decomposition and projection-path stability

EMPIRICAL. Frozen experimental restricted LDPE m4/kappa.5; production defaults unchanged. Main: 200 datasets per four logistic cases and 100 per two Gaussian controls. Oracle interventions are diagnostics only. All uncertainties below are dataset-level Monte Carlo standard errors.

## Logistic active coverage and signed error decomposition

| Case | Weights | Initial | Coverage +/- MCSE | Stochastic z | Linear z | Nonlinear z |
|---|---|---|---|---|---|---|
| binomial-benign | estimated | estimated | 91.8% +/- 1.0pp | 0.343 | -0.000 | -0.466 |
| binomial-benign | estimated | oracle | 93.2% +/- 0.9pp | 0.343 | 0.000 | 0.000 |
| binomial-benign | oracle | estimated | 94.0% +/- 0.9pp | 0.008 | -0.001 | 0.650 |
| binomial-benign | oracle | oracle | 94.7% +/- 0.8pp | 0.008 | 0.000 | 0.000 |
| binomial-correlated | estimated | estimated | 97.2% +/- 0.5pp | 0.028 | -0.010 | -0.076 |
| binomial-correlated | estimated | oracle | 96.6% +/- 0.6pp | 0.028 | 0.000 | 0.000 |
| binomial-correlated | oracle | estimated | 93.8% +/- 0.8pp | 0.039 | -0.020 | 0.264 |
| binomial-correlated | oracle | oracle | 95.2% +/- 0.7pp | 0.039 | 0.000 | 0.000 |
| binomial-moderate | estimated | estimated | 92.5% +/- 0.9pp | 0.019 | -0.117 | -0.535 |
| binomial-moderate | estimated | oracle | 97.3% +/- 0.6pp | 0.019 | 0.000 | 0.000 |
| binomial-moderate | oracle | estimated | 60.3% +/- 2.0pp | -0.003 | -0.105 | 1.751 |
| binomial-moderate | oracle | oracle | 95.8% +/- 0.7pp | -0.003 | 0.000 | 0.000 |
| binomial-random | estimated | estimated | 94.2% +/- 0.8pp | 0.069 | -0.075 | -0.539 |
| binomial-random | estimated | oracle | 98.0% +/- 0.5pp | 0.069 | 0.000 | 0.000 |
| binomial-random | oracle | estimated | 73.0% +/- 1.9pp | -0.018 | -0.134 | 1.477 |
| binomial-random | oracle | oracle | 94.7% +/- 0.8pp | -0.018 | 0.000 | 0.000 |

## Grid effects with estimated initialization and weights

| Case | Grid | Active coverage +/- MCSE | Null type I | Active interval length | Lower endpoint |
|---|---|---|---|---|---|
| binomial-benign | base | 91.8% +/- 1.0pp | 4.5% | 0.4730 | 100.0% |
| binomial-benign | deep | 91.7% +/- 1.0pp | 4.5% | 0.4733 | 100.0% |
| binomial-benign | deeper | 91.7% +/- 1.0pp | 4.5% | 0.4733 | 100.0% |
| binomial-benign | dense | 91.8% +/- 1.0pp | 4.5% | 0.4730 | 100.0% |
| binomial-benign | exact | 91.7% +/- 1.0pp | 4.5% | 0.4733 | 0.0% |
| binomial-correlated | base | 97.2% +/- 0.5pp | 5.1% | 3.5071 | 46.4% |
| binomial-correlated | deep | 96.9% +/- 0.6pp | 4.9% | 3.5240 | 38.3% |
| binomial-correlated | deeper | 96.9% +/- 0.6pp | 4.8% | 3.5688 | 0.1% |
| binomial-correlated | dense | 97.1% +/- 0.5pp | 5.2% | 3.5106 | 46.4% |
| binomial-moderate | base | 92.5% +/- 0.9pp | 3.0% | 0.9518 | 77.6% |
| binomial-moderate | deep | 92.5% +/- 0.9pp | 3.1% | 0.9564 | 72.8% |
| binomial-moderate | deeper | 92.3% +/- 0.9pp | 3.1% | 0.9824 | 0.5% |
| binomial-moderate | dense | 92.5% +/- 0.9pp | 3.0% | 0.9521 | 77.6% |
| binomial-random | base | 94.2% +/- 0.8pp | 2.3% | 1.0141 | 65.7% |
| binomial-random | deep | 94.0% +/- 0.8pp | 2.4% | 1.0181 | 60.4% |
| binomial-random | deeper | 94.2% +/- 0.7pp | 2.1% | 1.0387 | 0.3% |
| binomial-random | dense | 94.2% +/- 0.8pp | 2.2% | 1.0145 | 65.7% |
| gaussian-benign | base | 95.6% +/- 0.9pp | 5.0% | 0.2122 | 100.0% |
| gaussian-benign | deep | 95.6% +/- 0.9pp | 5.0% | 0.2123 | 100.0% |
| gaussian-benign | deeper | 95.6% +/- 0.9pp | 5.0% | 0.2123 | 100.0% |
| gaussian-benign | dense | 95.6% +/- 0.9pp | 5.0% | 0.2122 | 100.0% |
| gaussian-benign | exact | 95.6% +/- 0.9pp | 5.0% | 0.2123 | 0.0% |
| gaussian-moderate | base | 96.2% +/- 0.9pp | 6.0% | 0.4572 | 81.8% |
| gaussian-moderate | deep | 96.2% +/- 0.9pp | 6.2% | 0.4595 | 76.0% |
| gaussian-moderate | deeper | 96.4% +/- 0.9pp | 6.2% | 0.4734 | 1.0% |
| gaussian-moderate | dense | 96.2% +/- 0.9pp | 6.0% | 0.4573 | 81.8% |

## Paired grid changes

| Case | Grid | Absolute estimate change / base SE | Absolute fractional SE change | Coverage change +/- MCSE |
|---|---|---|---|---|
| binomial-benign | deep | 0.00145 | 0.00055 | -0.10 +/- 0.10pp |
| binomial-benign | deeper | 0.00146 | 0.00056 | -0.10 +/- 0.10pp |
| binomial-benign | dense | 0.00000 | 0.00000 | 0.00 +/- 0.00pp |
| binomial-benign | exact | 0.00146 | 0.00056 | -0.10 +/- 0.10pp |
| binomial-correlated | deep | 0.02424 | 0.00581 | -0.30 +/- 0.17pp |
| binomial-correlated | deeper | 0.05435 | 0.01928 | -0.30 +/- 0.22pp |
| binomial-correlated | dense | 0.00408 | 0.00149 | -0.10 +/- 0.10pp |
| binomial-moderate | deep | 0.02498 | 0.00503 | 0.00 +/- 0.00pp |
| binomial-moderate | deeper | 0.08218 | 0.03346 | -0.20 +/- 0.32pp |
| binomial-moderate | dense | 0.00136 | 0.00034 | 0.00 +/- 0.00pp |
| binomial-random | deep | 0.02010 | 0.00451 | -0.20 +/- 0.14pp |
| binomial-random | deeper | 0.05853 | 0.02577 | 0.00 +/- 0.20pp |
| binomial-random | dense | 0.00164 | 0.00064 | 0.00 +/- 0.00pp |
| gaussian-benign | deep | 0.00122 | 0.00053 | 0.00 +/- 0.00pp |
| gaussian-benign | deeper | 0.00123 | 0.00054 | 0.00 +/- 0.00pp |
| gaussian-benign | dense | 0.00000 | 0.00000 | 0.00 +/- 0.00pp |
| gaussian-benign | exact | 0.00123 | 0.00054 | 0.00 +/- 0.00pp |
| gaussian-moderate | deep | 0.02691 | 0.00526 | 0.00 +/- 0.00pp |
| gaussian-moderate | deeper | 0.08911 | 0.03650 | 0.20 +/- 0.35pp |
| gaussian-moderate | dense | 0.00105 | 0.00027 | 0.00 +/- 0.00pp |

## Checks and limitations

```json
{
  "finite_estimates_and_components": true,
  "gaussian_nonlinear_max": 1.0685937497186953e-16,
  "oracle_initial_linear_max": 0.0,
  "oracle_initial_nonlinear_max": 0.0,
  "exact_projection_linear_max": 1.1102230246251565e-15,
  "max_noise_budget_ratio": 1.4999998760323736,
  "worker_resource_records": 48,
  "worker_exits_zero": true
}
```

Signed components are aligned with the true coefficient sign for active targets. Component quantiles/SD and per-target summaries are in detail.json and targets.json; targets within datasets are dependent. Random-support targets are reported by selection slot rather than pooling unrelated coordinate IDs. The true-variance quadratic form is diagnostic when scores depend on the response. Gaussian controls use known sigma1. Exact inverse projection is restricted to benign p<n. No Poisson, competitor, or simultaneous-coverage validation is included. Grids change discrete stage-1 reference as well as floor and density. Raw score vectors were not retained; paired stability metrics describe estimates and SE, not a norm of score-vector changes. Timing includes compilation and reuse and is not an end-to-end competitor benchmark.

Worker processes: 48; maximum whole-worker RSS 1126.5 MiB. Process seconds sum 28072.7; CPU seconds sum 27898.0. These include many diagnostic interventions per dataset.
