# Gaussian variance development study

EMPIRICAL. Discovery job13448770 completed in269seconds on32allocatedCPUs/64GiB. The study used100development datasets in each of nine Gaussian designs (900datasets). All90000coefficient records were unique and finite, with no fit errors. Role aggregation13448783 checked complete datasets and constant truth for each requested role. Full records remain in the run registry and on Discovery.

The five variance policies were model-based, HC0, HC1, HC2 and HC3. Each was paired with CV-min regularized initialization or CV-1SE post-selection OLS initialization, using the same restricted LDPE projection (four X-only unpenalized controls, SD inflation budget.5). This comparison changes neither projection nor point estimator within an initialization policy. HC3 leverages come from the selected initial design, including its intercept. Influences are centered. No truth-dependent adjustment is used.

The regularized CV-min/HC3 recipe had pooled active coverage95.6–97.4% across all nine development designs, with100/100complete datasets in every design. Its conservative intervals and power must be reviewed along with coverage. Post-selection refitting did not uniformly improve calibration. Model-based and HC1 rules still undercovered some designs. Development selection is not independent validation and HC3 after selection has no coverage theorem here.

The chosen Gaussian recipe was frozen in commit e2a30f4 before fresh confirmation. Confirmation adds a prespecified stress where error variance depends on the first tested predictor. It uses500fresh datasets in each of ten designs, and is reported separately. No production default was changed by this development result.

`STRICT_RESULTS.md` contains every policy, including adverse refit results. `per-slot.json` reviews requested roles rather than mixing physical columns under random support. Monte Carlo errors use datasets, not independent-target assumptions. Timings include shared fitting and compilation and are not matched competitor engine speed measurements.

Recreate from immutable remote gvar-v1 with gvar-development.sbatch, then slots-gvar.sbatch. Fitting source f3c60f1/source-v5/env-v5; frozen follow-up recipe is ../../frozen-gaussian-hc3.json. Source hashes, full coefficient records, resource receipts and exact-job accounting are retained under .slurm/runs/20261026-cd77-defaults.
