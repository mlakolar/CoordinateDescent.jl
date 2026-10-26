# Official DoubleMLLPLR sparse-learner pilot:20datasets per design

EMPIRICAL. Native6884b5f1 framework with the declared sparse eight-penalty learners, nested five-by-five cross-fitting, both native scores, and two requested roles/dataset. This is not an author-bootstrap replication or a universal package-default comparison: DoubleML requires explicit nuisance learners.

There are240datasets,480dataset/score workers and960planned target fits. The run returned839unique finite target records and121target errors, with62,925retained warning entries. All worker completion markers are present. Native score/sandwich SE parity is2.22e-16. All24planned case/method groups, including zero-success groups if any, are enumerated in all-planned-outcomes.json. Strict summaries require both requested target roles in a dataset.

The moderate design has active coverage45%(nuisance_space) and55%(instrument), each on20/20complete datasets; MCSE is about11.4percentage points. The correlated instrument score gives95%active coverage and0%null rejection on20/20, with active length1.693 and power25%; its active-coverage MCSE is5percentage points. This is a favorable pilot result, not broad independent calibration. The larger-n design has only12/20complete datasets per score and active coverage58.3%/83.3%. The paper-style Liu n1000,p201 control gives100%active coverage in20/20for both scores, with90%null coverage. A20/20coverage result still has a Wilson95%Monte Carlo lower bound of83.9%, so zero empirical MCSE does not mean certainty.

See STRICT_RESULTS.md for all designs, roles, power and interval length, and role-monte-carlo-bounds.json for per-role boundary-aware uncertainty. MC intervals here are conditional on successful complete datasets; availability is reported separately. A convergence warning may originate from a CV path level; the declared strict warning policy is conservative, not a failure theorem for DoubleML or all learners. Failed scientific fits were not silently retried.

```json
{
  "worker_receipts": 480,
  "expected_workers": 480,
  "cpu_seconds": 185014.228189,
  "process_seconds_sum": 187081.65245147655,
  "max_worker_rss_gib": 0.251495361328125,
  "note": "Single dataset/score per fresh process, two requested targets. Actual child CPU;per-worker peakRSS,not aggregate memory. Allocation128CPUs/128GiB for3025s. Full results include121targeterrors;processsuccessisnot calibration."
}
```

No logistic default or full logistic campaign is eligible from this pilot. The original broad-grid configuration and its failures remain separate. These results support testing favorable published regimes and improving nuisance estimation/finite-sample bias control; they do not justify truth-based variance inflation.
