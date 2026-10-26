# Gaussian competitor pilot: native defaults and production-policy timing

EMPIRICAL. Twenty matched response datasets in each of Gaussian moderate/correlated/benign designs;60datasets,420fit tasks. All input hashes match the frozen1000-per-design own arm. This is a native/default pipeline comparison. The earlier fixed-penalty engine comparison remains a separate matched statistical exercise. hdi/EconML/sslasso infer all slopes; Julia, SIHR and hdm request ten targets.

|Design|Method|Finite intervals /20|Strict checks /20|Active coverage ± MCSE (%)|Null type I ± MCSE (%)|Power (%)|Active length|Mean fit (s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|gaussian-moderate|ours_warm|20|20|99.00 ± 1.00|2.00 ± 1.38|99.00|0.5165|5.681|
|gaussian-moderate|hdi|20|20|73.00 ± 4.65|2.00 ± 1.38|100.00|0.3152|204.595|
|gaussian-moderate|EconML|20|0|67.00 ± 5.29|1.00 ± 1.00|100.00|0.2776|60.249|
|gaussian-moderate|sslasso|20|0|73.00 ± 4.65|1.00 ± 1.00|100.00|0.3206|46.214|
|gaussian-moderate|SIHR|20|20|65.00 ± 5.21|1.00 ± 1.00|100.00|0.2832|22.403|
|gaussian-moderate|hdm|20|20|91.00 ± 3.07|2.00 ± 1.38|99.00|0.3339|3.541|
|gaussian-correlated|ours_warm|20|20|97.00 ± 2.19|1.00 ± 1.00|14.00|1.9971|3.356|
|gaussian-correlated|hdi|20|20|81.00 ± 4.22|5.00 ± 1.99|30.00|0.8388|57.035|
|gaussian-correlated|EconML|20|12|72.00 ± 5.69|6.00 ± 2.10|31.00|0.7545|24.109|
|gaussian-correlated|sslasso|20|0|7.00 ± 2.19|17.00 ± 5.48|29.00|0.4607|17.507|
|gaussian-correlated|SIHR|20|20|78.00 ± 5.41|5.00 ± 1.99|17.00|0.8353|13.028|
|gaussian-correlated|hdm|20|20|98.00 ± 2.00|4.00 ± 1.84|40.00|1.2445|9.717|
|gaussian-benign|ours_warm|20|20|98.00 ± 1.38|1.00 ± 1.00|100.00|0.2175|0.738|
|gaussian-benign|hdi|20|20|91.00 ± 3.07|1.00 ± 1.00|100.00|0.1903|8.591|
|gaussian-benign|EconML|20|20|96.00 ± 1.84|1.00 ± 1.00|100.00|0.1819|0.952|
|gaussian-benign|sslasso|20|20|98.00 ± 1.38|0.00 ± 0.00|100.00|0.2068|0.143|
|gaussian-benign|SIHR|20|20|97.00 ± 1.64|0.00 ± 0.00|100.00|0.1920|17.782|
|gaussian-benign|hdm|20|20|96.00 ± 1.84|1.00 ± 1.00|100.00|0.1895|0.498|

Coverage is conditional on finite returned native intervals, including convergence-warned or uncertified native results. Strict checks show availability after rejecting exposed convergence warnings and requiring the declared sslasso QP feasibility/gap tolerances. A good conditional coverage number does not repair failed availability. A convergence warning can originate at an unselected CV path level; strict warning rejection is conservative and is not itself the native final-model failure rate. Dataset-level MCSE pools target roles within each dataset. Twenty datasets give imprecise calibration estimates. These data were not used to tune the frozen Julia default.

Native sslasso uses its adaptive per-row mu search and bias-allowance intervals. Its native p-values need not be dual to those expanded intervals. Known author-source bugs were not patched here. The original certificate recorder lost M/G through an R environment-binding mistake after writing the native intervals; the observer-only recovery saves theta and corrects those bindings. All60 recovered point/interval outputs match the original saved outputs. See parity-and-certificates.json and qp-summary.json. A failed certificate is reported, not repaired by relabeling a native row optimal.

The prior matched hdi/EconML comparison used the same initial/nodewise penalties and500slopes; EconML subtracts selected slopes from n in its noise denominator, while Julia/hdi also subtract an intercept degree of freedom. The certified sslasso component comparison used the documented diagonal correction and fixed shared penalty. Those results are not a timing comparison of the new automatic HC3 policy.

Julia cold-fit timers include fitting compilation after package import. The complete cold process resource includes imports and file loading. Warm timers follow a small independent warmup; the warm process also includes a separate CV/initializer measurement, so its complete process time is not a pure single-fit startup measurement. Initial-CV component times are measured separately, not obtained by subtraction. Actual kernel times, process times and per-worker RSS are retained. Peak RSS is not simultaneous total job memory.

Full-study resource estimates below cover only these three core designs and named tasks, with a50%planning margin. Expanded competitor1000-replication arms have not been launched at report creation.

```json
{
  "three_core_native_methods": {
    "datasets_per_case": 1000,
    "cases": 3,
    "tasks_per_dataset": 5,
    "consumed_cpu_hours": 141.9637132222222,
    "summed_process_hours": 145.2977298016065,
    "ideal_process_wall_hours64cpu": 2.2702770281501015,
    "planning_process_wall_hours64cpu_50pct_margin": 3.4054155422251524,
    "cpus": 64,
    "memory_gib": 128,
    "max_observed_worker_rss_gib": 0.7496414184570312,
    "note": "Conditional extrapolation from20datasets/case, excludes queue delay, future input generation, analysis and extra designs. Native tasks infer different numbers of slopes. Includes each measured process warmup. Expanded ten-design and matched-statistical campaigns are not forecast here."
  },
  "three_core_exact_seven_timing_tasks": {
    "datasets_per_case": 1000,
    "cases": 3,
    "tasks_per_dataset": 7,
    "consumed_cpu_hours": 193.17519991666668,
    "summed_process_hours": 197.25300595925526,
    "ideal_process_wall_hours64cpu": 3.0820782181133635,
    "planning_process_wall_hours64cpu_50pct_margin": 4.623117327170045,
    "cpus": 64,
    "memory_gib": 128,
    "max_observed_worker_rss_gib": 0.9223251342773438,
    "note": "Conditional extrapolation from20datasets/case, excludes queue delay, future input generation, analysis and extra designs. Native tasks infer different numbers of slopes. Includes each measured process warmup. Expanded ten-design and matched-statistical campaigns are not forecast here."
  }
}
```

Recreation: gaussian-native-v1 canary, v2 required-library/observer diagnostics, v3 observer-only certificate recovery; same92300000 campaign inputs checked against originalFloat64-byte hashes. analyze-gaussian-native-v3.py pilot20-v3 20 then this script, on a Discovery compute node. All raw warnings, failed attempts, native output tables, theta/certificate records and worker receipts are retained.
