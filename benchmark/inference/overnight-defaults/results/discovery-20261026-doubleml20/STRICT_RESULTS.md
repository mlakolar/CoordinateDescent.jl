# Complete-dataset inference summaries

EMPIRICAL. Every requested target must be finite, unique and converged before a dataset contributes. Missing/failed fits count against reliability. Per-slot results use fixed requested target roles; physical columns vary in random-support designs. The older per-target-by-column file mixes roles in those designs and is superseded. All Monte Carlo errors use independent datasets.

|Case|Method|Class|Valid/planned|Coverage ± MCSE|Rejection ± MCSE|Length|
|---|---|---|---|---|---|---|
|belloni-0|doubleml_sparse_instrument|null|16/20|0.8438 ± 0.05983919423477113|0.1562 ± 0.05983919423477113|0.5183|
|belloni-0|doubleml_sparse_nuisance_space|null|18/20|0.7500 ± 0.0728728860162672|0.2500 ± 0.0728728860162672|0.4743|
|belloni-0p2|doubleml_sparse_instrument|active|18/20|0.8889 ± 0.0762215933966706|0.7778 ± 0.10083169033033669|0.4617|
|belloni-0p2|doubleml_sparse_instrument|null|18/20|0.9444 ± 0.05555555555555556|0.0556 ± 0.05555555555555556|0.6272|
|belloni-0p2|doubleml_sparse_nuisance_space|active|18/20|0.6667 ± 0.1143323900950059|1.0000 ± 0.0|0.4552|
|belloni-0p2|doubleml_sparse_nuisance_space|null|18/20|0.8889 ± 0.0762215933966706|0.1111 ± 0.0762215933966706|0.5837|
|binomial-benign|doubleml_sparse_instrument|active|20/20|0.9500 ± 0.049999999999999996|1.0000 ± 0.0|0.4059|
|binomial-benign|doubleml_sparse_instrument|null|20/20|0.8500 ± 0.08191780219091253|0.1500 ± 0.08191780219091253|0.3740|
|binomial-benign|doubleml_sparse_nuisance_space|active|20/20|0.9500 ± 0.049999999999999996|1.0000 ± 0.0|0.4432|
|binomial-benign|doubleml_sparse_nuisance_space|null|20/20|0.8500 ± 0.08191780219091253|0.1500 ± 0.08191780219091253|0.3704|
|binomial-correlated|doubleml_sparse_instrument|active|20/20|0.9500 ± 0.049999999999999996|0.2500 ± 0.09933992677987827|1.6931|
|binomial-correlated|doubleml_sparse_instrument|null|20/20|1.0000 ± 0.0|0.0000 ± 0.0|1.7853|
|binomial-correlated|doubleml_sparse_nuisance_space|active|20/20|0.8500 ± 0.08191780219091253|0.3000 ± 0.10513149660756936|1.4564|
|binomial-correlated|doubleml_sparse_nuisance_space|null|20/20|0.9000 ± 0.06882472016116853|0.1000 ± 0.06882472016116853|1.4947|
|binomial-independent|doubleml_sparse_instrument|active|20/20|0.8500 ± 0.08191780219091253|1.0000 ± 0.0|0.6121|
|binomial-independent|doubleml_sparse_instrument|null|20/20|0.9500 ± 0.049999999999999996|0.0500 ± 0.049999999999999996|0.5523|
|binomial-independent|doubleml_sparse_nuisance_space|active|20/20|0.9000 ± 0.06882472016116853|1.0000 ± 0.0|0.6487|
|binomial-independent|doubleml_sparse_nuisance_space|null|20/20|0.9000 ± 0.06882472016116853|0.1000 ± 0.06882472016116853|0.5705|
|binomial-largen|doubleml_sparse_instrument|active|12/20|0.5833 ± 0.1486470975026408|1.0000 ± 0.0|0.3237|
|binomial-largen|doubleml_sparse_instrument|null|12/20|0.8333 ± 0.11236664374387369|0.1667 ± 0.11236664374387369|0.2998|
|binomial-largen|doubleml_sparse_nuisance_space|active|12/20|0.8333 ± 0.11236664374387369|1.0000 ± 0.0|0.3491|
|binomial-largen|doubleml_sparse_nuisance_space|null|12/20|0.7500 ± 0.1305582419667734|0.2500 ± 0.1305582419667734|0.3032|
|binomial-moderate|doubleml_sparse_instrument|active|20/20|0.5500 ± 0.11413288653790231|0.8500 ± 0.08191780219091253|0.6074|
|binomial-moderate|doubleml_sparse_instrument|null|20/20|1.0000 ± 0.0|0.0000 ± 0.0|0.5834|
|binomial-moderate|doubleml_sparse_nuisance_space|active|20/20|0.4500 ± 0.11413288653790231|0.7500 ± 0.09933992677987827|0.6189|
|binomial-moderate|doubleml_sparse_nuisance_space|null|20/20|0.9500 ± 0.049999999999999996|0.0500 ± 0.049999999999999996|0.5472|
|binomial-random|doubleml_sparse_instrument|active|20/20|0.7500 ± 0.09933992677987827|1.0000 ± 0.0|0.6191|
|binomial-random|doubleml_sparse_instrument|null|20/20|0.8500 ± 0.08191780219091253|0.1500 ± 0.08191780219091253|0.5759|
|binomial-random|doubleml_sparse_nuisance_space|active|20/20|0.7000 ± 0.10513149660756936|1.0000 ± 0.0|0.6173|
|binomial-random|doubleml_sparse_nuisance_space|null|20/20|0.8500 ± 0.08191780219091253|0.1500 ± 0.08191780219091253|0.5409|
|binomial-sihr|doubleml_sparse_nuisance_space|active|2/20|1.0000 ± 0.0|1.0000 ± 0.0|0.7851|
|binomial-sihr|doubleml_sparse_nuisance_space|null|2/20|1.0000 ± 0.0|0.0000 ± 0.0|0.6642|
|binomial-weak|doubleml_sparse_instrument|active|20/20|1.0000 ± 0.0|0.5000 ± 0.11470786693528089|0.6015|
|binomial-weak|doubleml_sparse_instrument|null|20/20|1.0000 ± 0.0|0.0000 ± 0.0|0.5746|
|binomial-weak|doubleml_sparse_nuisance_space|active|20/20|0.9000 ± 0.06882472016116853|0.5500 ± 0.11413288653790231|0.5871|
|binomial-weak|doubleml_sparse_nuisance_space|null|20/20|0.9500 ± 0.049999999999999996|0.0500 ± 0.049999999999999996|0.5300|
|liu-published|doubleml_sparse_instrument|active|20/20|1.0000 ± 0.0|1.0000 ± 0.0|0.3170|
|liu-published|doubleml_sparse_instrument|null|20/20|0.9000 ± 0.06882472016116853|0.1000 ± 0.06882472016116853|0.2747|
|liu-published|doubleml_sparse_nuisance_space|active|20/20|1.0000 ± 0.0|1.0000 ± 0.0|0.3501|
|liu-published|doubleml_sparse_nuisance_space|null|20/20|0.9000 ± 0.06882472016116853|0.1000 ± 0.06882472016116853|0.2867|
|liu-small|doubleml_sparse_instrument|active|8/20|1.0000 ± 0.0|0.8750 ± 0.125|0.7062|
|liu-small|doubleml_sparse_instrument|null|8/20|1.0000 ± 0.0|0.0000 ± 0.0|0.6039|
|liu-small|doubleml_sparse_nuisance_space|active|11/20|0.9091 ± 0.09090909090909091|0.8182 ± 0.12196734422726126|0.8056|
|liu-small|doubleml_sparse_nuisance_space|null|11/20|1.0000 ± 0.0|0.0000 ± 0.0|0.6497|
