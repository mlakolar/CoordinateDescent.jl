# rrpack golden data

Generated on USC CARC Discovery by `benchmark/lowrank/oracle.jl generate` and
`benchmark/lowrank/oracle.R`, run `20261025T0910Z-cd81`, R 4.5.2 and rrpack
0.1-14. Seeds 8101–8103 produce three Gaussian designs, with rank-two signal
and sparse predictor rows. `X*.csv` and `Y*.csv` are shared inputs; the other
CSVs are independent R outputs, read directly by `test/lowrank_golden.jl`.
The test suite needs no R installation.

RRR compares coefficients on identifiable designs and fitted values for the
wide design: rrpack returns basic QR coefficients, while this implementation
returns the minimum-norm coefficients. `df-*.csv` contains exact Stein degrees
of freedom. SRRR uses three penalties with unit/adaptive weights and an
explicit common RRR initialization; rrpack's level is `n*λ`.

RSSVD uses ten penalty candidates and actual BIC selection. rrpack 0.1-14
swaps BIC/BICP selection labels inside `rssvd.eea`; the oracle script requests
its BICP label and verifies the selected coefficients against the BIC column
of the independent R path. The Julia methods use the documented formulas.
The component-level diagnostic agreed within 3.1e-9 before model selection.
