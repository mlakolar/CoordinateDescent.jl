# Alternating-order timing sensitivity, fixed before execution

The primary main runtime launch scheduled all baseline workers before candidate
workers. Methods/payloads inside candidate workers were rotated, but the
before/after variant comparison can have a phase confound. Preserve every
primary timing and profile; do not pool or replace those observations.

Reuse the same105 primary inputs, byte-check them, and run70 fresh processes
covering only the equicorrelated baseline/candidate procedures. Each of35
case/rep pairs runs both variants serially on the same allocated CPU, with
candidate first when(case+rep)is even and baseline first otherwise. Retain
three warmed rounds per input and three inputs per replication. Keep all
controls, covariance, innovations and seeds fixed. Repeat complete legacy
field/matrix checks and numerical certificates, without more profiling.

Report all25 payload/case comparisons and process ranges. The independent
timing units are seven process pairs; use within-input round medians, paired
medians over three inputs per process, then median across seven pairs and
10,000 process-cluster bootstrap resamples, seed106999001. Label this an
order-controlled sensitivity prompted by the primary launch design, not a
new independent dataset cohort or an unconditional causal speed estimate.
The new validated diagonal factory is distinct from the unchanged legacy
vector selector. No source/numerical/control tuning follows these outcomes.
