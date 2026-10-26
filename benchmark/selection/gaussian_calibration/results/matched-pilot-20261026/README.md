# Matched-accuracy pilot and default checks, excluded from evaluation

All numerical work ran under Discovery Slurm. These are the declared rep0
pilot inputs, reused for diagnosed harness/stopping controls, not extra
independent evaluation datasets.

- Statistical32-input pilot:352/352 original-design/intercept certificates
  pass at1e-7;192 common-weight backend pairs have no selection disagreement;
  maximum coefficient/objective relative differences1.403e-7/4.576e-15;
  native automatic/explicit Gaussian fits bitwise equal. No API errors.
- Native stopping probe13455366:36 paths on12 existing inputs,720/720
  certificates pass at unchanged1e-7, all backend convergence flags true at
  tol1e-8/max100000. Cold probe timings include compilation and are not ranked.
- Corrected runtime/default pilot13455427:13,500/13,500 timed-point certificates
  pass on12 inputs; maximum residual7.500e-8. Complete public/prepared result
  fields bitwise equal and72 allocation differences positive. All288 CPU
  profiles nonempty;288 allocation reports retained remotely. Default probes
  use the unchanged public tol1e-7/max10000; study native tol1e-8/max100000,
  SLOPE.jl/R tol1e-10 and their original iteration caps.

Adverse evidence is retained. The original1e-9 native runtime paths can hit
100000 iterations, warn and consume seconds/GiB cumulative allocation; their
v1/v3 timing rows are preserved and excluded from the main matched setting.
Unbuffered allocation-stack writes caused the first pilot to be stopped;
stage recovery preserves126/81/102 original timed-result records in native
cases1/2/3, reuses complete workers/profiles and adds only missing stages.
Native1e-8/default outputs then passed, but original external1e-8 paths failed
48 certificates (three rounds count separately):6 R dense points and42
SLOPE.jl correlated points. Max residuals1.154e-7/1.120e-6. Tightening those
external controls before the main cohort resolves every pilot accuracy failure.
All old results and checker failures remain available. No residual threshold,
sequence, signal or cutoff was weakened or tuned to main outcomes.

The first statistical checker failure concerned R's rational normal quantile
approximation, not optimizer accuracy; the separate statistical-pilot directory
preserves its diagnosis and saved-fit-only recovery.

`pilot-report-v5.tar.gz` contains full per-point certificates, comparisons,
complete timing/record/witness CSVs, source/thread metadata and worker resource
receipts, statistical uncertainty tables, stopping-probe arrays/records and old
failed gate receipts. All raw shared X/y and timed fitted arrays, profiles and
frozen source/environments remain under Discovery issue107. Profiles and
certificates are outside timers; complete returned results are timed. Peak RSS
includes imports/JIT/profiling. This pilot does not establish a statistical
improvement or default change. The main cohorts are separate fresh rep1-7 data.

Archive SHA256: `6c29d7cfcbc8102aac4bb8fec109196e57127cb822c705550b729946f918d62b`.
