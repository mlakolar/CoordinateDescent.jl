# Native Gaussian core comparison, frozen before submission

The frozen JSON in the parent directory records design, method, seed, resource
forecast and limitations. These sources run from the Discovery run root's
source-v1 working directory using immutable gaussian-native-full-v1 and existing
pinned environments. No formula or native tuning default is patched.

Generate tasks from the three named cases. Preparation has eight shards per
case, maxrep1000, stagecampaign. Fitting tasks are hdi/EconML/sslasso/SIHR/hdm
for reps21–1000. Reps1–20 reuse the completed pilot, including all native warnings
and certificate failures. Do not rerun a completed scientific failure.

Prepare exact92300000 campaign data against saved own-arm SHA256 values, then
submit the fitting job after successful preparation. The run registry records
literal commands, archive hash and returned IDs. The source archive also contains
all generated task lists and confirmation-data.jl. Approximately4–6GB additional
text inputs are expected. Theta matrices are not retained beyond the first20;
per-row feasibility/gap certificates are retained for every sslasso fit.

The native pilot estimates142consumedCPU-hours for all three cases at1000/case,
with3.4hours planning walltime at64CPUs/128GiB after50%margin. This excludes
queue delay, preparation, analysis, seven additional designs and new matched
statistical arms. A five-hour allocation bounds this stage. Report queued/running
state honestly. The complete ten-design defaults-and-matched campaign is not
finished by completing this arm.

Native CV convergence warnings are kept. They can originate at unselected
path levels, so the strict rejection policy is conservative and is not itself
a native final-model failure rate. Native ssLasso's bias-expanded intervals and
its p-values need not be dual. Forty of sixty pilot fits failed at least one
strict QP certificate; native outputs remain visible. This comparison can
measure undercoverage and imperfect numerical defaults without relabeling them
validated. Existing matched fixed-penalty engine timings are a separate scope.
