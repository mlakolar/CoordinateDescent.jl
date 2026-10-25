Measured production: candidate-v2.tar / candidate2/, lowrank.jl SHA256
99a97cc5151408181ac880cc6bc3cd271b24d7b779deb36087ff77cf93bcd5fc.
Baseline master295fbf5420d6384a1ca261023367ee86d0a40b8c, lowrank.jl SHA256
b611a82eb5228e1f33ca2d1dce66aa682187764025519e648a759fa2461de448.
Source/harness frozen before new generation at protocol36275f8.

Primary harness64a4146abdac8a9b237717d7455687bf6de03c22ed58e114c971a72a4c1cd3ae;
R-only sensitivity harness e943734d3b2679350a5e55814beeb6a94476366ebcdfd7f68ff1327b728de22c.
Primary reporting delivery SHA256
7a77d8f14fd5c45bc6dc2fc1e69ce9c00b9ef3d1deea5778a599ba917fa41b32.
The compressed delivery contains every reporting worker's profiles, metadata,
returned scalar diagnostics, hashes and logs; raw coefficient/factor arrays are
excluded and remain in the owning Discovery run. The R follow-up delivery
includes only its240 new R worker directories, rather than dereferencing720
reused Julia directories. Top-level follow-up tables explicitly reuse Julia
results; they are not additional independent inputs or new Julia processes.

Julia1.13.2; targeted correctness also1.10.12. EPYC7513; one Julia/BLAS/OpenMP
thread. R4.5.2, rrpack0.1-14, explicit shared weights, lambda_R=n*lambda_Julia.
Dependencies unchanged before/after: PBb6c949f3dd26cad09252e1c2c9e49810eb69403e,
PS531ff55fcf8a5ce2c6997c144d4f57e85cf79f3e,
KS1feff286d0020764d2c1b03d733ab316cf948ead. Exact installed R bodies/session/
BLAS details and native per-worker source/CPU/thread records are in deliveries.

Owning registry `.slurm/runs/20261026-hdm102.json`; remote root
`/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue102`.
Primary command `sbatch --parsable main-v1/full_main.sbatch`; R follow-up
`sbatch --parsable r-tol-v1/run.sbatch`. The checked-in harness/README describes
staging names; source and environment snapshots remain in that run. No local
Julia computation, no merge, and no automatic cleanup were performed.
