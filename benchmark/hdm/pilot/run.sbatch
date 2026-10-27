#!/bin/bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
: "${RUNTAG:?Set RUNTAG and pass --job-name and --comment at submission}"
: "${TASKS:?Set TASKS to the task-list file}"
: "${SLURM_MEM_PER_NODE:?Submit with --mem so each worker step gets a share of the job memory}"
JULIA=${JULIA:-julia}
OUTDIR=${OUTDIR:-$PWD/outputs/$RUNTAG}
export JULIA OUTDIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_PKG_SERVER="" JULIA_PKG_OFFLINE=true
WORKERS=$((SLURM_CPUS_PER_TASK / 2))
STEP_MEM=$(( (SLURM_MEM_PER_NODE - 256) / WORKERS ))M
export STEP_MEM
mkdir -p "$OUTDIR"
echo "[$(date -Is)] run=$RUNTAG job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} workers=$WORKERS"

while read -r case first last; do
    [ -n "$case" ] || continue
    [ -f "$OUTDIR/case-$case-reps-$first-$last.csv" ] || echo "$case $first $last"
done < "$TASKS" | xargs -r -n3 -P "$WORKERS" sh -c '
    exec srun --exact -n1 -c2 --mem="$STEP_MEM" "$JULIA" --startup-file=no --project=. worker.jl "$1" "$2" "$3" "$OUTDIR"
' _
echo "[$(date -Is)] done run=$RUNTAG"
