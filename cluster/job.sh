#!/bin/bash
# ===========================================================================
# Solstorm (NTNU, SGE) job: one step of scripts/run_experiment.py on a compute node.
#
#   qsub -l hostname=compute-5-1 -pe smp 8 -N full_build \
#        -v ARGS="build --name full --T 102 --K 200 --paths 20000" gdsddp_public/cluster/job.sh
#   qsub -l hostname=compute-5-1 -pe smp 1 -N full_rn -hold_jid <build job id> \
#        -v ARGS="train --name full --h 0" gdsddp_public/cluster/job.sh
#
# Submit from ~/projects/masteroppgave (the parent of gdsddp_public/). Environment as in the
# master_dir jobs: Miniconda + the gurobi module (token-server licence) + conda env gdsddp.
# Training is single-threaded (1 slot); the lattice build may use THREADS slots for K-means.
# NOTE: qsub -v splits its value list on COMMAS, so ARGS must not contain any: write contract
# sets with '+' (--contracts H2:27-52+Y2:53-102); parse_contracts accepts both.
# ===========================================================================
#$ -N gdsddp
#$ -cwd
#$ -S /bin/bash
#$ -o gdsddp_public/cluster/logs/
#$ -j y

set -euo pipefail
mkdir -p gdsddp_public/cluster/logs
: "${ARGS:?ARGS not set (qsub -v ARGS=\"build --name ...\")}"

module load Miniconda3/23.10.0-1 2>/dev/null || true
module load gurobi/13.0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-gdsddp}"
T="${THREADS:-1}"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T

echo "[job] host=$(hostname) job=${JOB_ID:-NA} slots=${NSLOTS:-NA} threads=$T args: $ARGS"
cd gdsddp_public
python -u scripts/run_experiment.py $ARGS
