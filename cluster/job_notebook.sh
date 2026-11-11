#!/bin/bash
# ===========================================================================
# Solstorm (SGE) job: set the parameters of notebooks/04_run_gdsddp.ipynb and EXECUTE the notebook
# in place, so the pulled-back notebook carries the outputs of the full-size run.
#
#   printf '%s\n' T=102 K=200 N_LATTICE_PATHS=30000 SDDP_ITERATIONS=40 N_SIM_PATHS=500 \
#          'H_VALUES=[0.5, 1.0]' 'RUN_NAME="full_notebook"' > gdsddp_public/cluster/nb_params.txt
#   qsub -l hostname=compute-5-1 -pe smp 4 -N nb_full \
#        -v NB_PARAMS_FILE=gdsddp_public/cluster/nb_params.txt,THREADS=4 gdsddp_public/cluster/job_notebook.sh
#
# Submit from ~/projects/masteroppgave. The file holds one NAME=VALUE per line, VALUE being Python
# source (strings need quotes). A file is used because qsub -v splits its value on commas, which
# would cut a list such as [0.5, 1.0] in two.
# ===========================================================================
#$ -N gdsddp_nb
#$ -cwd
#$ -S /bin/bash
#$ -o gdsddp_public/cluster/logs/
#$ -j y

set -euo pipefail
mkdir -p gdsddp_public/cluster/logs
module load Miniconda3/23.10.0-1 2>/dev/null || true
module load gurobi/13.0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-gdsddp}"
T="${THREADS:-1}"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
# nbconvert and ipykernel must already be in the env (compute nodes cannot build wheels; install once
# on the login node:  conda install -n gdsddp -c conda-forge nbconvert ipykernel nbformat)
python -c "import nbconvert, ipykernel, nbformat" || { echo "nbconvert/ipykernel missing in env ${CONDA_ENV:-gdsddp}: see the note above"; exit 1; }
PF="${NB_PARAMS_FILE:-}"
echo "[job] host=$(hostname) job=${JOB_ID:-NA} threads=$T params file: ${PF:-<none: notebook defaults>}"
if [ -n "$PF" ]; then
  [ -r "$PF" ] || { echo "parameter file $PF not readable"; exit 1; }
  PF=$(readlink -f "$PF")
fi
cd gdsddp_public
NB=${NOTEBOOK:-notebooks/04_run_gdsddp.ipynb}
if [ -n "$PF" ]; then
  python scripts/set_notebook_params.py "$NB" --file "$PF"
fi
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$NB"
echo "[job] executed $NB"
