#!/bin/bash
# OVERNIGHT PLAN on Solstorm: one 102-week lattice with K states (default 200), then in parallel
#   risk neutral;  thesis 2-contract set (H2 27-52, Y2 53-102) at h = 0.25 0.5 0.75 1.0 1.5;
#   no contracts at h = 0.75 1.0;  thesis 8-contract set at h = 0.75 1.0;
# plus a second lattice with K2 states (default 300): risk neutral and h = 0.75 (2c) as a resolution check.
# Also executes notebooks/04_run_gdsddp.ipynb in place with the full settings (job <name>_nb).
# Every training job uses ONE core and up to ~50 GB at K=200 (~75 GB at K=300): the node LPs carry one
# cut row per successor and grow with the iterations. Jobs are round-robined over the nodes given and
# CHAINED per node (each holds on the previous job of its node), so one job runs per node at a time
# (set CHAIN=0 to run them concurrently; two per 96 GB node were OOM-killed on 2026-11-10).
#
#   cd ~/projects/masteroppgave
#   gdsddp_public/cluster/submit_overnight.sh <node1> [node2 ...]                 # e.g. compute-5-1 compute-5-3 compute-5-4 compute-5-5 compute-5-6
#   K=200 K2=300 PATHS=30000 IT=50 SIM=500 NAME=full gdsddp_public/cluster/submit_overnight.sh <nodes>   # overrides
#   BUILD=0 ... : reuse the lattices already in results/<name>/ and results/<name>_K<K2>/ (no build jobs)
#   RN=0    ... : skip the risk-neutral job of lattice 1 (RN2=0 the one of lattice 2; default RN2=RN)
#   NB=0    ... : skip the notebook job;   K2=0 : skip the K2 lattice entirely
# (qsub -v splits on commas, hence '+' between contracts and a parameter FILE for the notebook job.)
set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: submit_overnight.sh <node1> [node2 ...]"; exit 1; }
NODES=("$@"); N=${#NODES[@]}
NAME=${NAME:-full}; T=${T:-102}; K=${K:-200}; K2=${K2:-300}; PATHS=${PATHS:-30000}; IT=${IT:-50}; SIM=${SIM:-500}
BUILD=${BUILD:-1}; RN=${RN:-1}; RN2=${RN2:-$RN}; NB=${NB:-1}; CHAIN=${CHAIN:-1}
C2="H2:27-52+Y2:53-102"
C8="Feb:5-8+Mar:9-13+Apr:14-17+Q2:14-26+Q3:27-39+Q4:40-52+Q1y2:53-65+Y2:53-102"
mkdir -p gdsddp_public/cluster/logs
jid() { echo "$1" | grep -o '[0-9][0-9]*' | head -1; }
i=0
k=0; while [ "$k" -lt "$N" ]; do LAST[$k]=""; k=$((k + 1)); done   # last job id submitted to each node
# the notebook job (~6 h on its own) takes the first node; training jobs round-robin over the others
if [ "$NB" = "1" ] && [ "$N" -gt 1 ]; then OFF=1; M=$((N - 1)); else OFF=0; M=$N; fi
slot() { echo $(( (i % M) + OFF )); }
node() { echo "${NODES[$(slot)]}"; }
holds() {           # <build id>: comma list of the ids this job must wait for ('' if none)
  local hl="$1"
  if [ "$CHAIN" = "1" ] && [ -n "${LAST[$(slot)]}" ]; then hl="${hl:+$hl,}${LAST[$(slot)]}"; fi
  echo "$hl"
}
submit_train() {   # <run name> <hold id> <h> <contracts|none|default> <tag>
  local run=$1 hold=$2 h=$3 cs=$4 tag=$5 jn args hl
  jn="${run}_$([ "$h" = "0" ] && echo rn || echo "h${h}${tag:+_$tag}")"
  args="train --name $run --h $h --iterations $IT --sim-paths $SIM"
  [ "$cs" != "default" ] && args="$args --contracts $cs"
  [ -n "$tag" ] && args="$args --tag $tag"
  hl=$(holds "$hold")
  if [ -n "$hl" ]; then
    J=$(jid "$(qsub -N "$jn" -l hostname="$(node)" -pe smp 1 -hold_jid "$hl" -v ARGS="$args" gdsddp_public/cluster/job.sh)")
  else
    J=$(jid "$(qsub -N "$jn" -l hostname="$(node)" -pe smp 1 -v ARGS="$args" gdsddp_public/cluster/job.sh)")
  fi
  echo "  [$J] $jn -> $(node) (holds ${hl:-nothing})"
  LAST[$(slot)]="$J"
  i=$((i + 1))
}
# ---- notebook 4 executed in place with the full settings (its own lattice; sequential policies, ~6 h)
if [ "$NB" = "1" ]; then
  PF="gdsddp_public/cluster/nb_params_${NAME}.txt"
  printf '%s\n' "T=$T" "K=$K" "N_LATTICE_PATHS=$PATHS" "SDDP_ITERATIONS=${NB_IT:-40}" "N_SIM_PATHS=$SIM" \
         "H_VALUES=[0.5, 1.0]" "RUN_NAME=\"${NAME}_notebook\"" > "$PF"
  J=$(jid "$(qsub -N "${NAME}_nb" -l hostname="${NODES[0]}" -pe smp 4 -v NB_PARAMS_FILE="$PF",THREADS=4 gdsddp_public/cluster/job_notebook.sh)")
  echo "  [$J] ${NAME}_nb (notebook 04 executed in place, parameters in $PF) -> ${NODES[0]}"
  LAST[0]="$J"
fi
# ---- lattice 1 (K) and its 10 policies
B=""
if [ "$BUILD" = "1" ]; then
  B=$(jid "$(qsub -N "${NAME}_build" -l hostname="${NODES[0]}" -pe smp 8 -v ARGS="build --name $NAME --T $T --K $K --paths $PATHS",THREADS=8 gdsddp_public/cluster/job.sh)")
  echo "  [$B] ${NAME}_build -> ${NODES[0]} (8 slots; K=$K, $PATHS paths)"
else
  [ -f "gdsddp_public/results/$NAME/lattice.npz" ] || { echo "BUILD=0 but results/$NAME/lattice.npz is missing"; exit 1; }
  echo "  reusing results/$NAME/lattice.npz"
fi
[ "$RN" = "1" ] && submit_train "$NAME" "$B" 0 default ""
for h in 0.25 0.5 0.75 1.0 1.5; do submit_train "$NAME" "$B" "$h" "$C2" 2c; done
for h in 0.75 1.0; do submit_train "$NAME" "$B" "$h" none 0c; done
for h in 0.75 1.0; do submit_train "$NAME" "$B" "$h" "$C8" 8c; done
# ---- lattice 2 (K2) resolution check: RN + h 0.75 with 2c
if [ "$K2" != "0" ]; then
  NAME2="${NAME}_K${K2}"; B2=""
  if [ "$BUILD" = "1" ]; then
    B2=$(jid "$(qsub -N "${NAME2}_build" -l hostname="${NODES[$((1 % N))]}" -pe smp 8 -v ARGS="build --name $NAME2 --T $T --K $K2 --paths $PATHS",THREADS=8 gdsddp_public/cluster/job.sh)")
    echo "  [$B2] ${NAME2}_build -> ${NODES[$((1 % N))]} (8 slots; K=$K2)"
  else
    [ -f "gdsddp_public/results/$NAME2/lattice.npz" ] || { echo "BUILD=0 but results/$NAME2/lattice.npz is missing"; exit 1; }
    echo "  reusing results/$NAME2/lattice.npz"
  fi
  [ "$RN2" = "1" ] && submit_train "$NAME2" "$B2" 0 default ""
  submit_train "$NAME2" "$B2" 0.75 "$C2" 2c
fi
echo "submitted over $N node(s); logs in gdsddp_public/cluster/logs/"
