#!/bin/bash
# One full-horizon experiment on Solstorm: build the lattice (8 threads), then one single-slot
# training job per policy (risk neutral + every h given), all holding on the build.
#
#   cd ~/projects/masteroppgave
#   gdsddp_public/cluster/submit_full.sh <name> <T> <K> <paths> <iterations> "<h values>" <node1> [node2 node3 ...]
#   gdsddp_public/cluster/submit_full.sh full 102 200 20000 30 "0.5 1.0" compute-5-1 compute-5-3 compute-5-4
#
# Policies are round-robined over the nodes given (a single node holds them all fine: each job uses
# one core). Logs: gdsddp_public/cluster/logs/<name>_*.o<jobid>. Afterwards, on the Mac:
#   rsync -avP sigurdtv@solstorm-login.iot.ntnu.no:~/projects/masteroppgave/gdsddp_public/results/ gdsddp_public/results/
#   python3 gdsddp_public/scripts/run_experiment.py report --name <name>
set -euo pipefail
[ "$#" -ge 7 ] || { echo "usage: submit_full.sh <name> <T> <K> <paths> <iterations> \"<h values>\" <node1> [node2 ...]"; exit 1; }
NAME=$1; T=$2; K=$3; PATHS=$4; IT=$5; HS=$6; shift 6
NODES=("$@")
mkdir -p gdsddp_public/cluster/logs
jid() { echo "$1" | grep -o '[0-9][0-9]*' | head -1; }
B=$(jid "$(qsub -N "${NAME}_build" -l hostname="${NODES[0]}" -pe smp 8 -v ARGS="build --name $NAME --T $T --K $K --paths $PATHS",THREADS=8 gdsddp_public/cluster/job.sh)")
echo "  [$B] ${NAME}_build -> ${NODES[0]} (8 slots)"
i=0
for h in 0 $HS; do
  node=${NODES[$((i % ${#NODES[@]}))]}
  tag=$([ "$h" = "0" ] && echo rn || echo "h$h")
  J=$(jid "$(qsub -N "${NAME}_${tag}" -l hostname="$node" -pe smp 1 -hold_jid "$B" -v ARGS="train --name $NAME --h $h --iterations $IT" gdsddp_public/cluster/job.sh)")
  echo "  [$J] ${NAME}_${tag} -> $node (holds $B)"
  i=$((i + 1))
done
