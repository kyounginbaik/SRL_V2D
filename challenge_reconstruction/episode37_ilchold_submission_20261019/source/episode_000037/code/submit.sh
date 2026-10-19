#!/bin/bash
set -euo pipefail
NODE=${1:-node100}
SEED=${2:-42}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be a nonnegative integer"; exit 2; }
case "$NODE" in node01|node100|node200) ;; *) echo 'Expected node01, node100, or node200'; exit 2;; esac
BASE=/data1/SRL_V2D/challenge_reconstruction/uniretarget_wrist_eigend_mixed_500k_ep01_20261013
sbatch --parsable --nodelist="$NODE" "$BASE/code/train.sbatch" "$SEED"
