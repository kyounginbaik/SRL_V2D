#!/bin/bash
# usage: chain.sh ROOT SRC EPD SEED CUPMESH SPOONMESH "H:T0-T1 H:T0-T1 ..." [W] [SCALE] [DEP]
# H = R (right hand, protect cup) or L (left hand, protect spoon). Submits one rsc_j job per window, each dependent on the previous.
ROOT=$1; SRC=$2; EPD=$3; SEED=$4; CUPMESH=$5; SPOONMESH=$6; WINS=$7; W=${8:-1.0}; SCALE=${9:-0.5}; DEP=${10:-}
J=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233; k=0; S=$SRC
for w in $WINS; do H=${w%%:*}; r=${w#*:}; T0=${r%-*}; T1=${r#*-}; k=$((k+1)); O=$ROOT/c${k}_${H}_${T0}_${T1}
  if [ $H = L ]; then SIDE=left; PROT=wooden_spoon; else SIDE=right; PROT=blue_cup; fi
  jid=$(SRC=$S EPD=$EPD OUT=$O T0=$T0 T1=$T1 SEED=$((SEED+k)) W=$W SCALE=$SCALE SIDE=$SIDE PROT=$PROT CUPMESH=$CUPMESH SPOONMESH=$SPOONMESH sbatch --parsable --job-name=J-rsc-$(basename $ROOT)-c$k ${DEP:+--dependency=afterok:$DEP} $J/code/rsc_j.sbatch)
  echo "submitted $jid -> $O (dep ${DEP:-none})"; DEP=$jid; S=$O
done
