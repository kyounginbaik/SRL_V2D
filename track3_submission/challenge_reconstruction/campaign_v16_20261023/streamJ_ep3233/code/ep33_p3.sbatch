#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --job-name=J33-p3
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/ep33_p3-%A_%a.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
T1=$R/refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3; T2=$R/refinement_20261021_ep33/search/tail/c1_12.4_16.5; G=$J/ep33/p3
cd /tmp; K=$SLURM_ARRAY_TASK_ID
case $K in
 0) ilc $T2 $G/T2_ilcL12.3_g0.5 left blue_cup 12.3 13.8 14.0 $E33 2 0.5;;
 1) ilc $T2 $G/T2_ilcL12.3_g0.3 left blue_cup 12.3 13.8 14.0 $E33 2 0.3;;
 2) ilc $T1 $G/T1_ilcL12.3_g0.5 left blue_cup 12.3 13.8 14.0 $E33 2 0.5;;
 3) ilc $T2 $G/T2_ilcL12.3_g0.5b left blue_cup 12.3 13.8 14.0 $E33 1 0.5 && rel $G/T2_ilcL12.3_g0.5b/iter0 $E33 $G/T2_ilcL12.3_g0.5_relL16.0 left blue_cup 16.0 16.6;;
 4) ilc $T2 $G/T2_ilcL10.5_g0.5 left blue_cup 10.5 13.8 14.0 $E33 2 0.5;;
 5) ilc $J/ep33/p1/T2_relR14.4 $G/T2relR_ilcL12.3_g0.5 left blue_cup 12.3 13.8 14.0 $E33 2 0.5;;
esac
