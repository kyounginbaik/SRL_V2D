#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=J33-p1
#SBATCH --array=0-16
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/ep33_p1-%A_%a.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
T1=$R/refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3; T2=$R/refinement_20261021_ep33/search/tail/c1_12.4_16.5; T3=$R/refinement_20261021_ep33/search/chainB/c1_7.3_9.0; G=$J/ep33/p1
cd /tmp; K=$SLURM_ARRAY_TASK_ID
case $K in
 0) rel $T1 $E33 $G/T1_relL15.3 left blue_cup 15.3 15.9;;
 1) rel $T1 $E33 $G/T1_relL15.6 left blue_cup 15.6 16.2;;
 2) rel $T1 $E33 $G/T1_relR14.4 right wooden_spoon 14.4 15.0 && rel $G/T1_relR14.4 $E33 $G/T1_relR14.4_relL15.6 left blue_cup 15.6 16.2;;
 3) rel $T1 $E33 $G/T1_relR14.7 right wooden_spoon 14.7 15.3 && rel $G/T1_relR14.7 $E33 $G/T1_relR14.7_relL15.6 left blue_cup 15.6 16.2;;
 4) rel $T2 $E33 $G/T2_relL15.6 left blue_cup 15.6 16.2;;
 5) rel $T2 $E33 $G/T2_relR14.4 right wooden_spoon 14.4 15.0 && rel $G/T2_relR14.4 $E33 $G/T2_relR14.4_relL15.6 left blue_cup 15.6 16.2;;
 6) seat $T1 $E33 $G/T1_seat3.5 3.5 3.0 13.6;;
 7) seat $T1 $E33 $G/T1_seat4.5 4.5 3.2 13.6;;
 8) seat $T1 $E33 $G/T1_seat12.8 12.8 3.0 13.6;;
 9) seat $T2 $E33 $G/T2_seat3.5 3.5 3.0 13.6;;
 10) seat $T2 $E33 $G/T2_seat4.5 4.5 3.2 13.6;;
 11) seat $T1 $E33 $G/T1_seat4.5_rot 4.5 3.2 13.6 0.5 --rot-only;;
 12) ilc $T1 $G/T1_ilcL_cup left blue_cup 2.0 15.0 15.6 $E33 2 0.3;;
 13) ilc $T1 $G/T1_ilcR_spoon right wooden_spoon 3.2 13.6 14.2 $E33 2 0.3;;
 14) silc $T1 $G/T1_silc_g0.3 3.2 13.6 2 0.3 $E33;;
 15) seat $T3 $E33 $G/T3_seat4.5 4.5 3.2 13.6;;
 16) rel $T3 $E33 $G/T3_relR14.4 right wooden_spoon 14.4 15.0 && rel $G/T3_relR14.4 $E33 $G/T3_relR14.4_relL15.6 left blue_cup 15.6 16.2;;
esac
