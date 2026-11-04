#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --job-name=J33-p2
#SBATCH --array=0-7
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/ep33_p2-%A_%a.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
HV=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/code/hold_variant.py
hold() { $PY -B -u $HV --teacher $1 --episode-dir $E33 --out $2 --side right --object wooden_spoon --t-hold $3 --ramp 0.2; }
T1=$R/refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3; T2=$R/refinement_20261021_ep33/search/tail/c1_12.4_16.5; G=$J/ep33/p2
cd /tmp; K=$SLURM_ARRAY_TASK_ID
case $K in
 0) hold $T1 $G/T1_hold14.3 14.3;;
 1) hold $T1 $G/T1_hold14.6 14.6 && rel $G/T1_hold14.6 $E33 $G/T1_hold14.6_relL15.6 left blue_cup 15.6 16.2;;
 2) hold $T1 $G/T1_hold14.0 14.0 && rel $G/T1_hold14.0 $E33 $G/T1_hold14.0_relL15.6 left blue_cup 15.6 16.2;;
 3) hold $T2 $G/T2_hold14.3 14.3;;
 4) hold $T2 $G/T2_hold14.6 14.6 && rel $G/T2_hold14.6 $E33 $G/T2_hold14.6_relL15.6 left blue_cup 15.6 16.2;;
 5) hold $T2 $G/T2_hold14.0 14.0 && rel $G/T2_hold14.0 $E33 $G/T2_hold14.0_relL15.6 left blue_cup 15.6 16.2;;
 6) hold $T1 $G/T1_hold14.3b 14.3 && rel $G/T1_hold14.3b $E33 $G/T1_hold14.3_relL16.0 left blue_cup 16.0 16.6;;
 7) hold $T2 $G/T2_hold14.3b 14.3 && rel $G/T2_hold14.3b $E33 $G/T2_hold14.3_relL16.0 left blue_cup 16.0 16.6;;
esac
