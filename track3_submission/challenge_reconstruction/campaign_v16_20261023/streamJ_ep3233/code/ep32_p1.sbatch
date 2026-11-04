#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=J32-p1
#SBATCH --array=0-12
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/ep32_p1-%A_%a.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
T=$R/refinement_20261021_ep32/search/tail/c1_8.3_11.5; G=$J/ep32/p1
cd /tmp; K=$SLURM_ARRAY_TASK_ID
case $K in
 0) rel $T $E32 $G/T_relR11.2 right wooden_spoon 11.2 11.8;;
 1) rel $T $E32 $G/T_relR11.5 right wooden_spoon 11.5 12.1;;
 2) rel $T $E32 $G/T_relR10.9 right wooden_spoon 10.9 11.5;;
 3) seat $T $E32 $G/T_seat4.0 4.0 3.0 11.2;;
 4) seat $T $E32 $G/T_seat7.3 7.3 3.0 11.2;;
 5) seat $T $E32 $G/T_seat5.0_on4.2 5.0 4.2 11.2;;
 6) seat $T $E32 $G/T_seat9.6 9.6 3.0 11.2;;
 7) seat $T $E32 $G/T_seat9.6_on8.6 9.6 8.6 11.2;;
 8) seat $T $E32 $G/T_seat4.0_rot 4.0 3.0 11.2 0.5 --rot-only;;
 9) ilc $T $G/T_ilcL_cup left blue_cup 1.0 11.5 12.0 $E32 2 0.3;;
 10) ilc $T $G/T_ilcR_spoon right wooden_spoon 3.2 11.2 11.6 $E32 2 0.3;;
 11) silc $T $G/T_silc_g0.3 3.2 11.2 2 0.3 $E32;;
 12) silc $T $G/T_silc_g0.2_4.3 4.3 11.2 2 0.2 $E32;;
esac
