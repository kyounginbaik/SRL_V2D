#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=J32r-p1
#SBATCH --array=0-4
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/ep32_57_p1-%A_%a.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
T=$J/ep32/probe_cup57bb/teacher_tail; G=$J/ep32/w57/p1
cd /tmp; K=$SLURM_ARRAY_TASK_ID
case $K in
 0) silc $T $G/silc_g0.5 3.0 11.2 2 0.5 $E32R;;
 1) silc $T $G/silc_g0.3 3.0 11.2 3 0.3 $E32R;;
 2) ilc $T $G/ilcR_g0.3 right wooden_spoon 3.0 11.2 11.6 $E32R 2 0.3;;
 3) seat $T $E32R $G/seat4.0 4.0 3.0 11.2;;
 4) silc $T $G/silc_g0.5_abs 3.0 11.2 2 0.5 $E32R --absolute;;
esac
