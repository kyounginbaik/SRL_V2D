#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=N16-ilc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/logs/ilc-%A_%a.log
# reference-level wrist ILC (streamK ilc_axis.py). usage: WORLD=... sbatch --array=0-(N-1) ilc.sbatch SRC OUT_BASE "NAME T0 T1 TCLEAR AXIS GROT GPOS SIGMA ITER" [...]
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
SRC=$1; O=$2; shift 2; A=("$@"); set -- ${A[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $N/code/ilc_axis.py --source $SRC --out $O/$1 --side right --object wooden_spoon --t0 $2 --t1 $3 --t-clear $4 --rot-axis $5 --gain-rot $6 --gain-pos $7 --sigma-s $8 --iterations $9 --episode-dir $EPDIR
