#!/usr/bin/env bash
# Usage: ./record.sh N prep_job train_job score_job [train_seed]   Adds the failure video, the baseline run and the three videos to a pilot.
N=$1; A=$2; B=$3; C=$4; TS=${5:-42}; P=base_suma_rtx3090,dell_rtx3090; G=$P,suma_rtx4090,gigabyte_a5000,asus_a5000,suma_a6000,gigabyte_a6000; cd /scratch2/exaFLOPs/srl_yam_run/pipe
F=$(sbatch --parsable -J fp$N -p $G --dependency=afternotok:$A diag/fail_video.sbatch $N live); L=$(sbatch --parsable -J a${N}s$TS -p $G --dependency=afterok:$A train_a.sbatch $N $TS); V=$(sbatch --parsable -J v$N -p $P --dependency=afterany:$C,afterok:$L video.sbatch $N $TS); echo "  record: fail video $F baseline $L videos $V"
