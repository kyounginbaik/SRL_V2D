#!/usr/bin/env bash
# Usage: ./ablate.sh N PREP_JOB  Three ablation arms with the same penalty as the full method: baseline (Mink, no updates), no search (Mink, updates), no updates (searched grasp, no updates). A prep job id delays them; low priority so episodes go first.
N=$1; PREP=${2:-}; EP=$(printf recon_%06d $N); G=base_suma_rtx3090,dell_rtx3090,suma_rtx4090,gigabyte_a5000,asus_a5000,suma_a6000,gigabyte_a6000; P=base_suma_rtx3090,dell_rtx3090; cd /scratch2/exaFLOPs/srl_yam_run/pipe
any=${PREP:+--dependency=afterany:$PREP}; ok=${PREP:+--dependency=afterok:$PREP}
S=$(sbatch --parsable --nice=100 -J ab${N}nosearch -p $G $any --export=ALL,NO_SEARCH=1 train.sbatch $N 42); SS=$(sbatch --parsable --nice=100 -J sab${N}nosearch -p $P --dependency=afterok:$S --export=ALL,ARM=${EP}_nosearch_s42,DIRN=U-nosearch-seed42 score.sbatch $N 42)
U=$(sbatch --parsable --nice=100 -J ab${N}noupdate -p $G $ok --export=ALL,TAG=-search,FROM_POOL=1 train_a.sbatch $N 42); SU=$(sbatch --parsable --nice=100 -J sab${N}noupdate -p $P --dependency=afterok:$U --export=ALL,ARM=${EP}_noupdate_s42,DIRN=A-search-seed42 score.sbatch $N 42)
echo "ep$N no-search train $S score $SS; no-update train $U score $SU"
