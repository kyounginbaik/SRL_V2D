#!/usr/bin/env bash
# Usage: ./pilot.sh N [search_seed] [train_seed]  Same command for every episode; nothing is tuned per episode.
# Pass or fail, every episode ends with videos: a failed search gets the reference and best attempt, a trained run gets the three comparison videos.
N=$1; SS=${2:-41}; TS=${3:-42}; P=base_suma_rtx3090,dell_rtx3090; G=$P,suma_rtx4090,gigabyte_a5000,asus_a5000,suma_a6000,gigabyte_a6000; cd /scratch2/exaFLOPs/srl_yam_run/pipe
A=$(sbatch --parsable -J p$N -p $P prep.sbatch $N $SS); B=$(sbatch --parsable -J t${N}s$TS -p $G --dependency=afterok:$A train.sbatch $N $TS); C=$(sbatch --parsable -J s${N}s$TS -p $P --dependency=afterok:$B score.sbatch $N $TS); ./record.sh $N $A $B $C $TS; echo "episode $N prep $A train $B score $C"
