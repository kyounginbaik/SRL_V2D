#!/usr/bin/env bash
# VOC ablation: time schedule and progress schedule against the existing no-VOC runs, on ep5 (g1, g2), ep30 and ep37. Tests and scoring stay at VOC 0.
cd /scratch2/exaFLOPs/srl_yam_run/pipe; B=big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090; A=$B,suma_rtx4090,gigabyte_a5000,asus_a5000,suma_a6000,gigabyte_a6000; P=base_suma_rtx3090,dell_rtx3090; : > voc_ablate.ids
sub() { # episode grip tag curriculum vocargs name
  local E=ALL,TAG=$3,CURR=$4,VOCARGS=$5; [ -n "$2" ] && E=$E,GRIP=$2; T=$(sbatch --parsable --qos=big_qos -J $6 -p $A --export=$E train.sbatch $1 42); EP=$(printf recon_%06d $1); D=U${2:+-$2}$3-seed42; C=$(sbatch --parsable -J s$6 -p $P --dependency=afterok:$T --export=ALL,ARM=${EP}${2:+_$2}${3//-/_}_s42,DIRN=$D score.sbatch $1 42); echo "$6 $1 $D $T $C" | tee -a voc_ablate.ids; }
PROG=+adaptive_voc.mode=progress
sub 5 g1 -vocp yam_frame0_50m $PROG t5g1p
sub 5 g2 -vocp yam_frame0_50m $PROG t5g2p
sub 30 "" -vocp yam_frame0_50m $PROG t30p
sub 30 "" -voc yam_voc_50m "" t30v
sub 37 "" -vocp yam_frame0_50m $PROG t37p
sub 37 "" -voc yam_voc_50m "" t37v
