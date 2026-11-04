#!/usr/bin/env bash
# Waits for every VOC comparison score to finish, then prints the comparison.
cd /scratch2/exaFLOPs/srl_yam_run/pipe; J=$(awk '{print $3}' novoc.ids voc_rerun.ids | paste -sd,)
for i in $(seq 1 1200); do sleep 60; s=$(squeue -h -j $J -o %i 2>/dev/null) || continue; [ -z "$s" ] && break; done; python3 voc_compare.py; date
