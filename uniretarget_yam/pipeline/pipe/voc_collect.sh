#!/usr/bin/env bash
# Collects the VOC reruns: stops at each episode whose videos are done.
cd /scratch2/exaFLOPs/srl_yam_run/pipe
for i in $(seq 1 590); do sleep 60; while read n t c a v; do [ -f .vcollected_$n ] && continue; [ -z "$(squeue -h -j $v -o %i 2>/dev/null)" ] || continue; L=vv$n-$v.log; touch .vcollected_$n; EP=$(printf recon_%06d $n); RN=${EP}_voc; [ $n = 5 ] && RN=recon_000005_g1_voc; if [ -f $L ] && grep -q VIDEO_DONE $L; then RN=$RN VT=-voc ./collect.sh $n; else echo "ep$n no videos (job $v)"; fi; python3 /scratch2/exaFLOPs/srl_yam_run/pipe/voc_show.py $n $RN; date; exit 0; done < voc_rerun.ids; done; echo TIMEOUT
