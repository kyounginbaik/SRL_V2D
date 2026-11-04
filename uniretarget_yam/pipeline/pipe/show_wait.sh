#!/usr/bin/env bash
cd /scratch2/exaFLOPs/srl_yam_run/pipe
for i in $(seq 1 300); do sleep 60; s=$(squeue -h -j 2682486,2682795 -o %i 2>/dev/null) || continue; [ -z "$s" ] && break; done
for x in "5 2682486" "10 2682795"; do set -- $x; L=vv$1-$2.log; echo "== $L"; grep -E "^U |\"first\"|VIDEO_DONE|Traceback|Error" $L | cut -c1-170; for n in Fixed_Reference_Low_Level UniRetarget_Physics_Result UniRetarget_Reference_Refinement; do F=../videos/$(printf recon_%06d $1)-voc/$(printf ep%02d $1)_$n.mp4; [ -f $F ] && VT=-voc ./pip.sh $1 $n; done; done; date
