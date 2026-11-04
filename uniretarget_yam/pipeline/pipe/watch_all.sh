#!/usr/bin/env bash
# Wakes when one of the current searches ends or crashes. Usage: watch_all.sh "n:jobid ..."
cd /scratch2/exaFLOPs/srl_yam_run/pipe
for i in $(seq 1 580); do sleep 60
  for x in $1; do n=${x%%:*}; j=${x##*:}; L=p$n-$j.log; [ -f $L ] || continue
    [ -f .seen_$j ] && continue
    if grep -qE 'Traceback' $L; then touch .seen_$j; echo "TRACEBACK ep$n $L"; grep -E 'Error' $L | tail -2 | cut -c1-200; date; exit 0; fi
    if ! squeue -j $j -h 2>/dev/null | grep -q .; then touch .seen_$j; echo "ep$n search ended: $(grep -E "^PREP_DONE $n|STEP_FAILED" $L | tail -1)"; grep -E 'mean_tip_cm' $L | tail -1 | grep -oE '"mean_tip_cm": [0-9.]+|"palm_cm": [0-9.]+' | tr '\n' ' '; echo; grep -E 'playback' $L | tail -2 | cut -c1-210; date; exit 0; fi
  done
done
echo TIMEOUT
