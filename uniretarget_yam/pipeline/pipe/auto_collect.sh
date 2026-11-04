#!/usr/bin/env bash
# Waits for the rerender of ep37 and each queued episode's video job, collects passes, and stops at the first news.
cd /scratch2/exaFLOPs/srl_yam_run/pipe
for i in $(seq 1 590); do sleep 60
  st=$(squeue -u $USER -h -o %j)
  for x in 30:2679712 20:2679730 19:2679736 32:2679742 1:2679768 10:2679774 28:2679780 25:2679786 6:2679792 15:2679798 35:2679804 36:2679810 38:2679855; do n=${x%:*}; a=${x#*:}; [ -f .finished_$n ] && continue; [ -z "$(squeue -h -j $a -o %i 2>/dev/null)" ] || continue; L=p$n-$a.log; [ -f $L ] && grep -q STEP_FAILED $L || continue; touch .finished_$n; ./finish.sh $n $a; done
  for n in 37 38 30 16 8 20 19 32 33 5 1 10 28 25 6 15 35 36; do L=$(ls -t v$n-*.log 2>/dev/null | head -1); [ -n "$L" ] || continue; [ -f .collected_$n ] && continue; if grep -q VIDEO_DONE $L && ! echo "$st" | grep -qE "^v$n$"; then touch .collected_$n; ./collect.sh $n; date; exit 0; fi; done
done; echo TIMEOUT
