#!/usr/bin/env bash
# Starts training for each restarted episode as its search ends; stops when all four have started.
cd /scratch2/exaFLOPs/srl_yam_run/pipe
for i in $(seq 1 1400); do sleep 60; n=0; while read N A; do [ -f .rs_$N ] && { n=$((n+1)); continue; }; s=$(squeue -h -j $A -o %i 2>/dev/null) || continue; [ -n "$s" ] && continue; touch .rs_$N; ./rs.sh $N $A; grep -E "^schedule|STEP_FAILED|PREP_DONE" p$N-$A.log | cut -c1-260; n=$((n+1)); done < restart.ids; [ $n = 4 ] && break; done; date
