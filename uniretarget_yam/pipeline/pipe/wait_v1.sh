#!/usr/bin/env bash
cd /scratch2/exaFLOPs/srl_yam_run/pipe
for i in $(seq 1 300); do sleep 30; s=$(squeue -h -j 2682798 -o %i 2>/dev/null) || continue; [ -z "$s" ] && break; done
grep -E "objects moved|VIDEO_DONE|Traceback|Error" v1-2682798.log | cut -c1-200
for n in Fixed_Reference_Low_Level UniRetarget_Physics_Result UniRetarget_Reference_Refinement; do ./pip.sh 1 $n >/dev/null 2>&1 && cp ../videos/recon_000001/ep01_${n}_With_Original.mp4 ../videos/successes/ep01/; done
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; timeout 300 apptainer exec --containall --no-home --home /tmp --env MUJOCO_GL=egl,PYTHONDONTWRITEBYTECODE=1 -B "$H:/uniretarget_yam" -B $R/videos/recon_000001:/vid -B $R/pipe/diag:/d --pwd /d $R/flash_chord_yam.sif python hold_check.py /uniretarget_yam/episodes/recon_000001 /vid/mU.npz 2>&1 | tail -4 | cut -c1-300
date
