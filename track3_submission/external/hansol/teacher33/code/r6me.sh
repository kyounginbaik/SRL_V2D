#!/bin/bash
# R6M end-of-clip fix (main session request 23:58): burst beam on R6M_C final table, right fingers held from 13.9 s, E1 wrist held from 15.0 s, E2 wrist free to 17.25 s
set -euo pipefail
cd /data1/hansol/teacher33
D=/data1/hansol/teacher33/runs; R=$D/cw; T=/data1/hansol/teacher33/teacher; TGD=/data1/hansol/retarget33/targets/R33_HGA_r6
WE=/data1/hansol/teacher33/worlds/R33r6M/ep33/episode_corrected/episode_000033
export E31_TGT=$TGD T33EPD=$WE T33TGT=$TGD TG=R33_HGA_r6 ACT=$D/distill/DR6Mm1
export WB=wbeamHGg.py SH=4,4,5,280,322 NC=60 WK=24 HANDS=wooden_spoon:right,blue_cup:none RS=right
W=13.75:14.00,14.00:14.25,14.25:14.50,14.50:14.75,14.75:15.00,15.00:15.25,15.25:15.50,15.50:15.75,15.75:16.00
export T33=$T/FN_r6M T33NOTE="T33 teacher FN_r6M (team streamB ilc2 iter1 ported by delta_port_t33 into the moved-start world R33r6M = scene start poses at R33_HGA_r6 frame 0, then fcem33 vs R33_HGA_r6 with finger-count cost; not shipped); shipped actor = actor_grid_c distilled from it + team finger margins stage 1 + window-beam R6M_A/B/C residuals + end-of-clip burst (right hand held from 13.9 s)"
L=""
for V in E1:15.0:4511 E2:17.25:4512; do
  N=${V%%:*}; R2=${V#*:}; TW=${R2%%:*}; SEED=${R2#*:}; FZS=right,13.9,0.2,$TW
  jb=$(FZ=$FZS bash code/t33hg.sh burst R6M_$N $R/R6M_C/final_0.npz none $W "" $SEED)
  js=$(sbatch --parsable --export=ALL --dependency=afterany:$jb code/save33fz.sbatch $ACT $R/R6M_$N/final_0.npz $D/save_R6M_$N $FZS)
  jp=$(sbatch --parsable --dependency=afterany:$js -c 12 -J T33-pertR6M$N code/job.sbatch perturb33.py /data1/hansol/teacher33/out/perturb_R6M$N 16 12 R6M_$N=$D/save_R6M_$N/actor)
  jt=$(sbatch --parsable --dependency=afterany:$js -c 2 -J T33-tabR6M$N code/job.sbatch t33table.py auto $D/save_R6M_$N/actor/evaluation/execution/trajectory.npz:R6M_${N}_vs_r6)
  jv=$(sbatch --parsable --export=ALL --dependency=afterany:$js code/t33svst.sbatch R6M_$N $D/save_R6M_$N)
  jk=$(sbatch --parsable --dependency=afterany:$jv code/t33pkg.sbatch R33_HGA_r6_R6M_$N track3_FINAL_ep33T33_R6M${N}_r6moved_20261104)
  L="$L | $N (FZ $FZS): beam $jb save $js pert $jp table $jt svst $jv pkg $jk"
done
echo "$L"; echo "$(date +%FT%T) R6M end-of-clip fix (main session request; table R6M_C final_0, windows 13.75-16.00, SH 4,4,5,280,322, NC 60, WK 24)$L" >> out/jobs.txt
