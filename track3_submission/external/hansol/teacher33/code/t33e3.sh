#!/bin/bash
# RPM_E2: RPM natural release ~14.9 s (r33endjit on save_RPM_C_f0), so hold the right hand from 14.7 s (release - 0.2 s), wrist from 15.2 s
set -euo pipefail
cd /data1/hansol/teacher33
D=/data1/hansol/teacher33/runs; R=$D/cw; T=/data1/hansol/teacher33/teacher; N=RPM_E2; PK=track3_FINAL_ep33T33_RPME2_RPmoved_20261104
W=13.75:14.00,14.00:14.25,14.25:14.50,14.50:14.75,14.75:15.00,15.00:15.25,15.25:15.50,15.50:15.75,15.75:16.00
FZS=right,14.7,0.2,15.2
export WB=wbeamHGg.py SH=4,4,5,280,322 NC=60 WK=24 HANDS=wooden_spoon:right,blue_cup:none RS=right
export TG=R33_RP TGD=/data1/hansol/retarget33/targets/R33_RP
export E31_TGT=$TGD T33TGT=$TGD T33EPD=/data1/hansol/teacher33/worlds/R33RPM/ep33/episode_corrected/episode_000033 ACT=$D/distill/DRPMm1 T33=$T/FN_RPM
export T33NOTE="T33 teacher FN_RPM (moved-start world R33RPM, fcem33 vs R33_RP; not shipped); shipped actor = actor_grid_c distilled + margins stage 1 + window beam RPM_A/B/C + end-of-clip burst RPM_E2 (right hand held from 14.7 s, wrist from 15.2 s)"
jb=$(FZ=$FZS bash code/t33hg.sh burst $N $R/RPM_C/final_0.npz none $W "" 4532)
js=$(sbatch --parsable --export=ALL --dependency=afterany:$jb code/save33fz.sbatch $ACT $R/$N/final_0.npz $D/save_$N $FZS)
jp=$(sbatch --parsable --dependency=afterany:$js -c 12 -J T33-pert$N code/job.sbatch perturb33.py /data1/hansol/teacher33/out/perturb_$N 16 12 $N=$D/save_$N/actor)
jt=$(sbatch --parsable --dependency=afterany:$js -c 2 -J T33-tab$N code/job.sbatch t33table.py auto $D/save_$N/actor/evaluation/execution/trajectory.npz:${N}_vs_RP)
jv=$(sbatch --parsable --export=ALL --dependency=afterany:$js code/t33svst.sbatch $N $D/save_$N)
jk=$(sbatch --parsable --dependency=afterany:$jv:$jp:$jt code/t33pkg.sbatch ${TG}_$N $PK)
jr=$(sbatch --parsable --dependency=afterany:$jk code/t33report.sh $N $PK $D/save_$N $js $jp $jt $jk)
L="beam $jb save $js pert $jp table $jt svst $jv pkg $jk report $jr"; echo "$L"
echo "$(date +%FT%T) RPM_E2 end-of-clip chain (FZ $FZS; RPM release ~14.9 s): $L" >> out/jobs.txt
echo "| RPM_E2 (TG R33_RP, world R33RPM; release ~14.9 s so t_h 14.7) | $FZS | $jb | $js | $jp | $jt | $jv | $jk | $jr | queued after R6M2_E1/RPM_E1 free CPUs | ~01:35 | $PK |" >> out/STATUS_20261105.md
