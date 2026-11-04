#!/bin/bash
# 00:2x KST Nov 5: R6M2_E1 + RPM_E1 end-of-clip chains + STATUS reporters (main session request). usage: bash t33e2.sh
set -euo pipefail
cd /data1/hansol/teacher33
D=/data1/hansol/teacher33/runs; R=$D/cw; T=/data1/hansol/teacher33/teacher; ST=out/STATUS_20261105.md
W=13.75:14.00,14.00:14.25,14.25:14.50,14.50:14.75,14.75:15.00,15.00:15.25,15.25:15.50,15.50:15.75,15.75:16.00
FZS=right,13.9,0.2,15.0
export WB=wbeamHGg.py SH=4,4,5,280,322 NC=60 WK=24
chain() { # NAME PKG LABEL_TG_TAG DEP SEED  (env: TG ACT T33 T33NOTE T33EPD T33TGT E31_TGT HANDS RS ADVS HG OW)
  local N=$1 PK=$2 DEP=$3 SEED=$4
  jb=$(FZ=$FZS bash code/t33hg.sh burst $N $R/${N%_E1}_C/final_0.npz none $W "$DEP" $SEED)
  js=$(sbatch --parsable --export=ALL --dependency=afterany:$jb code/save33fz.sbatch $ACT $R/$N/final_0.npz $D/save_$N $FZS)
  jp=$(sbatch --parsable --dependency=afterany:$js -c 12 -J T33-pert$N code/job.sbatch perturb33.py /data1/hansol/teacher33/out/perturb_$N 16 12 $N=$D/save_$N/actor)
  jt=$(sbatch --parsable --dependency=afterany:$js -c 2 -J T33-tab$N code/job.sbatch t33table.py auto $D/save_$N/actor/evaluation/execution/trajectory.npz:${N}_vs_${TG#R33_})
  jv=$(sbatch --parsable --export=ALL --dependency=afterany:$js code/t33svst.sbatch $N $D/save_$N)
  jk=$(sbatch --parsable --dependency=afterany:$jv:$jp:$jt code/t33pkg.sbatch ${TG}_$N $PK)
  jr=$(sbatch --parsable --dependency=afterany:$jk code/t33report.sh $N $PK $D/save_$N $js $jp $jt $jk)
  echo "$N|$jb|$js|$jp|$jt|$jv|$jk|$jr|$PK|$DEP"
}
# R6M2_E1: both-hands flags as R6M2 (left hand keeps the cup; freeze is right-only)
export TG=R33_HGA_r6 TGD=/data1/hansol/retarget33/targets/R33_HGA_r6
export E31_TGT=$TGD T33TGT=$TGD T33EPD=/data1/hansol/teacher33/worlds/R33r6M/ep33/episode_corrected/episode_000033 ACT=$D/distill/DR6Mm1 T33=$T/FN_r6M
export HANDS=wooden_spoon:right,blue_cup:left RS=right,left ADVS=right,both HG=6,1.0 OW='{"wooden_spoon":3,"blue_cup":2}'
export T33NOTE="T33 teacher FN_r6M (moved-start world R33r6M, fcem33 vs R33_HGA_r6; not shipped); shipped actor = actor_grid_c distilled + margins stage 1 + both-hands window beam R6M2_A/B/C + end-of-clip burst R6M2_E1 (right hand held from 13.9 s, wrist from 15.0 s)"
C1=$(chain R6M2_E1 track3_FINAL_ep33T33_R6M2E1_r6moved_20261104 "" 4521)
# RPM_E1: right hand only, after the RPM package
export TG=R33_RP TGD=/data1/hansol/retarget33/targets/R33_RP
export E31_TGT=$TGD T33TGT=$TGD T33EPD=/data1/hansol/teacher33/worlds/R33RPM/ep33/episode_corrected/episode_000033 ACT=$D/distill/DRPMm1 T33=$T/FN_RPM
export HANDS=wooden_spoon:right,blue_cup:none RS=right; unset ADVS HG OW
export T33NOTE="T33 teacher FN_RPM (moved-start world R33RPM, fcem33 vs R33_RP; not shipped); shipped actor = actor_grid_c distilled + margins stage 1 + window beam RPM_A/B/C + end-of-clip burst RPM_E1 (right hand held from 13.9 s, wrist from 15.0 s)"
C2=$(chain RPM_E1 track3_FINAL_ep33T33_RPME1_RPmoved_20261104 69734 4531)
# reporters for chains already queued (R6M_E1, R6M_E2, RPM)
r1=$(sbatch --parsable --dependency=afterany:69886 code/t33report.sh R6M_E1 track3_FINAL_ep33T33_R6ME1_r6moved_20261104 $D/save_R6M_E1 69882 69883 69884 69886)
r2=$(sbatch --parsable --dependency=afterany:69892 code/t33report.sh R6M_E2 track3_FINAL_ep33T33_R6ME2_r6moved_20261104 $D/save_R6M_E2 69888 69889 69890 69892)
r3=$(sbatch --parsable --dependency=afterany:69734 code/t33report.sh RPM_C track3_FINAL_ep33T33_RPMCf0_RPmoved_20261104 $D/save_RPM_C_f0 69730 69731 69732 69734)
echo "C1 $C1"; echo "C2 $C2"; echo "R $r1 $r2 $r3"
echo "$(date +%FT%T) end-of-clip E1 chains (FZ $FZS, windows 13.75-16.00, wbeamHGg SH 4,4,5,280,322 NC 60 WK 24): R6M2_E1 [$C1] RPM_E1 [$C2]; STATUS reporters R6M_E1 $r1 R6M_E2 $r2 RPM_C $r3" >> out/jobs.txt
