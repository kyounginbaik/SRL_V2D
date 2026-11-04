#!/bin/bash
# package of R9A (R7 + approach guide at the 10 s cup entry, r6 recon)
set -e
R=/data1/hansol/retarget33; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33; export TG=R33_HGA_r6
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/runs/save_ep33_b1/actor
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033
W=$R/worlds/R33_HGA_r6/ep33/episode_corrected/episode_000033; cd $R
[ -d $W ] || { j=$(E31_TGT=$R/targets/$TG sbatch --parsable -c 4 -J R33-mkref6 code/job.sbatch mkref.py 33 $K/runs/save_K33_cwHGAfix_ep33_HGA_c9/actor $M R33_HGA_r6); while squeue -h -j $j | grep -q .; do sleep 15; done; }
j=$(E31_TGT=$R/targets/$TG E31_EPD=$W sbatch --parsable -c 4 -J R33-saveR9A code/job.sbatch mresH.py save --ep 33 --actor $A --table $R/runs/cw/R9A_C/final_0.npz --meshes $M --out $R/runs/save_R33_HGA_r6_R9A --workers 4)
while squeue -h -j $j | grep -q .; do sleep 15; done; grep -E "^FINAL" $(ls $R/logs/*-$j.log) | cut -c1-200
j=$(sbatch --parsable code/r33svst.sbatch R9A $R/runs/save_R33_HGA_r6_R9A); while squeue -h -j $j | grep -q .; do sleep 20; done; tail -1 $R/logs/R33-svst-$j.log
j=$(sbatch --parsable code/r33pkg.sbatch R33_HGA_r6_R9A track3_FINAL_ep33R33R9A_r6_20261104); while squeue -h -j $j | grep -q .; do sleep 10; done; grep -E "VERIFY|KIT|SHA" $R/logs/R33-pkg-$j.log
P=$R/packages/track3_FINAL_ep33R33R9A_r6_20261104
SC=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py
j=$(sbatch --parsable -p debug --nodelist=node100 -c 2 --mem=12G --time=00:20:00 -J R33-scoreR9A -o $R/logs/%x-%j.log --wrap "export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages; cd /tmp; /home/hansol/v2d/.venvs/v2d-host/bin/python -B $SC $P/all_20_episodes/track3_submission.parquet $P/all_20_episodes/results /data1/hansol/ep33_20261102/k33x/targets/HGAZr6_HGA $P/score_vs_r6.json 33")
while squeue -h -j $j | grep -q .; do sleep 10; done; grep -o '"objects".\{0,300\}' $P/score_vs_r6.json | head -2
echo PKG_R9A_DONE
