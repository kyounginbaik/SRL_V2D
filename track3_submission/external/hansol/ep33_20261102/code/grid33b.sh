#!/bin/bash
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
cd /data1/hansol/mink_S1_20261031
/home/hansol/v2d/.venvs/v2d-host/bin/python -B code/composeGrid.py /data1/hansol/ep33_20261102/out/ep33_final_grid.mp4 --cols 2 --scale 0.5   "HGA (before)=/home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGA_sharpa/videos_sharpa/epi33.mp4"   "HGA + roll (new)=/data1/hansol/mink_S1_20261031/out_contact/recon_E14_best_B2A_S1_HGAR8s15/epi33.mp4"   "HGB (before)=/home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGB_sharpa/videos_sharpa/epi33.mp4"   "HGB + roll (new)=/data1/hansol/mink_S1_20261031/out_contact/recon_E14_best_B2A_S1_HGBR8s15/epi33.mp4"
FF=/home/hansol/v2d/.venvs/v2d-host/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
$FF -y -loglevel error -i /data1/hansol/ep33_20261102/out/ep33_final_grid.mp4 -vf "select='eq(n\,176)+eq(n\,248)'" -vsync 0 /data1/hansol/ep33_20261102/out/finalgrid_%d.jpg
