# Stream D shared paths
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023; V22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; SD=$V22/streamJ8_19_25_28_38/ep28
RM7=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
T1=$R/refinement_20261021_ep1/lid/shift_0.02_0_0_12.8
T20=$RM7/diagnostics/episode20_hold/final_pitcher_release_75_neutral/trial1
W1=$SD/ep1/world/episode_target/episode_000001; W20=$SD/ep20/world/episode_target/episode_000020
P1=left=white_pot,right=white_pot_lid; P20=left=blue_cup,right=water_pitcher
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
