# Stream T shared paths
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023; ST=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins; TS=$C/streamT_C356
T3=$C/streamB_tilt38/ep3/release_grid/pass2_s73/open1.5_frac0.65_handle_rs0.6_rm0.08
T5=$C/streamA2_crotch/ep5/ilc/gr0.3_gp0.3_s0.35/iter1
T6=$R/refinement_20261020_pot/power_grasp/retention_full/episode_000006/near_b_v2
O3=$C/streamB_tilt38/ep3/calibrated_inputs/episodes/episode_000003
O5=$R/refinement_20261018/calibrated_scene_candidates/episodes/episode_000005
O6=$R/episodes/episode_000006
W3=$ST/ep3/world/episode_compose/episode_000003; W5=$ST/ep5/world/episode_compose/episode_000005
TW3=$TS/ep3/world/episode_compose/episode_000003; TW5=$TS/ep5/world/episode_compose/episode_000005
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
