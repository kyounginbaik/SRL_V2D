# H13a: v27 component actors (the staged checkpoints, sha-checked) and component mesh dirs; COMP = v27 component dirs
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; H=$C22/streamC30_compose/h13a; PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
A20=$C22/streamF9_tight/runs/ep20/save_b4/actor;   COMP20=$C22/streamF9_tight/staging/ep20_b4/components/episode_000020
A5=$C22/streamC9_tight/runs/ep5/save_g0/actor;     COMP5=$C22/streamC9_tight/staging/ep5_g0/components/episode_000005
A35=$C22/streamC9_tight/runs/ep35/save_g0/actor;   COMP35=$C22/streamC9_tight/staging/ep35_g0/components/episode_000035
A37=$C22/streamK8_35_36_37/runs/i37_a_save/actor;  COMP37=$C22/streamK8_35_36_37/staging_K8_37b/components/episode_000037
A32=$C22/streamH12_ep32/runs/save_e1f0/actor;      COMP32=$C22/streamH12_ep32/staging/ep32_e1f0/components/episode_000032
A38=$C22/streamF9_tight/runs/ep38/save_b3/actor;   COMP38=$C22/streamF9_tight/staging/ep38_b3/components/episode_000038
A28=$C22/streamF9_tight/runs/ep28/save_b1/actor;   COMP28=$C22/streamF9_tight/staging/ep28_b1/components/episode_000028
A36=$C22/streamC9_tight/runs/ep36/save_f0/actor;   COMP36=$C22/streamC9_tight/staging/ep36_f0/components/episode_000036
A33=$C22/streamC9_tight/runs/ep33/save_d0/actor;   COMP33=$C22/streamC9_tight/staging/ep33_d0/components/episode_000033
A6=$C22/streamD9_tight/runs/ep6/save_b2/actor;     COMP6=$C22/streamD9_tight/staging/ep6_b2/components/episode_000006
for e in 20 5 35 37 32 38 28 36 33 6; do eval "M$e=\$COMP$e/meshes/\$(printf episode_%06d $e)"; done
