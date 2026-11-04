CAN=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical
ls $CAN/public/episode_000021/exports/ | head -40
head -c 1200 $CAN/public/episode_000021/exports/r8_base/public_score.json; echo
/home/hansol/v2d/.venvs/v2d-host/bin/python /home/hansol/v2d/outputs/variants_20261030/CD_work/code/pubtab.py r8_base g15_camrt
