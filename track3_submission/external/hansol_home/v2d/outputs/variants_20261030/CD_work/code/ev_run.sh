W=/home/hansol/v2d/outputs/variants_20261030/CD_work; PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
EV=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13/code/ev.py
ROOT=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000020
cd /tmp
OMP_NUM_THREADS=2 $PY -B $EV $ROOT $W/ev/ep20_v8.npz $W/ev/ep20_v8.json > $W/ev/ep20_v8.log 2>&1 &
OMP_NUM_THREADS=2 $PY -B $EV $ROOT $W/ev/ep20_C.npz $W/ev/ep20_C.json > $W/ev/ep20_C.log 2>&1 &
wait; echo EVDONE
