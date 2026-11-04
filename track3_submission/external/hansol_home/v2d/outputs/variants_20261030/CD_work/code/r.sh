V=/home/hansol/v2d/outputs/variants_20261030
W=$V/CD_work; PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
RR=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026
mkdir -p $W/shift
for s in 0 0.5 0.9 1.4 1.85 2.4; do $PY $W/code/mkshift.py $RR/selected20_final_v8_g10b 20 $s $W/shift/ep20_s$s.npz; done
cat > $W/code/shift_run.sh <<'EOF'
W=/home/hansol/v2d/outputs/variants_20261030/CD_work; PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
EV=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13/code/ev.py
ROOT=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000020
cd /tmp
for s in 0 0.5 0.9 1.4 1.85 2.4; do
  OMP_NUM_THREADS=2 $PY -B $EV $ROOT $W/shift/ep20_s$s.npz $W/shift/ep20_s$s.json > $W/shift/ep20_s$s.log 2>&1 &
done
wait; echo SHIFTDONE
EOF
nohup bash $W/code/shift_run.sh > $W/shift/run.log 2>&1 &
uptime; nproc; cat $W/logs_rescore.txt
