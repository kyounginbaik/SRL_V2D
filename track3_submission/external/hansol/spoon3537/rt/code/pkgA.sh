#!/bin/bash
# package A = combined final + ep37 s1a rows + ep36 s1d rows; waits for the s1d chain, then runs s3pkg in Slurm
R=/data1/hansol/spoon3537/rt; C=$R/code
until [ -z "$(squeue -h -u hansol -n S3-sv-ep36-s1d,S3-post-ep36-s1d)" ]; do sleep 20; done
B37=$R/packages/track3_FINALc_ep37s1a_tgt_s1/all_20_episodes/track3_submission.parquet
rm -rf $R/packages/track3_FINALc_A_ep36s1d_ep37s1a
J=$(sbatch --parsable -p debug --exclude=node200 -c 2 --mem=16G --time=00:20:00 -J S3-pkgA -o $R/logs/%x-%j.log --wrap "cd $R/tmp && PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages /home/hansol/v2d/.venvs/v2d-host/bin/python -B $C/s3pkg.py 36 S3_ep36_s1d track3_FINALc_A_ep36s1d_ep37s1a $B37")
until [ -z "$(squeue -h -j $J)" ]; do sleep 10; done
echo "=== s1d"; grep -E "BUILD|official_eval_scorer|Error|Trace" $(ls -t $R/logs/S3-sv-ep36-s1d-*.log | head -1) | cut -c1-200
grep -E "SCORE|VERIFY|max \|pos|SHA|Error|Trace" $(ls -t $R/logs/S3-post-ep36-s1d-*.log | head -1) | cut -c1-260
echo "=== package A"; grep -v Warn $R/logs/S3-pkgA-$J.log | tail -8
python3 - <<'P'
import pandas as pd
R='/data1/hansol/spoon3537/rt/packages/'
a=pd.read_parquet(R+'track3_FINALc_A_ep36s1d_ep37s1a/all_20_episodes/track3_submission.parquet').set_index('row_id')
b=pd.read_parquet('/data1/hansol/teacher33/packages/track3_FINAL_ep32T32cb1_ep33R6M2_20261105/all_20_episodes/track3_submission.parquet').set_index('row_id')
d=(a!=b)&~(a.isna()&b.isna()); r=d.any(axis=1); print('A vs combined final: episodes changed', sorted(set(i.split('/')[1] for i in a.index[r])), 'rows', int(r.sum()))
P
