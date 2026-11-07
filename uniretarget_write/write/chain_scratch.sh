#!/usr/bin/env bash
# When a from-scratch Newton grip search finishes, train no-assistance tracking RL from its best reference (2 seeds), then eval/zero/robust.
# Usage: chain_scratch.sh N   (search /oak/scratch/sN)
export SLURM_CONF=/tmp/claude-10449/-home-exaflops/e4f5314f-bd92-42fb-a84f-f4260f6f585b/scratchpad/sl/slurm.conf
N=$1; O=/scratch2/exaFLOPs/srl_yam_run/ablate/oak; cd $O/write
until [ -f $O/scratch/s$N/best.json ]; do /bin/sleep 120; done
SHA=$(python3 -c "import json;print(json.load(open('$O/scratch/s$N/best.json'))['sha256'])")
for S in 42 43; do
  t=$(TAG=_sc$N CURR=write_100m_novoc EXTRA="task.robot_reference_path=/oak/scratch/s$N/best.npz task.robot_reference_sha256=$SHA" sbatch --parsable -J sc${N}x_$S train.sbatch write_wb track $S)
  e=$(sbatch --parsable -d afterok:$t -J sc${N}e_$S eval.sbatch write_wb_sc$N track $S)
  z=$(sbatch --parsable -d afterok:$t -J sc${N}z_$S evalzero.sbatch write_wb_sc$N track $S)
  r=$(sbatch --parsable -d afterok:$t -J sc${N}r_$S robust_rl.sbatch runs/write_wb_sc$N/track-s$S 100007936)
  echo "scratch s$N track-s$S train=$t eval=$e zero=$z robust=$r sha=$SHA" | tee -a research_r8_jobs.txt
done
