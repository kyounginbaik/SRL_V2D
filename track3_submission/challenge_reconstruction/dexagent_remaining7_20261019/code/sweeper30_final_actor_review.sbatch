#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=sweeper30-finalreview
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-finalreview-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u - <<'PY'
from pathlib import Path
import json,subprocess,sys
import numpy as np,mujoco
b=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019');work=b/'physics_collision_v2'
actor=work/'sweeper_transfer/liftclear_0.06_s2.5_v1/release_neutral/trial1/actor_grid/completed.json'
selection=json.loads(actor.read_text());report=json.loads((Path(selection['evaluation'])/'result.json').read_text())
m=mujoco.MjModel.from_xml_path(report['execution']['scene']);z=np.load(report['execution']['trajectory']);samples=[]
for seconds in [0.,2.5,3.,4.,6.,8.,10.,11.8,12.3,12.8,13.1]:
 k=min(round(seconds*float(z['fps'])),len(z['qpos'])-1);d=mujoco.MjData(m)
 d.qpos[:]=z['qpos'][k];d.qvel[:]=z['qvel'][k];d.ctrl[:]=z['ctrl'][k];mujoco.mj_forward(m,d);contacts=[]
 for i,c in enumerate(d.contact):
  names=[m.body(int(m.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
  if not any(n.startswith('robot_') for n in names) or not any(n in ('mini_sweeper','plastic_dish_rack') for n in names):continue
  force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
  contacts.append(dict(bodies=names,force_n=float(force[0]),penetration_mm=max(0.,-float(c.dist)*1000)))
 samples.append(dict(seconds=seconds,contacts=contacts))
audit=dict(actor_report=str(actor),evaluation=selection['evaluation'],samples=samples,
 final_right_object_contact_count=sum(any(n.startswith('robot_right_') for n in c['bodies']) for c in samples[-1]['contacts']))
(b/'review/episode30_final_actor_contacts.json').write_text(json.dumps(audit,indent=2)+'\n')
subprocess.run([sys.executable,str(b/'code/render_selected.py'),'--episode','30','--episode-dir',selection['episode_dir'],
 '--baseline',str(work/'runs/baseline_v1/episode_000030/baseline'),'--refined',selection['evaluation'],
 '--output',str(work/'review/episode_000030/final_release_actor'),'--stride','1'],check=True)
PY
