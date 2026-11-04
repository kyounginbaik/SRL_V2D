"""Fit carry-correction parameters from a previous physical rollout and write a parameter grid + sbatch.

Measures, over the reference lift window: mean world rotation taking the reference pot orientation to
the achieved one (to be inverted), mean reference-minus-achieved pot position, the pot centre in the
carrying hand's frame (pivot), and the end of hand/pot contact.
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd, mujoco
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
GRID = dict(rot_gain=(0.7, 1.0), pos_gain=(0.0, 1.0), pivot_fraction=(0.5, 1.0), end_mode=('setdown', 'release'))


def measure(episode, teacher_result, side='left'):
    res = json.load(open(teacher_result)); tr = np.load(res['execution']['trajectory'])
    m = mujoco.MjModel.from_xml_path(res['execution']['scene']); d = mujoco.MjData(m)
    names = tr['actuator_names'].tolist()
    pos_ids = [names.index(f'robot_{side}_{side}_pos_{k}_position') for k in 'xyz']
    rot_ids = [names.index(f'robot_{side}_{side}_rot_{k}_position') for k in 'xyz']
    qpos_pos = m.jnt_qposadr[m.actuator_trnid[pos_ids, 0]]; qpos_rot = m.jnt_qposadr[m.actuator_trnid[rot_ids, 0]]
    sub = pd.read_parquet(ROOT / f'refinement_20261018/shared_policy_handoff/episode_{episode:06d}/results/episode_{episode:06d}.parquet')
    p = sub[sub.object_body_name == 'white_pot'].sort_values('step')
    t = p.time_s.to_numpy(); assert np.allclose(t, tr['timestamps_seconds'], atol=1e-6)
    pr = p[['reference_position_x', 'reference_position_y', 'reference_position_z']].to_numpy()
    pa = p[['achieved_position_x', 'achieved_position_y', 'achieved_position_z']].to_numpy()
    qr = R.from_quat(p[['reference_quaternion_x', 'reference_quaternion_y', 'reference_quaternion_z', 'reference_quaternion_w']].to_numpy())
    qa = R.from_quat(p[['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy())
    lifted = np.where(pr[:, 2] > pr[0, 2] + 0.01)[0]; w = lifted
    rotvec = (qa[w] * qr[w].inv()).as_rotvec().mean(0)          # world: reference -> achieved
    dpos = (pr[w] - pa[w]).mean(0)                                # world: achieved -> reference
    i = list(tr['object_names']).index('white_pot'); pot_sim = tr['object_positions'][:, i]
    Q = tr['qpos']; hand_p = Q[:, qpos_pos]; hand_R = R.from_euler('XYZ', Q[:, qpos_rot])
    pivot = hand_R[w].inv().apply(pot_sim[w] - hand_p[w]).mean(0)
    pot = m.body('white_pot').id; robot_side = {b for b in range(m.nbody) if m.body(b).name.startswith(f'robot_{side}')}
    last_contact = None
    for k, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        for ci in range(d.ncon):
            c = d.contact[ci]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if {b1, b2} & {pot} and ({b1, b2} & robot_side): last_contact = k
    return dict(episode=episode, side=side, teacher_result=str(teacher_result), pos_actuator_ids=pos_ids, rot_actuator_ids=rot_ids,
                rotation_vector=(-rotvec).tolist(), rotation_deg=float(np.degrees(np.linalg.norm(rotvec))), translation_m=dpos.tolist(),
                pivot_hand_frame_m=pivot.tolist(), lift_start_s=float(t[w[0]]), setdown_s=float(t[w[-1]]),
                last_contact_s=float(t[last_contact]), frames_in_window=int(len(w)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--episodes', type=int, nargs='+', default=[3, 5, 6, 8])
    ap.add_argument('--teacher-dir', default=str(ROOT / 'refinement_20261020_pot/mass_grid'))
    ap.add_argument('--out', default=str(ROOT / 'refinement_20261020_pot/carry_tilt_grid')); ap.add_argument('--ramp', type=float, default=0.6)
    ap.add_argument('--start-delay', type=float, default=0.3); ap.add_argument('--submit', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tasks = []
    for e in a.episodes:
        teacher = Path(a.teacher_dir) / f'episode_{e:06d}/mass_200g/result.json'
        meas = measure(e, teacher); json.dump(meas, open(out / f'episode_{e:06d}_measurement.json', 'w'), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in meas.items() if k not in ('teacher_result',)}))
        res = json.load(open(teacher))
        for rg, pg, pf, em in itertools.product(*GRID.values()):
            name = f'rg{rg:.2f}_pg{pg:.1f}_pf{pf:.1f}_{em}'
            hold_end = meas['setdown_s'] - a.ramp if em == 'setdown' else meas['last_contact_s']
            params = dict(meas, rot_gain=rg, pos_gain=pg, pivot_fraction=pf, end_mode=em, ramp_s=a.ramp,
                          start_s=meas['lift_start_s'] + a.start_delay, hold_end_s=hold_end)
            pdir = out / f'episode_{e:06d}' / name; pdir.mkdir(parents=True, exist_ok=True)
            json.dump(params, open(pdir / 'params.json', 'w'), indent=1)
            tasks.append(dict(episode=e, episode_dir=res['input_dir'], scene=res['execution']['scene'], checkpoint=res['policy_checkpoint'],
                              params=str(pdir / 'params.json'), output=str(pdir / 'evaluation')))
    json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
    sb = f"""#!/bin/bash
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --array=0-{len(tasks)-1}%16
#SBATCH --job-name=carry-tilt
#SBATCH --output={ROOT}/refinement_20261020_pot/logs/carry-tilt-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
root={ROOT}
eval "$("$root/.venv/bin/python" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys,shlex
t=json.load(open("{out}/tasks.json"))[int(sys.argv[1])]
for k,v in t.items(): print(f"{{k}}={{shlex.quote(str(v))}}")
PY
)"
echo "episode=$episode params=$params"
"$root/.venv/bin/python" "$root/code/evaluate_carry_tilt_variant.py" --episode "$episode" --episode-dir "$episode_dir" --scene "$scene" --checkpoint "$checkpoint" --output "$output" --params "$params"
"""
    (out / 'run_grid.sbatch').write_text(sb); print(len(tasks), 'tasks ->', out / 'run_grid.sbatch')
    if a.submit:
        import subprocess; print(subprocess.run(['sbatch', str(out / 'run_grid.sbatch')], capture_output=True, text=True).stdout)


if __name__ == '__main__':
    main()
