"""Round-2 carry-correction grid with per-episode settings (reuses fit_carry_tilt.measure)."""
import itertools, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'code'))
from fit_carry_tilt import measure
OUT = ROOT / 'refinement_20261020_pot/carry_tilt_grid_r2'; OUT.mkdir(parents=True, exist_ok=True)
# per-episode: rot_gain, pos_gain, pivot_fraction, end_mode, start_delay, ramp
SETTINGS = {
    3: dict(rot_gain=(1.0,), pos_gain=(0.0, 0.3), pivot_fraction=(1.0, 1.2), end_mode=('setdown',), start_delay=(0.3,), ramp=(0.6,)),
    5: dict(rot_gain=(1.0, 1.3), pos_gain=(0.5, 1.0), pivot_fraction=(1.0, 1.3), end_mode=('setdown',), start_delay=(0.3,), ramp=(0.6,)),
    6: dict(rot_gain=(0.3, 0.5, 0.7), pos_gain=(0.0, 0.5), pivot_fraction=(1.0,), end_mode=('release_start',), start_delay=(0.6, 1.2), ramp=(1.0,)),
    8: dict(rot_gain=(0.85, 1.0), pos_gain=(0.0, 0.5), pivot_fraction=(1.0, 1.2), end_mode=('setdown',), start_delay=(0.3,), ramp=(0.6,)),
}
tasks = []
for e, S in SETTINGS.items():
    teacher = ROOT / f'refinement_20261020_pot/mass_grid/episode_{e:06d}/mass_200g/result.json'
    meas = measure(e, teacher); res = json.load(open(teacher))
    # release primitive start of the pot (left) hand, if the checkpoint has one
    with np.load(res['policy_checkpoint'], allow_pickle=False) as z:
        rel = [float(z[k]) for k in z.files if k.endswith('start_s') and k.startswith(('release', 'release0_', 'release1_'))
               and (k.replace('start_s', 'actuator_ids') in z.files) and set(z[k.replace('start_s', 'actuator_ids')].tolist()) & set(range(28))]
    meas['pot_hand_release_start_s'] = min(rel) if rel else None
    json.dump(meas, open(OUT / f'episode_{e:06d}_measurement.json', 'w'), indent=1)
    for rg, pg, pf, em, sd, ramp in itertools.product(*S.values()):
        if em == 'release_start' and meas['pot_hand_release_start_s'] is None: em = 'release'
        hold_end = {'setdown': meas['setdown_s'] - ramp, 'release': meas['last_contact_s'], 'release_start': meas['pot_hand_release_start_s']}[em]
        name = f'rg{rg:.2f}_pg{pg:.1f}_pf{pf:.1f}_sd{sd:.1f}_rp{ramp:.1f}_{em}'
        params = dict(meas, rot_gain=rg, pos_gain=pg, pivot_fraction=pf, end_mode=em, ramp_s=ramp, start_s=meas['lift_start_s'] + sd, hold_end_s=hold_end)
        pdir = OUT / f'episode_{e:06d}' / name; pdir.mkdir(parents=True, exist_ok=True); json.dump(params, open(pdir / 'params.json', 'w'), indent=1)
        tasks.append(dict(episode=e, episode_dir=res['input_dir'], scene=res['execution']['scene'], checkpoint=res['policy_checkpoint'], params=str(pdir / 'params.json'), output=str(pdir / 'evaluation')))
json.dump(tasks, open(OUT / 'tasks.json', 'w'), indent=1)
sb = (ROOT / 'refinement_20261020_pot/carry_tilt_grid/run_grid.sbatch').read_text().replace('carry_tilt_grid/tasks.json', 'carry_tilt_grid_r2/tasks.json').replace('--job-name=carry-tilt', '--job-name=carry-tilt-r2').replace('carry-tilt-%A', 'carry-tilt-r2-%A')
import re; sb = re.sub(r'--array=0-\d+%16', f'--array=0-{len(tasks)-1}%16', sb)
(OUT / 'run_grid.sbatch').write_text(sb); print(len(tasks), 'tasks'); print({e: SETTINGS[e] for e in SETTINGS})
print(subprocess.run(['sbatch', str(OUT / 'run_grid.sbatch')], capture_output=True, text=True).stdout)
