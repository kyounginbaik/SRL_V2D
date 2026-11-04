"""Mini round for episode 6 only: gentle gains, slower ramps, two end modes."""
import itertools, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'code'))
from fit_carry_tilt import measure
OUT = ROOT / 'refinement_20261020_pot/carry_tilt_grid_r3_ep6'; OUT.mkdir(parents=True, exist_ok=True)
e = 6; teacher = ROOT / f'refinement_20261020_pot/mass_grid/episode_{e:06d}/mass_200g/result.json'
meas = measure(e, teacher); meas['pot_hand_release_start_s'] = 10.30000000000011; res = json.load(open(teacher)); tasks = []
for rg, ramp, em, sd in itertools.product((0.4, 0.5, 0.6), (1.0, 1.5), ('release_start', 'setdown'), (0.6,)):
    hold_end = {'setdown': meas['setdown_s'] - ramp, 'release_start': meas['pot_hand_release_start_s']}[em]
    name = f'rg{rg:.2f}_pg0.0_pf1.0_sd{sd:.1f}_rp{ramp:.1f}_{em}'
    params = dict(meas, rot_gain=rg, pos_gain=0.0, pivot_fraction=1.0, end_mode=em, ramp_s=ramp, start_s=meas['lift_start_s'] + sd, hold_end_s=hold_end)
    pdir = OUT / f'episode_{e:06d}' / name; pdir.mkdir(parents=True, exist_ok=True); json.dump(params, open(pdir / 'params.json', 'w'), indent=1)
    tasks.append(dict(episode=e, episode_dir=res['input_dir'], scene=res['execution']['scene'], checkpoint=res['policy_checkpoint'], params=str(pdir / 'params.json'), output=str(pdir / 'evaluation')))
json.dump(tasks, open(OUT / 'tasks.json', 'w'), indent=1)
sb = (ROOT / 'refinement_20261020_pot/carry_tilt_grid/run_grid.sbatch').read_text().replace('carry_tilt_grid/tasks.json', 'carry_tilt_grid_r3_ep6/tasks.json').replace('--job-name=carry-tilt', '--job-name=carry-tilt-r3').replace('carry-tilt-%A', 'carry-tilt-r3-%A')
sb = re.sub(r'--array=0-\d+%16', f'--array=0-{len(tasks)-1}%12', sb); (OUT / 'run_grid.sbatch').write_text(sb)
print(len(tasks), 'tasks'); print(subprocess.run(['sbatch', str(OUT / 'run_grid.sbatch')], capture_output=True, text=True).stdout)
