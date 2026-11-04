"""ALL20-K orchestrator (light, login node, nohup): every 10 min
 1. gate every finished A20 build-v2 candidate (a20select.py: donor package + kit-faithful AUC/MPPE/RPE vs tgt_v32 vs baseline);
 2. per episode pick the ACCEPTed candidate with the lowest MPPE (ties: higher AUC);
 3. when the choice set changes, assemble the next v32 package (v32build.py on the cwmesh base; ep35/ep33 default to the fixed
    K35_H2joint_b1 / K33_H7ball_b5 rows), write its README provenance, queue a full local score, dry-run the upload, log ALL20-K lines.
Never uploads. Stops when no A20 jobs remain and everything is gated. usage: a20orch.py"""
import json, re, subprocess, time, pathlib, datetime
C22 = pathlib.Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'); S = C22 / 'streamALL20K'
R = pathlib.Path('/data1/SRL_V2D/challenge_submissions_SRL'); CAND = pathlib.Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates')
PY = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python'
FIX = {35: R / 'ep35_fix_20261102/K35_H2joint_b1/all_20_episodes', 33: R / 'ep33_fix_20261102/K33_H7ball_b5/all_20_episodes'}
STATE = S / 'out/orch_state.json'
def now(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M')
def status(line):
    with open(C22 / 'STATUS.md', 'a') as f: f.write(f'- {now()} UTC ALL20-K {line}\n')
def log(msg):
    with open(S / 'out/orch.log', 'a') as f: f.write(f'{now()} {msg}\n')
st = json.loads(STATE.read_text()) if STATE.exists() else dict(builds=[], chosen={}, gated=[])
while True:
    jobs = [l.split() for l in (S / 'out/jobs.txt').read_text().splitlines() if l.strip()]
    runs = sorted({(int(j[1][2:]), j[2]) for j in jobs})
    for ep, name in runs:
        tag = f'A20_ep{ep}_{name}'; key = f'{ep}:{name}'
        if key in st['gated']: continue
        if not (CAND / tag / f'episode_{ep:06d}/completed.json').exists(): continue
        p = subprocess.run([PY, '-B', str(S / 'code/a20select.py'), str(ep), tag], cwd='/tmp', capture_output=True, text=True)
        line = (p.stdout.strip().splitlines() or [p.stderr.strip()[-300:]])[-1]; log(line); st['gated'].append(key)
        status(f'gate {line}')
    acc = {}
    for f in (S / 'out/select').glob('ep*_A20_*.json'):
        if f.name.endswith('_score.json'): continue
        r = json.loads(f.read_text())
        if r['verdict'] != 'ACCEPT': continue
        cur = acc.get(r['ep'])
        if cur is None or (r['new']['scores']['MPPE'], -r['new']['scores']['AUC']) < (cur['new']['scores']['MPPE'], -cur['new']['scores']['AUC']): acc[r['ep']] = r
    chosen = {str(e): r['tag'] for e, r in sorted(acc.items())}
    if chosen and chosen != st['chosen']:
        letter = 'bcdefghijklmnopqrstuvwxyz'[len(st['builds'])]; lab = f'v32{letter}'
        spec = {e: str(FIX[e]) for e in FIX}; spec.update({int(e): acc[int(e)]['donor'] for e in chosen})
        sp = S / f'out/spec_{lab}.json'; sp.write_text(json.dumps({str(k): v for k, v in spec.items()}, indent=1))
        pkg = R / f'track3_{lab}_a20_20261102/all_20_episodes'
        p = subprocess.run([PY, '-B', str(S / 'code/v32build.py'), str(sp), str(pkg), lab], cwd='/tmp', capture_output=True, text=True)
        log(p.stdout + p.stderr[-500:])
        if p.returncode == 0:
            rows = []
            for e in sorted(set(int(k) for k in spec)):
                if str(e) in chosen:
                    r = acc[e]; o, n = r['v29']['scores'], r['new']['scores']
                    rows.append(f"| {e} | {r['tag']} (ALL20-K tightened build-v2 rollout) | {r['v29']['objects']} -> {r['new']['objects']} | "
                                f"AUC {o['AUC']:.4f}->{n['AUC']:.4f}, MPPE {o['MPPE']:.3f}->{n['MPPE']:.3f}, RPE {o['RPE']:.3f}->{n['RPE']:.3f} |")
                else: rows.append(f'| {e} | {spec[e]} (fixed-recon component, unchanged) | | |')
            (pkg / 'README.md').write_text(
                f'# SRL Track 3 {lab} (ALL20-K, {now()} UTC) - NOT uploaded\n\nBase: `track3_v30c1_cwmesh_nogt_20261101` (= v29 build-v2 rollouts + 17 GT-free coworker meshes aligned to our v29 meshes; '
                'official v30c1_cwmesh_nogt AUC 0.87701 / MPPE 1.740 / RPE 1.580 / CD-O 0.26116 / MP 0.85).\nAll other episodes keep the v29 rollout + cwmesh/v29 mesh.\n'
                'Replaced episodes (pose + checkpoint columns from the donor; mesh payload kept from base when the donor mesh == v29 mesh):\n\n'
                '| ep | source | objects cm/deg vs our target (v29/base -> new) | local kit metrics vs our target |\n|---|---|---|---|\n' + '\n'.join(rows) +
                '\n\nOur target = selected20_final_v14_g17 (+ K35_H2joint ep35, K33_H7ball ep33). No public/GT mesh or pose used. Local scores are a proxy, not official.\n'
                '\nUpload (all 6 boards), from submission_kit/:\n\n    python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m "SRL ' + lab + '" --wait\n')
            sc = S / f'out/score_{lab}'
            subprocess.run(['sbatch', '-J', f'A20-score-{lab}', str(S / 'code/score_all.sbatch'), str(pkg), str(S / 'targets/tgt_v32'), str(sc)], capture_output=True)
            d = subprocess.run(f'cd {pkg}/submission_kit && {R}/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m dry --dry-run',
                               shell=True, capture_output=True, text=True)
            ok = 'looks valid' in d.stdout and 'nothing submitted' in d.stdout
            sha = json.loads((pkg / 'V32_BUILD.json').read_text())['sha']
            st['builds'].append(dict(label=lab, pkg=str(pkg), sha=sha, chosen=chosen, dry_run_ok=ok, at=now()))
            status(f'{lab} built {pkg} sha {sha[:8]} dry-run {"OK" if ok else "FAILED"}; tightened eps: ' +
                   ', '.join(f"{e}:{t}" for e, t in chosen.items()) + f' (full local score -> streamALL20K/out/score_{lab})')
        st['chosen'] = chosen
    STATE.write_text(json.dumps(st, indent=1))
    q = subprocess.run(['squeue', '-u', 'exaFLOPs', '-h', '-o', '%j'], capture_output=True, text=True).stdout
    pending = [k for k in [f'{e}:{n}' for e, n in runs] if k not in st['gated']]
    if 'A20-' not in q: log(f'done (ungated: {pending})'); break
    time.sleep(600)
