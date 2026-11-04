"""ALL20-K candidate gate: for a build-v2 candidate TAG of episode EP (common_v2_candidates/TAG), build its donor package (a20splice:
v29 with only EP rows rebuilt), score EP with the unchanged kit scorers (AUC/MPPE/RPE/MP-SR, our tgt_v32 as stand-in truth, the
candidate's own = v29 meshes) and compare with v29's own local score of EP. Verdict ACCEPT only if: build-v2 exact, meshes == v29,
every object <= 1.5 cm / 15 deg vs tgt_v32 (MP-SR safety margin), AUC not lower by > 0.002, MPPE lower by >= 0.02 cm, RPE not higher by > 0.03 cm.
usage: a20select.py EP TAG -> out/select/epEP_TAG.json"""
import sys, json, subprocess, pathlib
EP, TAG = int(sys.argv[1]), sys.argv[2]
C22 = pathlib.Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'); S = C22 / 'streamALL20K'
CAND = pathlib.Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates') / TAG / f'episode_{EP:06d}'
DON = pathlib.Path('/data1/SRL_V2D/challenge_submissions_SRL/a20_donors_20261102') / TAG / 'all_20_episodes'
PY = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python'
comp = json.loads((CAND / 'completed.json').read_text()); exact = comp.get('exact_previous_physical_states_actions')
if not DON.exists():
    subprocess.run([PY, '-B', str(S / 'code/a20splice.py'), str(EP), TAG, str(DON), TAG], check=True, cwd='/tmp')
od = S / 'out/select'; od.mkdir(parents=True, exist_ok=True); sj = od / f'ep{EP}_{TAG}_score.json'
if not sj.exists():
    subprocess.run([PY, '-B', str(S / 'code/score_all.py'), str(DON / 'track3_submission.parquet'), str(DON / 'results'), str(S / 'targets/tgt_v32'),
                    str(sj), str(EP)], check=True, cwd='/tmp', capture_output=True)
new = json.loads(sj.read_text())[str(EP)]; bp = S / f'out/score_base/ep{EP}.json'   # ep33/35: the fixed-recon packages are the baseline
old = json.loads((bp if bp.exists() else S / f'out/score_v29/ep{EP}.json').read_text())[str(EP)]
ns, os_ = new['scores'], old['scores']
safe = all(p <= 1.5 and d <= 15 for p, d in new['objects'].values())
gain = dict(dAUC=ns['AUC'] - os_['AUC'], dMPPE=ns['MPPE'] - os_['MPPE'], dRPE=ns['RPE'] - os_['RPE'])
ok = bool(exact) and safe and ns['MP-SR'] == 1.0 and gain['dAUC'] >= -0.002 and gain['dMPPE'] <= -0.02 and gain['dRPE'] <= 0.03
res = dict(ep=EP, tag=TAG, verdict='ACCEPT' if ok else 'REJECT', exact=exact, safe_1p5_15=safe, new=new, v29=old, gain=gain, donor=str(DON))
(od / f'ep{EP}_{TAG}.json').write_text(json.dumps(res, indent=1))
print(f"ep{EP} {TAG} {res['verdict']} exact={exact} safe={safe} objs {new['objects']} (v29 {old['objects']}) "
      f"AUC {os_['AUC']:.4f}->{ns['AUC']:.4f} MPPE {os_['MPPE']:.3f}->{ns['MPPE']:.3f} RPE {os_['RPE']:.3f}->{ns['RPE']:.3f}")
