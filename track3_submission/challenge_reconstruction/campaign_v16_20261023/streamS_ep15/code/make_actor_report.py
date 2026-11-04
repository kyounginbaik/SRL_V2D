"""Write a staging base (actor_report.json = the actor's attempts.json row + training_report + selection; input_manifest.json of
the world) for one distilled actor of a Stream-S world. usage: make_actor_report.py BASE ACTOR_DIR WORLD 'selection text'"""
import json, shutil, sys
from pathlib import Path
base, actor, world, sel = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), sys.argv[4]
if base.exists(): raise SystemExit(f'exists: {base}')
rows = [r for r in json.loads((actor.parent / 'attempts.json').read_text()) if Path(r['checkpoint']).resolve() == actor / 'training/policy.npz']
assert len(rows) == 1, rows
rep = dict(rows[0]); rep.update(training_report=str(actor / 'training/training_report.json'), reference_proxy_mp_sr=1.0 if rep['passed'] else 0.0,
                                hand_source='HaWoR', selection=sel)
assert rep['passed'] and Path(rep['training_report']).exists() and Path(rep['episode_dir']).resolve() == world / 'episodes/episode_000015'
base.mkdir(parents=True)
(base / 'actor_report.json').write_text(json.dumps(rep, indent=2) + '\n'); shutil.copyfile(world / 'input_manifest.json', base / 'input_manifest.json')
print(json.dumps(rep, indent=1))
