"""Stream S: re-target streamE_ep15 pickup-search specs (physics_pickup_search_v2 / rack_safe_pickup) to a Stream-S world.
Anchors, seed reference, open fingers, timing and every search setting are kept byte-for-byte (so streamE progress.json files
can be resumed); only the world-specific paths change (scene, original_reference, geometry, manifest). Optional overrides:
--release T (release time; horizon stays 8.25 s), --suffix NAME.
usage: port_specs.py WORLD OUT_DIR SPEC.json [SPEC.json ...] [--release T --times JSON --suffix S]"""
import json, sys, argparse
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('world'); ap.add_argument('out'); ap.add_argument('specs', nargs='+')
ap.add_argument('--release', type=float); ap.add_argument('--times', help='JSON dict of times overrides (e.g. earlier grasp)'); ap.add_argument('--suffix', default=''); a = ap.parse_args()
W = Path(a.world).resolve(); EP = W / 'episodes/episode_000015'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
man = json.loads((EP / 'input_manifest.json').read_text()); made = []
for sp in a.specs:
    s = json.loads(Path(sp).read_text()); s['ported_from_spec'] = str(Path(sp).resolve())
    s.update(scene=str(W / 'scene/controller_scene.xml'), original_reference=str(EP / 'tracking_reference.npz'),
             geometry=str(EP / 'metric_geometry.json'), manifest=str(EP / 'input_manifest.json'), source=man['source'])
    if a.release is not None:
        s['times'] = dict(s['times'], release=a.release); s['right_hand_holds_until_end'] = a.release >= s['horizon_s']
    if a.times: s['times'] = dict(s['times'], **json.loads(a.times)); s['times_override'] = json.loads(a.times)
    name = Path(sp).stem + a.suffix; p = out / f'{name}.json'
    if p.exists(): raise SystemExit(f'exists: {p}')
    p.write_text(json.dumps(s, indent=1) + '\n'); made.append(str(p))
allp = out / f'specs{a.suffix}.json'; allp.write_text(json.dumps(made, indent=1)); print(allp, len(made))
