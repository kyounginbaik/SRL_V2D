"""Regenerate outputs/COMPOSE_V2D_RESULTS.md from the kit log, the per-object results and compose-sequences/SCREENING.md."""
import json, glob, re, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT
kit = open(ROOT / 'logs/kit_public_best_final.log').read()
overall = re.findall(r'^  (AUC|SP-SR|MP-SR|RPE|MPPE)\s+([\d.]+)', kit, re.M)
per_ep = re.findall(r'^  ep\s*(\d+) (\S+)\s+AUC ([\d.]+)\s+SP-SR ([\d.]+)\s+MP-SR ([\d.]+)\s+RPE ([\d.]+)\s+MPPE ([\d.]+)', kit, re.M)
picks = re.findall(r'^episode (\d+) (\w+): (\S+) \(in-script AUC ([\d.]+)\)', kit, re.M)
scr = open(ROOT / 'compose-sequences/SCREENING.md').read().splitlines()
rows = [l for l in scr if l.startswith('| ') and not l.startswith('| episode')]
ok_eval = [l for l in rows if '| evaluation |' in l and '| ok |' in l]; rej_eval = [l for l in rows if '| evaluation |' in l and ('| rejected |' in l or '| unverified |' in l)]
ok_pub = [l for l in rows if '| public |' in l and '| ok |' in l]
L = ['# ComPose + stereo on V2D Track 3 - results (updated 2026-10-12)', '',
     'All numbers GT-free (team meshes, stereo-implied metric scale, our own ego-motion). Public scores from the NVIDIA kit scorer',
     '(`compose_v2d/kit_score.py --runs public_best`: per object the tag/variant with the best in-script ADD-AUC, episodes scored jointly).', '',
     '## 1. What changed since the last report', '',
     '- **Lid-on pot (ep0/ep1/ep2)**: at frame 0 the lid closes the pot, the Sim(3) anchor cannot tell upright from upside-down and the',
     '  prep rule kept the lid points inside the pot cloud. Fix: `--anchor_free_of white_pot_lid` anchors on the first lid-free frame and',
     '  tracks backwards to frame 0 (`_v9lid`); prep now subtracts the lid whenever >= 15 % of the pot mask survives. ep0 pot AUC 0.42 -> 0.95',
     '  (kit episode AUC 0.21 -> 0.93); ep1 (eval) depth-screen bad-frame ratio 0.61 -> 0.00. ep2 is NOT fixed (see open issues).',
     '- **`_v7` regressions (ep26 pot 0.88 -> 0.41, ep18 dust pan 0.96 -> 0.57)**: the hard 25-deg jump gate blocked real fast rotations',
     '  (ablation: removing grasp_sym did not help, removing the gate did). `_v9` = soft gate (hold the previous rotation only while a',
     '  translation-only refit still fits the points, else accept the jump) + strict grasp_sym: ep26 0.87, ep18 0.97, ep42/43 0.93 kept,',
     '  ep11 cup 0.88. `_v9` is the default; it has been run on every evaluation object (public re-pass partly done).',
     '- **GT-free screening** (`compose_v2d/screen_depth.py`, calibrated on public): the tracked mesh is rendered into the stereo depth',
     '  (flipped / misplaced poses show "deep" pixels) and its silhouette spill outside mask+hand is measured (pot handle slips).',
     '  Public check: every run with AUC >= 0.8 passes except mini_sweeper (team mesh never fits the depth) ; flips, misplacements and',
     '  all handle slips fail. Cups that fail only the in-script ADD (axial rotation) pass, as they should for the kit (symmetric).',
     '- **Export now screened**: `compose-sequences/{public,evaluation}` hold only sequences that pass (public: AUC >= 0.8; evaluation:',
     '  depth/spill screen). Everything else is in `compose-sequences/_rejected/` or `_unverified/` with the reason in README.md and',
     '  `compose-sequences/SCREENING.md`.', '',
     '## 2. Kit scores, public split (per-object best, 20 episodes)', '',
     '| ' + ' | '.join(k for k, _ in overall) + ' |', '|' + '---|' * len(overall), '| ' + ' | '.join(v for _, v in overall) + ' |', '',
     '| episode | objects | AUC | SP-SR | MP-SR | RPE (cm) | MPPE (cm) |', '|---|---|---|---|---|---|---|']
L += [f'| {e} | {o} | {a} | {s} | {m} | {r} | {p} |' for e, o, a, s, m, r, p in per_ep]
L += ['', 'Per-object choice (tag/variant, in-script ADD-AUC):', '']
L += [f'- ep{e} {o}: {t} ({a})' for e, o, t, a in picks]
L += ['', '## 3. compose-sequences', '', f'- public: {len(ok_pub)} object sequences exported (AUC >= 0.8).', f'- evaluation: {len(ok_eval)} object sequences passed the GT-free screen:', '']
L += ['  ' + re.sub(r'\s*\|\s*', ' | ', l).strip(' |')[:150] for l in ok_eval]
L += ['', f'- evaluation rejected / unverified ({len(rej_eval)}):', '']
L += ['  ' + re.sub(r'\s*\|\s*', ' | ', l).strip(' |')[:220] for l in rej_eval]
L += ['', '## 4. Open issues', '',
      '- **ep2 (public, pot+lid)**: still AUC 0.2. Our overlay follows the pot in the image for the whole episode, but the GT (aligned at',
      '  frame 0) drifts away by ~19 cm while the pot is lifted (frames 84-252) and comes back when it is set down; the lid carries a',
      '  constant ~29 cm offset. Depth meta/scale and VO look like ep0. Suspect: GT/ego timing or mocap for this episode; not pursued further.',
      '- **water_pitcher**: team mesh wrong shape (CD-O 0.65) -> AUC ~0.2 on ep21/22, both eval pitchers rejected. Needs re-reconstruction.',
      '- **wooden pieces**: piece_1 mesh slot on the wrong side -> 180-deg twin, AUC 0.05-0.4 (ep23/24); ep25 unverified.',
      '- **mini_sweeper**: team mesh never matches the depth (median 18 mm) so the screen cannot verify it; ep19/30 rejected, ep31 0.19.',
      '- **wooden_spoon**: no team mesh -> 10 evaluation episodes untracked.',
      '- **cups in the dish rack / poured**: ep12 cup 0.29, ep20/32/33 eval cups fail the screen (flip when the opening faces the camera).',
      '- `compose_meshicp` (ComPose blend) is kept only where it agrees with the mesh-ICP track (p90 < 15 deg); where they disagree the',
      '  blend is the one that failed on every public case, so the exporter falls back to `icp_track_6dof`.', '',
      '## 5. Reproduce', '',
      '```', 'envs/v2d/bin/python compose_v2d/run_all.py --gpu 0 --name x --episodes 3 --only track --tag _v9 \\',
      '   --track_args "--icp_jump_gate_deg 25 --soft_jump_gate --adaptive_trim --scale_prior --grasp_sym --grasp_strict"',
      '# pot + lid episodes: add --anchor_free_of white_pot_lid (tag _v9lid)',
      'envs/v2d/bin/python compose_v2d/screen_depth.py --episodes 3 --out outputs/screen_depth2.json',
      'envs/v2d/bin/python compose_v2d/export_sequences.py', 'envs/v2d/bin/python compose_v2d/kit_score.py --runs public_best --per_episode', '```']
open(ROOT / 'outputs/COMPOSE_V2D_RESULTS.md', 'w').write('\n'.join(L) + '\n'); print(f'wrote doc: {len(per_ep)} kit episodes, {len(picks)} picks, {len(ok_eval)} ok eval, {len(rej_eval)} rejected/unverified eval')
