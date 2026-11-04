"""Track 3 complete handoff set for the final candidate (FINAL_20261104 + ep32 T32c_b1 + ep33 R6M2): per episode reconstruction,
retargeting (component / actor / beam table / build-v2 candidate) and videos, plus submission files, review report and docs. Copies only."""
import csv, json, shutil, hashlib, zipfile, subprocess
from pathlib import Path
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAME = 'track3_FINAL_complete_set_20261105'; O = Path('/data1/hansol/share') / NAME; ST = Path('/data1/hansol/share/_stage_complete')
CAMP = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'); REV = Path('/data1/hansol/episode_review/out')
T = Path('/data1/hansol/teacher33'); PK = T / 'packages/track3_FINAL_ep32T32cb1_ep33R6M2_20261105'
if O.exists(): shutil.rmtree(O)
O.mkdir(parents=True)


def cp(a, b):
    a, b = Path(a), Path(b)
    if a.is_dir():
        shutil.copytree(a, b, symlinks=False)
    else:
        b.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(a, b)


for f in ('track3_submission.parquet', 'track3_submission.cd-o.parquet'):
    cp(PK / 'all_20_episodes' / f, O / 'submission' / f)
cp(PK / 'COMBINED_PACKAGE.json', O / 'submission/COMBINED_PACKAGE.json')
for f in ('REPORT_episode_review_20261105.md', 'EP33_AGENT_SPOON_MARKS.md', 'V2D_SUBMISSION_ISSUES_FOR_TEAM.md', 'V2D_SUBMISSION_ISSUES_FOR_TEAM_EN.md', 'FINAL_0fca055b_episodes.tsv'):
    cp(ST / f, O / 'docs' / f)
rows = list(csv.reader(open(ST / 'FINAL_0fca055b_episodes.tsv'), delimiter='\t'))[1:]
index = []
for r in rows:
    ep = int(r[0]); E = O / f'episodes/ep{ep:02d}'; nm = f'episode_{ep:06d}'
    if ep == 32:
        tmp = O / '_ep32'; zipfile.ZipFile(ST / 'share_ep32_T32c_b1_20261104.zip').extractall(tmp); S = tmp / 'share_ep32_T32c_b1_20261104'
        cp(S / 'recon', E / 'recon'); cp(S / 'target', E / 'recon/target')
        for d in ('teacher', 'actor', 'beam_table', 'candidate', 'code', 'notes'):
            cp(S / d, E / 'retarget' / d)
        cp(S / 'videos', E / 'videos'); cp(S / 'README.md', E / 'README_ep32_T32c_b1.md'); shutil.rmtree(tmp)
        index.append(dict(ep=ep, final_source='T32c_b1 (team; replaces FINAL K32_R10)', note='MP-SR passes alone (0.95); recon uses human bowl marks (marks_ep32.json)'))
        continue
    if ep == 33:
        cp('/data1/hansol/ep33_20261102/orient/fit4/recon_E14_best_B2A_S1_HGAZr6/episode_000033', E / 'recon/bundle_r6')
        cp('/data1/hansol/retarget33/targets/R33_HGA_r6', E / 'recon/target_R33_HGA_r6')
        cp(T / 'worlds/R33r6M', E / 'retarget/world_R33r6M_moved_start')
        cp(T / 'teacher/FN_r6M', E / 'retarget/teacher_FN_r6M')
        cp(T / 'runs/distill/DR6Mm1', E / 'retarget/distilled_margins_DR6Mm1')
        for b in ('R6M2_A', 'R6M2_B', 'R6M2_C'):
            cp(T / 'runs/cw' / b, E / f'retarget/beam_{b}')
        cp(T / 'runs/save_R6M2_C_f0', E / 'retarget/saved_actor_R6M2_C_f0')
        cp(T / 'candidates/R33_HGA_r6_R6M2_C_f0', E / 'retarget/candidate_R33_HGA_r6_R6M2_C_f0')
        cp(T / 'out/perturb_R6M2C', E / 'retarget/perturb_R6M2C')
        for lg in (T / 'logs').glob('T33-tabR6M2C-*.log'):
            cp(lg, E / 'retarget' / lg.name)
        cp(T / 'README.md', E / 'retarget/TEACHER33_README.md')
        for v in ('/data1/hansol/retarget33/videos/R6M2/ep33_R6M2_vs_R6M_h264.mp4', '/data1/hansol/retarget33/videos/R6M/ep33_R6M_vs_M1C_h264.mp4'):
            cp(v, E / 'videos' / Path(v).name)
        for v in Path('/data1/hansol/retarget33/videos/R6M2').glob('comparison_*.jpg'):
            cp(v, E / 'videos' / ('R6M2_' + v.name))
        index.append(dict(ep=ep, final_source='R6M2 (ours; replaces FINAL K33_H5r16_c3)', note='MP-SR passes alone (0.95); recon uses human bowl marks (docs/EP33_AGENT_SPOON_MARKS.md); right hand releases the spoon about 0.5 s early'))
        continue
    comp = CAMP / r[6]
    cp(comp, E / 'retarget/component')
    if (comp / 'portable/source' / nm).exists():
        cp(comp / 'portable/source' / nm, E / 'recon/source')
    if (comp / 'meshes').exists():
        cp(comp / 'meshes', E / 'recon/meshes')
    if (REV / f'ep{ep:02d}').exists():
        cp(REV / f'ep{ep:02d}', E / 'review')
    index.append(dict(ep=ep, final_source=f'{r[2]}: {r[3]} ({r[4]})', note='unchanged from FINAL; see review/ and docs/REPORT_episode_review_20261105.md'))
(O / 'INDEX.json').write_text(json.dumps(index, indent=1))
tab = ['| ep | final source | note |', '|---|---|---|'] + [f"| {x['ep']} | {x['final_source']} | {x['note']} |" for x in sorted(index, key=lambda x: x['ep'])]
info = json.loads((PK / 'COMBINED_PACKAGE.json').read_text())
txt = ['# Track 3 final candidate: complete set (2026-11-05, Hansol)', '',
       f"Submission (`submission/`): FINAL_20261104 + ep32 T32c_b1 + ep33 R6M2. sha256 {info['sha']} (CD-O {info['cdo_sha'][:16]}).",
       'Kit offline validation OK; provenance uniform over 20 episodes; the ep32-only and ep33-only packages each scored MP-SR 0.95.', '',
       'Per episode (`episodes/epXX/`):',
       '- `recon/`: the reconstruction the robot was retargeted to (object poses, meshes, target trajectories).',
       '- `retarget/`: for the 18 FINAL episodes, the team component (portable actor, lineage code, original/validation rollouts, results).',
       '  ep32: teacher, actor, beam table, build-v2 candidate, code and notes from the teammate handoff.',
       '  ep33: moved-start world, new teacher, distilled+margins actor, beam tables A/B/C, saved actor, build-v2 candidate, perturbation, 0.5 s table.',
       '- `review/` (18 episodes): 4-panel H.264 video (RGB | recon + SAM2 outline | robot objects + recon outline | MuJoCo rollout) and per-frame / 0.5 s diagnostics. `videos/` for ep32 and ep33.',
       '- `docs/`: episode review report (worst windows, top-5 fixes), ep33 agent-vs-human spoon marks, submission issues (KR/EN), FINAL lineage table.', ''] + tab + ['',
       'Rules: no GT used for reconstruction or retargeting; all compute in Slurm on node01/node100; nothing uploaded from this set.']
(O / 'README.md').write_text('\n'.join(txt) + '\n')
lines = [f'{h(p)}  {p.relative_to(O)}' for p in sorted(O.rglob('*')) if p.is_file()]
(O / 'SHA256SUMS').write_text('\n'.join(lines) + '\n'); print('files', len(lines))
print(subprocess.run(['du', '-sh', str(O)], capture_output=True, text=True).stdout)
Z = O.with_suffix('.zip')
with zipfile.ZipFile(Z, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    for p in sorted(O.rglob('*')):
        if p.is_file():
            z.write(p, f'{NAME}/{p.relative_to(O)}')
print('zip', Z, Z.stat().st_size, h(Z))
