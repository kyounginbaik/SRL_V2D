"""Build a browsable report and portable ten-clip evidence bundle."""
from pathlib import Path
import html
import json
import zipfile

import cv2

ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'compatibility.json').read_text())
media={r['id']:r for r in json.loads((ROOT/'media_validation.json').read_text())}
runtime=json.loads((ROOT/'runtime.json').read_text())
results=[]
for row in rows:
    ident=row['id']
    clip=ROOT/'clips'/(ident+'.mp4')
    cap=cv2.VideoCapture(str(clip))
    decoded=0
    while cap.read()[0]:decoded+=1
    advertised=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert decoded==advertised and decoded>0,(ident,decoded,advertised)
    native_path=ROOT/'runs'/ident/'stage1/run.json'
    native=json.loads(native_path.read_text()) if native_path.exists() else None
    results.append(dict(row,media=media[ident],clip_decoded_frames=decoded,
                        native_stage1=native,
                        stage2='blocked_missing_perception_dependencies' if ident.startswith('01_') else 'not_run_stage1_goal_rejected',
                        stage3='not_run',stage4='not_run'))

stage2_path=ROOT/'runs/01_grasp_test_tube/stage2/run.json'
stage2=json.loads(stage2_path.read_text()) if stage2_path.exists() else None
perception_path=ROOT/'runs/01_grasp_test_tube/stage2/perception_attempt.json'
perception=json.loads(perception_path.read_text()) if perception_path.exists() else None
summary={'test_type':'operator_assisted_fixed_library_video_task_compatibility',
         'dexagent_commit':runtime['dexagent_commit'],'egotools_data_revision':runtime['data_revision'],
         'source_repository':'https://github.com/dexagent1/DexAgent',
         'data_repository':'https://huggingface.co/datasets/ropedia-ai/egotools-data',
         'paper':'https://arxiv.org/abs/2609.39378',
         'n_tasks':len(results),'decoded_clips':len(results),
         'schema_pass':sum(r['checks']['scene_semantic_schema_check']['ok'] is True for r in results),
         'consistency_pass':sum(r['checks']['scene_semantic_consistency_check']['ok'] is True for r in results),
         'goal_language_pass':sum(r['checks']['goal_language_coverage_check']['ok'] is True for r in results),
         'verified_robot_rollouts_produced':0,'robot_success_rate':None,
         'scope':'Current Codex session selected clips using published captions, inspected frame contact sheets, authored task semantics, and exercised unmodified native checks and external transport. No separate model API was used. The library was frozen: unsupported goals were retained rather than repaired with new tools. This is not an unaided VLM benchmark or the official EgoTools QA benchmark.',
         'limitations':['Goal-language rejection concerns the exact reviewed wording and the finite compiler; it is not proof that every equivalent task formulation or an extended library must fail.',
                        'Native Stage 1 schema/consistency/goal checks do not establish semantic ground truth or physical task success.',
                        'Released stereo_left_synced streams are 512x512 near 20 FPS. Task clips were encoded as H.264 at 25 FPS by ffmpeg, preserving elapsed-time windows and duplicating frames as necessary. They are not the paper benchmark canonical videos.',
                        'The paper-folding source advertises 3445 frames and decodes 3444; the selected extracted clip decodes fully.',
                        'Camera intrinsics come from calibration.json cam01 K. Bench extrinsics and moving-camera support were not calibrated. The grasping clip contains a large head turn.',
                        'Stage 2 object detection failed because torch and SAM 3 are absent from the configured interpreter. No alternative perception environment was configured or installed by this test.',
                        'Stages 3 and 4 were not run. Zero verified rollouts produced is not a measured 0% robot success rate.'],
         'regression_checks':[{'script':'tool_library/tests/test_stage1_goal_gate.py','exit_code':0,'printed_pass_cases':14},
                              {'script':'tool_library/tests/test_goal_language.py','exit_code':0,'printed_pass_cases':4}],
         'grasp_stage2':stage2,'grasp_perception_attempt':perception,'tasks':results}
(ROOT/'results.json').write_text(json.dumps(summary,indent=2))

esc=lambda value:html.escape(str(value))
cards=[]
for r in results:
    ident=r['id']; passed=r['compatible']
    goals=r['checks']['goal_language_coverage_check']
    evidence=goals.get('issues',[])
    status='Stage 1 checks pass; Stage 2 environment blocked' if passed else 'Stage 1 goal language rejected'
    native=r.get('native_stage1') or {}
    cards.append(f'''<article id="{ident}"><h2>{esc(ident[:2])}. {esc(r['task'])}</h2>
<p class="status {'pass' if passed else 'fail'}">{esc(status)}</p>
<p class="meta">Source {esc(r['uuid'])}<br>{r['start_s']:g}–{r['end_s']:g} s · 512 × 512 · {r['clip_decoded_frames']} decoded clip frames</p>
<video controls preload="metadata" src="clips/{ident}.mp4"></video>
<p><a href="clips/{ident}.mp4">Download clip</a> · <a href="runs/{ident}/stage1/overview.jpg">Frame evidence</a> · <a href="reviewed_semantics/{ident}.json">Reviewed semantics</a> · <a href="configs/{ident}.yaml">Run config</a></p>
<details><summary>Native goal checker evidence ({len(evidence)} issues)</summary><pre>{esc(json.dumps(goals,indent=2))}</pre></details>
<p class="meta">Native runner finished: {esc(native.get('finished','not recorded'))}; verdict OK: {esc((native.get('verdict') or {}).get('ok','not recorded'))}</p></article>''')
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DexAgent × EgoTools — 10-task test</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 22px;background:#f5f7fa;color:#182637}}h1{{font-size:34px;line-height:1.2}}a{{color:#145db5}}.lede{{max-width:950px}}.stats{{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}}.stats div{{background:white;padding:16px 22px;border-radius:10px;border:1px solid #d8e0e9}}.stats b{{font-size:26px;display:block}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:22px}}article{{background:white;border:1px solid #d8e0e9;border-radius:12px;padding:22px}}h2{{font-size:19px;margin-top:0}}video{{display:block;width:100%;max-height:420px;background:#152132;border-radius:6px}}.meta{{font-size:13px;color:#506278;overflow-wrap:anywhere}}.status{{font-weight:650}}.pass{{color:#146440}}.fail{{color:#9a431e}}pre{{font-size:12px;white-space:pre-wrap;word-break:break-word;max-height:460px;overflow:auto}}details{{margin:16px 0}}li{{margin:7px 0}}@media(max-width:500px){{.grid{{grid-template-columns:1fr}}body{{padding:0 14px}}}}</style>
<h1>DexAgent × EgoTools: 10-task compatibility test</h1>
<p class="lede">Ten public EgoTools video clips, including test-tube grasping and chopsticks. All clips decode fully and all reviewed documents pass the native schema and semantic-consistency checks. One task passes the goal-language checks; nine retain requirements that the current finite goal compiler rejects.</p>
<div class="stats"><div><b>10 / 10</b>Clips decoded</div><div><b>10 / 10</b>Schema + consistency</div><div><b>1 / 10</b>Goal language accepted</div><div><b>0</b>Verified robot rollouts produced</div></div>
<p class="lede"><strong>Scope:</strong> {esc(summary['scope'])} <strong>No robot success rate was measured.</strong></p>
<p class="lede">The grasping candidate passes the native Stage 1 delivery. Its Stage 2 attempt reaches object detection and reports missing SAM 3 / Torch in the configured interpreter; no verified scene is produced. The moving head camera also needs adaptation to DexAgent's documented fixed-camera input.</p>
<p><a href="results.json">Complete results JSON</a> · <a href="compatibility.json">Reproducible native check results</a> · <a href="https://github.com/dexagent1/DexAgent/tree/{runtime['dexagent_commit']}">Pinned DexAgent code</a> · <a href="https://huggingface.co/datasets/ropedia-ai/egotools-data/tree/{runtime['data_revision']}">Pinned public recordings</a> · <a href="https://arxiv.org/abs/2609.39378">EgoTools paper</a></p>
<div class="grid">{''.join(cards)}</div>
<h2>Interpretation and reproduction</h2><ul>{''.join('<li>'+esc(x)+'</li>' for x in summary['limitations'])}</ul>
<p>Two upstream regression scripts passed (18 printed PASS cases). The upstream checkout remains unmodified.</p>
<p>From the original workspace, use the Python executable recorded in <a href="runtime.json">runtime.json</a> to run <code>check_candidates.py</code>. Elsewhere, install the pinned checkout and dependencies, set <code>DEXAGENT_REPO=/path/to/DexAgent</code> and run that script in a matching Python environment. The portable bundle includes clips, source calibration/captions, configs, authored semantics, native run evidence, this report and reproduction scripts. Adjust absolute config paths before running the full pipeline elsewhere.</p></html>'''
(ROOT/'report.html').write_text(page)

# Include the ten selected clips and their evidence, not the full raw video corpus.
members=[p for p in ROOT.glob('*.py')]+[p for p in ROOT.glob('*.json') if p.name not in {'round1.json','round2.json'}]+[ROOT/'report.html']
for directory in ['clips','configs','reviewed_semantics','logs','runs','delivery_transport','reconstruction_transport']:
    members.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp')
members.extend([ROOT/'data/manifest.json',ROOT/'data/hub_info.json',ROOT/'data/caption_inventory.json'])
for r in results:
    source=ROOT/'data/raw'/r['uuid']/'ep1'
    members.extend(p for p in source.iterdir() if p.suffix in {'.json','.txt'})
with zipfile.ZipFile(ROOT/'dexagent_egotools_10.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for p in sorted(set(members)):
        archive.write(p,p.relative_to(ROOT))
    archive.writestr('README.txt','Open report.html for the ten task videos and test results.\nThis is an operator-assisted fixed-library compatibility test. No robot success rate was measured.\nReproduction: use check_candidates.py with the pinned DexAgent checkout and packages in runtime.json. Set DEXAGENT_REPO to the checkout path. Update absolute config paths before full pipeline runs elsewhere.\n')
print(json.dumps({k:summary[k] for k in ['n_tasks','decoded_clips','schema_pass','consistency_pass','goal_language_pass','verified_robot_rollouts_produced']}))
print(json.dumps({'bundle_bytes':(ROOT/'dexagent_egotools_10.zip').stat().st_size}))
