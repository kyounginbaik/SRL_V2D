"""Summarize recorded diagnostics without converting candidates into successes."""
from pathlib import Path
from collections import Counter
import datetime
import html
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'continuation'
def read(path,default=None):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else default
esc=lambda x:html.escape(str(x))
baseline=read('compatibility.json',[])
st2=Path('continuation/runs/grasp/stage2')
plane=read(st2/'table_plane.json')
rackfit=read(st2/'rack_placement.json')
rackvisual=read(st2/'rack_visual_checks.json',{})
rackrest=read(st2/'rack_rest_checks.json',{})
nominal=read('continuation/physics_probe/diagnostic.json',{})
lift=read('continuation/physics_probe/trials.json',[])
runtime=read('continuation/runtime_probe/result.json',{})
alternate=read('continuation/runtime_probe/result_alt.json',{})
optimized=read('continuation/runtime_probe/result_optimized.json',{})
search=read('continuation/runtime_probe/search_result.json',{})
optimization=read('continuation/runtime_probe/optimization_result.json',{})
native=read(st2/'run.json',{})
sam3=read('downloads/sam3/verified.json',{})
inferences={str(p.relative_to(OUT)):json.loads(p.read_text()) for p in (OUT/'runs/grasp/stage2/source_masks').glob('sam3_*/meta.json')}
sam_attempts=[]
trace=ROOT/st2/'trace.jsonl'
if trace.exists():
    for line in trace.read_text().splitlines():
        row=json.loads(line)
        if row.get('event')=='tool' and row.get('name')=='segment_with_text_prompts':
            result=row.get('result') or {}
            sam_attempts.append({'round':row.get('round'),'frames_dir':row.get('args',{}).get('frames_dir'),
                'ok':result.get('ok'),'error':result.get('error')})
functional_failures={r['id']:r['checks']['goal_language_coverage_check'].get('issues',[]) for r in baseline if not r['compatible']}
summary={
    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Operator-assisted continuation using an isolated adapted DexAgent checkout. Source reconstruction, nominal interaction tests, and mounted-arm diagnostics are reported separately.',
    'upstream_commit':'e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633',
    'verified_complete_robot_episodes':0,
    'benchmark_accuracy_measured':False,
    'baseline':{'task_count':len(baseline),'goal_language_accepted':sum(r['compatible'] for r in baseline),'functional_failures':functional_failures},
    'camera_plane':plane,'rack_placement':rackfit,'rack_visual_checks':rackvisual,
    'rack_rest_checks':{k:{kk:vv for kk,vv in v.items() if kk!='settled_qpos'} for k,v in rackrest.items()},
    'nominal_hand_diagnostic':nominal,
    'nominal_hand_lift_verdict':lift[0]['verdict'] if lift else None,
    'mounted_arm_diagnostics':{label:{'ok':r.get('ok'),'first_failure':r.get('first_failure')} for label,r in [('initial',runtime),('alternate',alternate),('optimized',optimized)] if r},
    'six_candidate_search':{'ok':search.get('ok'),'evaluations':search.get('evaluations'),
        'best_subgoals_passed':(search.get('best') or {}).get('subgoals_passed'),
        'failures':dict(Counter(', '.join(row.get('failed') or []) for row in search.get('table',[]))),
        'note':'The search tool reported improved=true by its shortfall ranking while every candidate passed zero subgoals. This report does not count that flag as task progress.'},
    'contact_optimization':{k:optimization.get(k) for k in ['ok','status','n_evals','n_tested','failure_classes']},
    'chopsticks_measurements':read('continuation/chopsticks/measurements.json'),
    'chopsticks_perception':read('continuation/chopsticks/perception_result.json'),
    'tube_size_sensitivity':read(st2/'tube_size_sensitivity.json'),
    'native_stage2':native,
    'native_finish_feedback':read('continuation/native_finish_feedback.json'),
    'sam3_checkpoint':sam3,'sam3_inference_outputs':inferences,'sam3_native_attempts':sam_attempts,
    'changes':[
        'Added construct_round_test_tube with real hollow collision geometry; three size variations compile and pass mass, inertia, cavity, free-joint and no-equality checks.',
        'Added fit_camera_to_table_plane_cv; fixes the sign mismatch with rigcam camera registration. Three synthetic depth planes recover pitch, roll and metric height.',
        'Patched SAM3 CPU initialization caches, fused-MLP dtype handling and GPU-only pinned memory. Three position-encoding resolutions and three MLP shapes match ordinary CPU operations; the 840,509,750-parameter image model builds on CPU. Inference outcomes are recorded separately.'],
    'limitations':[
        'The test-tube interaction assets use nominal tube dimensions and placement. Their local lift pass is not a video-reconstructed episode.',
        'The source-derived rack is a partial reconstruction. A rack fit does not validate the tube or whole scene.',
        'Near-table-edge alignment assumes the demonstration edge maps to the supplied bench edge and the camera is laterally centred. No station or robot was moved.',
        'The native right-arm reach bound passes for the rack but sampled grasp-orientation probes are inconclusive.',
        'CPU GrabCut could not isolate the transparent tube, including the later revealed stem. No accepted mask was painted or substituted.',
        'Chopsticks require independent contact-based tool success conditions. The original unsupported requirements remain present in all baseline results.',
        'This is not an official EgoTools multiple-choice benchmark evaluation or an unaided VLM evaluation.',
        'No verified complete robot episode or robot success rate was obtained. Failed recorded segments are diagnostic evidence only.'],
}
(OUT/'results.json').write_text(json.dumps(summary,indent=2))
tasks=''.join(f'<tr><td>{esc(r["id"][:2])}</td><td><a href="../clips/{esc(r["id"])}.mp4">{esc(r["task"])}</a></td><td>{"Accepted" if r["compatible"] else "Unsupported goal clause"}</td></tr>' for r in baseline)
lift_verdict=summary['nominal_hand_lift_verdict'] or {}
failure=runtime.get('first_failure') or {}
measured=failure.get('measured') or {}
sam_status='Checkpoint SHA256 verified; see inference outputs below.' if sam3 else 'Checkpoint download incomplete; no SAM3 inference claim.'
inference_text=json.dumps({'native_attempts':sam_attempts,'successful_outputs':inferences},indent=2)
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DexAgent × EgoTools: execution follow-up</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 20px;color:#182637;background:#f6f8fb}}a{{color:#145db5}}h1{{font-size:32px}}h2{{font-size:23px;margin-top:34px}}.card{{background:white;padding:22px;border:1px solid #d9e1eb;border-radius:10px;margin:20px 0}}.bad{{color:#9d321d}}.ok{{color:#17643c}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce3ed}}img,video{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:520px;overflow:auto}}.note{{font-size:14px;color:#526071}}</style>
<h1>DexAgent × EgoTools: execution follow-up</h1>
<p class="bad"><strong>No verified complete robot episode was produced.</strong> Testing progressed through source-derived rack reconstruction, an isolated hand lift test, mounted-arm physical execution, six grasp candidates and contact optimization.</p>
<p><a href="../report.html">Original 10-task compatibility report</a> · <a href="results.json">Follow-up results JSON</a> · <a href="changes/DexAgent_extensions.patch">DexAgent code patch</a> · <a href="changes/SAM3_cpu_initialization.patch">SAM3 CPU patch</a> · <a href="changes/README.txt">Reproduction notes</a></p>
<div class="card"><h2>Grasping: what passed and what failed</h2>
<p class="ok">The nominal open-tube hand test passed at 6, 10 and 15 cm, reaching 59.9, 99.9 and 149.9 mm. Maximum measured contact penetration was {(lift_verdict.get('penetration') or {}).get('max_contact_mm','unavailable')} mm.</p>
<p class="bad">The mounted-arm trial failed {esc(', '.join(failure.get('failed') or []))}. Final tube rise was {1000*measured.get('rise_m',0):.1f} mm; the rack moved {1000*((measured.get('untouched') or {}).get('rack_rack') or {}).get('moved_m',0):.1f} mm. The six-candidate search passed zero subgoals. An optimized contact pose also failed physical execution.</p>
<video controls preload="metadata" src="runtime_probe/video_failed/view0.mp4"></video>
<p class="note">Failed physical segment on nominal assets, shown for diagnosis. This is not a successful source-video robot rollout. The arm and hand are fixed together; the episode uses the supplied bounded motors.</p>
<p><a href="runtime_probe/execution/execution_report.json">Initial execution report</a> · <a href="runtime_probe/search_result.json">Six-candidate search</a> · <a href="runtime_probe/result_optimized.json">Optimized execution failure</a> · <a href="physics_probe/trials.json">Isolated hand stress test</a></p></div>
<div class="card"><h2>Source-derived partial reconstruction</h2>
<p>The recorded depth gives a camera height of {(plane or {}).get('camera_height_m','?')} m. The fitted rack is about 14.6 × 5.0 cm. Its native silhouette IoU is {((rackvisual.get('silhouette_iou_check') or {}).get('iou') or {}).get('rack_rack','?')}; native scale, settling and support checks pass.</p>
<img src="runs/grasp/stage2/rack_compare.png" alt="Video, reconstructed rack and blend through the fitted camera">
<p class="note">Only the rack is reconstructed in this comparison. The transparent tube's full dimensions and initial pose remain unverified. SAM3 returns empty masks at both views for “test tube” (0.3 threshold), and for “glass tube” and “transparent tube” in the revealed view (0.1). An empty mask in a visibly revealed view cannot certify that the initial object is hidden.</p>
<p class="bad">Native final verification rejects the partial scene: the required test tube is absent, and 51 rack collision geoms lie within the local neighborhood against the unchanged limit of 48. Source-video identity and source-mask provenance pass. This is a failed reconstruction attempt.</p>
<p><a href="runs/grasp/stage2/table_plane.json">Camera plane</a> · <a href="runs/grasp/stage2/table_edge_registration.json">Table alignment assumptions</a> · <a href="runs/grasp/stage2/rack_visual_checks.json">Visual checks</a> · <a href="runs/grasp/stage2/rack_rest_checks.json">Rest checks</a> · <a href="runs/grasp/stage2/tube_size_sensitivity.json">Tube size uncertainty</a></p></div>
<div class="card"><h2>Chopsticks</h2><p>The source clip and corresponding recorded depth were read. SAM3 detects both chopsticks, bread and the plate at source frame 0; no grape instance passes its default 0.3 threshold. The native goal compiler rejects the requirement that the grape be clamped between the two chopstick tips. No chopstick task success or tool-mediated grasp is claimed.</p>
<video controls preload="metadata" src="../clips/02_chopsticks_grape.mp4"></video>
<img src="chopsticks/perception_overlay.jpg" alt="Two chopstick instances, bread and plate detected; no grape instance">
<p><a href="chopsticks/perception_result.json">SAM3 perception results</a> · <a href="chopsticks/depth_source.json">Depth byte-range provenance</a> · <a href="chopsticks/measurements.json">Measurement diagnostics</a> · <a href="../reviewed_semantics/02_chopsticks_grape.json">Unchanged tool requirements</a></p></div>
<h2>All ten selected tasks</h2><table><tr><th>#</th><th>Video task</th><th>Original goal checker</th></tr>{tasks}</table>
<h2>Perception setup and code verification</h2><p>{esc(sam_status)} The CPU-only perception environment is isolated from shared environments. SAM3 required CPU cache, precision and pinned-memory adaptations; their patch and test logs are included.</p>
<p><a href="sam3_cpu_cache_tests.log">CPU cache tests</a> · <a href="sam3_cpu_mlp_tests.log">CPU MLP tests</a> · <a href="test_sam3_cpu.py">Reproduce CPU checks</a> · <a href="sam3_cpu_build.log">CPU image-model build</a> · <a href="changes/tool_library/tests/test_construct_round_test_tube.py">Tube tests</a> · <a href="changes/tool_library/tests/test_fit_camera_to_table_plane_cv.py">Camera tests</a></p>
<details><summary>SAM3 inference outputs</summary><pre>{esc(inference_text)}</pre></details>
<h2>Interpretation</h2><ul>{''.join('<li>'+esc(x)+'</li>' for x in summary['limitations'])}</ul>
<p>Upstream DexAgent remains unchanged at <a href="https://github.com/dexagent1/DexAgent/tree/{summary['upstream_commit']}">{summary['upstream_commit'][:12]}</a>. The <a href="https://github.com/Ropedia/EgoTools">official EgoTools benchmark</a> is a separate evaluation protocol. This experiment tests reconstruction and robot-execution compatibility of ten selected public recording clips.</p></html>'''
(OUT/'report.html').write_text(page)

if __name__=='__main__':
    print(json.dumps({'report':str(OUT/'report.html'),'verified_complete_robot_episodes':0,'sam3_checkpoint_verified':bool(sam3)}))
