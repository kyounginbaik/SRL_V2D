"""README.md for a variant-A14 bundle from its manifest.json (numbers only from recorded runs). usage: make_readme14.py BUNDLE_DIR"""
import sys, json
from pathlib import Path
B = Path(sys.argv[1])
m = json.loads((B / 'manifest.json').read_text())
p, pv = m['params'], m['public_validation']
cq, nq = m['eval_qc_clock_consistent'], m['eval_qc_unchanged_qcpy']
strong = m['variant'].endswith('strong')
L = [f"# {B.name}: Track 3 recon variant A on v14_g17 ({'strong' if strong else 'mild'} natural-motion smoother), 2026-10-30\n"]
L.append(f"Base: `selected20_final_v14_g17` (the reconstruction the team package v29 targets). Same folder layout and file set; only "
         f"`object_poses.npz` differs in {len(m['changed_episodes'])} episodes (all eval episodes except 32). Episode 32 and every other file "
         "(meshes, camera.npz, camera_pre_g15.npz, reports, selection/prepared/meshes json, bundle_manifest.json, G11-G17 CHANGES files) are byte-identical to v14_g17. "
         "npz written with savez_compressed like v14_g17. Frame 0 of every object is unchanged. No eval GT read; public GT only inside the unchanged scorer; our meshes only.\n")
L.append("## Rule\n")
L.append("Same variant-A rule as `A_smooth_mild` / `A_smooth_strong` (robust Whittaker smoother on mesh-centroid position and quaternions, Tukey spike "
         "rejection, no smoothing across persistent jumps > 60 deg / 6 cm, frame 0 pinned; then an image guard re-anchors any frame whose image fit gets worse).")
L.append("**Clock fix for v14:** G15 resampled every v14 pose (world_from_object, camera_from_object, world_from_rgb) at t + delta (delta 0.875-0.925 frames, ep20 0.925/1.85) "
         "onto the stereo clock, but the SAM2 masks and spoon ball detections are on the RGB clock. GT-free check on the stored v14 poses: shifting them back by -delta raises IoU "
         "(ep5 spoon 0.627 -> 0.744, ep33 spoon 0.604 -> 0.674, ep5 pot 0.706 -> 0.738) and lowers ball px (ep5 23.4 -> 20.1), while the stereo depth residual fits the stored index "
         "better for the moving spoons (ep5 11.8 vs 14.9 mm). So the guard (`guard_clip14.py`) scores IoU and balls with the track shifted by -delta (team `retime.shift`) "
         "and the depth residual with the track as stored. camera_from_object gets the same body-frame change as world_from_object.\n")
L.append(f"Parameters: `{json.dumps(p)}`.\n")
L.append("## Public / held-out validation (unchanged scorer, GT only inside it; rule applied on top of the team's G15 public export `g15_camrt`)\n")
L.append("| reference | MP-SR | SP-SR | mean pos cm | mean ori deg | MPPE | AUC | objects worse > 0.05 cm | max object delta |")
L.append("|---|---|---|---|---|---|---|---|---|")
g = pv['vs_g15_camrt']; r = pv['vs_r8_base']
L.append(f"| g15_camrt (v14 policy, base) | 20/20 | 20/20 | {g['base_pos']:.3f} | {g['base_ori']:.2f} | {g['base_mppe']:.3f} | {g['base_auc']:.4f} | - | - |")
L.append(f"| **this rule on g15_camrt** | {g['mpsr']:.0f}/20 | {g['spsr']:.0f}/20 | {g['pos']:.3f} | {g['ori']:.2f} | {g['mppe']:.3f} | {g['auc']:.4f} | {g['n_pos_worse']:.0f} | +{g['max_pos_delta']:.2f} cm / +{g['max_ori_delta']:.2f} deg |")
if r:
    L.append(f"| r8_base (v8 policy) for reference | 20/20 | 20/20 | {r['base_pos']:.3f} | {r['base_ori']:.2f} | {r['base_mppe']:.3f} | {r['base_auc']:.4f} | {r['n_pos_worse']:.0f} (rule vs r8_base) | +{r['max_pos_delta']:.2f} cm / +{r['max_ori_delta']:.2f} deg |")
L.append("")
c = cq['summary']
L.append("## Eval QC (GT-free)\n")
L.append(f"Clock-consistent (all frames, all 34 changed objects): IoU {c['iou_before']:.3f} -> {c['iou_after']:.3f}, median |depth residual| {c['absz_before']:.2f} -> {c['absz_after']:.2f} mm, "
         f"frames with IoU drop > 0.1: {c['iou_drop_gt0p1']}, spoon ball px mean {c['ball_before']:.2f} -> {c['ball_after']:.2f}. Steps > 30 deg {c['rot_gt30'][0]} -> {c['rot_gt30'][1]}, "
         f"> 15 deg {c['rot_gt15'][0]} -> {c['rot_gt15'][1]}, centre steps > 3 cm {c['pos_gt3'][0]} -> {c['pos_gt3'][1]}, angular-acceleration events > 20 deg/frame^2 {c['racc_gt20'][0]} -> {c['racc_gt20'][1]}.")
if nq:
    s = nq['summary']
    L.append(f"\nTeam `qc.py` unchanged (stored poses vs RGB-clock masks, the G15 README caveat; changed frames): IoU {s['iou_before']:.3f} -> {s['iou_after']:.3f}, "
             f"|depth| {s['absz_before']:.2f} -> {s['absz_after']:.2f} mm, frames with IoU drop > 0.1: {s['iou_drop_gt0p1']}, ball {s['ball_before']:.2f} -> {s['ball_after']:.2f}.")
L.append("\n| ep | object | IoU (RGB clock) | abs depth mm (stereo clock) | ball median px | ball p90 px | steps > 30 | steps > 15 | centre > 3 cm |")
L.append("|---|---|---|---|---|---|---|---|---|")
f2 = lambda x, k=2: '-' if x is None else f'{x:.{k}f}'
for ep, objs in cq['objects'].items():
    for n, o in objs.items():
        L.append(f"| {int(ep[-6:])} | {n} | {o['iou'][0]:.3f} -> {o['iou'][1]:.3f} | {o['absz'][0]:.2f} -> {o['absz'][1]:.2f} | {f2(o['ball_median'][0])} -> {f2(o['ball_median'][1])} | "
                 f"{f2(o['ball_p90'][0], 1)} -> {f2(o['ball_p90'][1], 1)} | {o['rot_gt30'][0]} -> {o['rot_gt30'][1]} | {o['rot_gt15'][0]} -> {o['rot_gt15'][1]} | {o['pos_gt3'][0]} -> {o['pos_gt3'][1]} |")
kept = [(ep, n, o['kept_jumps_at_frames']) for ep, v in m['changed_episodes'].items() for n, o in v['objects'].items() if o['kept_jumps_at_frames']]
L.append("\nPersistent jumps kept as in v14_g17: " + ('; '.join(f"ep{int(e[-6:])} {n} at f{','.join(map(str, cc))}" for e, n, cc in kept) or 'none') + ".\n")
L.append("## Limits\n- Not retargeted: no robot rollout evidence for this bundle.\n- No public clip has a spoon; spoon changes are checked GT-free only.")
if strong:
    L.append("- The strong rule accepts a bounded per-frame image cost (IoU <= 0.1, depth <= 6 mm, ball <= 6 px per frame) for smoother motion; see the per-object table.")
L.append("\n## Files\n- `manifest.json`: changed objects (change size, kept jumps, steps before/after, delta), sha256 before/after per changed npz, sha256 of every file, "
         "parameters, public scores per object vs g15_camrt and vs r8_base, eval QC (clock-consistent and unchanged qc.py).\n"
         "- Code: `/home/hansol/v2d/outputs/variants_20261030/A_work/code/` (slib.py, guard_clip.py, guard_clip14.py, build_bundle14.py, compare14.py, clockqc14.py, qc_compare.py).")
(B / 'README.md').write_text('\n'.join(L) + '\n')
print('README', B / 'README.md')
