"""README.md for a variant-A bundle from its manifest.json (numbers only from the recorded runs). usage: make_readme.py BUNDLE_DIR"""
import sys, json
from pathlib import Path
B = Path(sys.argv[1])
m = json.loads((B / 'manifest.json').read_text())
p, pub, qs, qo = m['params'], m['public_validation'], m['eval_qc_summary'], m['eval_qc_per_object']
strong = m['variant'].endswith('strong')
L = []
L.append(f"# {B.name}: Track 3 recon variant A ({'strong' if strong else 'mild'} natural-motion smoother), 2026-10-30\n")
L.append(f"Base: `selected20_final_v8_g10b` (= the files of the shared `recon_v26_selected20_final_v8_g10b`, without its README/videos). "
         f"Same folder layout and file set; only `object_poses.npz` differs in {len(m['changed_episodes'])} episodes "
         f"(all eval episodes except 32). Episode 32 is byte-identical to v8_g10b (owned by streams G12/H12). Meshes, cameras, "
         f"intrinsics, selection.json, prepared.json and `bundle_manifest.json` are byte-identical to v8_g10b; the step statistics inside "
         f"`bundle_manifest.json` therefore still describe the v8_g10b poses (the new ones are in `manifest.json`). Frame 0 of every object is unchanged, so the "
         f"scene start / anchor alignment is the same as v8_g10b. No ground truth read for eval; public GT used only inside the unchanged public scorer; our meshes only.\n")
L.append("## Rule (one rule for every clip and object)\n")
L.append("1. Robust SE(3) smoother per object track (world_from_object), at the mesh centroid and on sign-continuous quaternions: Whittaker smoother "
         "(2nd-difference = acceleration penalty), IRLS Tukey weights from a light pass (lam 2; 25 deg / 3 cm) so isolated spikes are bridged, "
         "and NO smoothing across persistent jumps (a step > 60 deg or 6 cm that does not come back within 4 frames is kept exactly, e.g. deliberate spoon rolls / re-registrations). Declared symmetries as in the public kit (cups 360, lid 2).")
L.append("2. Image guard (team evidence, `fuse.evidence` with `qc.py`'s exact mask/exclusion/depth preparation, and the spoon gray-ball "
         "reprojection with the spoon_roll_v8 model): a smoothed frame whose silhouette IoU drops by more than d_iou, whose median |mesh-stereo depth| "
         "rises by more than d_z, or whose ball reprojection rises by more than d_ball is re-anchored to the base (data weight x10) and the track re-smoothed, "
         "until no frame fails. So the smoother only moves frames where the image does not object (jitter, occluded frames, spikes); fast but image-consistent motion (e.g. ep5 stirring) is kept.\n")
L.append(f"Parameters: `{json.dumps(p)}` (defaults: jump 60 deg / 6 cm, hold 4, Tukey 25 deg / 3 cm, lam_det 2, pin frame 0; guard d_iou 0.05, d_z 3 mm, d_ball 3 px, 8 rounds unless set).\n")
L.append("## Public / held-out validation (unchanged scorer, GT only inside it; 10 public + 10 held-out, 33 objects)\n")
L.append("| variant | MP-SR | SP-SR | mean pos cm | mean ori deg | MPPE | AUC | objects worse > 0.05 cm | max object delta |")
L.append("|---|---|---|---|---|---|---|---|---|")
L.append(f"| r8_base (= base policy) | 20/20 | 20/20 | {pub['base_pos']:.3f} | {pub['base_ori']:.2f} | {pub['base_mppe']:.3f} | {pub['base_auc']:.4f} | - | - |")
L.append(f"| **this bundle's rule** | {pub['mpsr']:.0f}/20 | {pub['spsr']:.0f}/20 | {pub['pos']:.3f} | {pub['ori']:.2f} | {pub['mppe']:.3f} | {pub['auc']:.4f} | {pub['n_pos_worse']:.0f} | +{pub['max_pos_delta']:.2f} cm / +{pub['max_ori_delta']:.2f} deg |\n")
L.append("The other strengths tried are in the variant-A log (`docs/logs/2026-10-30-variant-A.md` in the local folder) and `/home/hansol/v2d/outputs/variants_20261030/README_A.md`.\n")
L.append("## Eval QC (GT-free; team `qc.py` unchanged + ball reprojection), on the frames this rule changed\n")
L.append(f"All changed objects: IoU {qs['iou_before']:.3f} -> {qs['iou_after']:.3f}, median |depth residual| {qs['absz_before']:.2f} -> {qs['absz_after']:.2f} mm, "
         f"frames with IoU drop > 0.1: {qs['iou_drop_gt0p1']}, spoon ball px (changed frames, mean) {qs['ball_before']:.2f} -> {qs['ball_after']:.2f}.\n")
L.append("| ep | object | changed frames | IoU | abs depth mm | steps > 30 deg | steps > 15 deg | center steps > 3 cm | ball clip median px | ball clip p90 px |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for ep, objs in qo.items():
    for n, r in objs.items():
        bm = f"{r['ball_clip_median'][0]:.2f} -> {r['ball_clip_median'][1]:.2f}" if 'ball_clip_median' in r else '-'
        b9 = f"{r['ball_clip_p90'][0]:.1f} -> {r['ball_clip_p90'][1]:.1f}" if 'ball_clip_p90' in r else '-'
        L.append(f"| {int(ep[-6:])} | {n} | {r['changed']} | {r['iou_before']:.3f} -> {r['iou_after']:.3f} | {r['absz_before']:.1f} -> {r['absz_after']:.1f} | "
                 f"{r['rot_gt30'][0]} -> {r['rot_gt30'][1]} | {r['rot_gt15'][0]} -> {r['rot_gt15'][1]} | {r['pos_gt3'][0]} -> {r['pos_gt3'][1]} | {bm} | {b9} |")
tot = lambda k, i: sum(r[k][i] for o in qo.values() for r in o.values())
L.append(f"\nTotals over the 19 episodes: steps > 30 deg {tot('rot_gt30',0)} -> {tot('rot_gt30',1)}, > 15 deg {tot('rot_gt15',0)} -> {tot('rot_gt15',1)}, "
         f"centre steps > 3 cm {tot('pos_gt3',0)} -> {tot('pos_gt3',1)}, angular-acceleration events > 20 deg/frame^2 {tot('racc_gt20',0)} -> {tot('racc_gt20',1)}.\n")
kept = [(ep, n, o['kept_jumps_at_frames']) for ep, v in m['changed_episodes'].items() for n, o in v['objects'].items() if o['kept_jumps_at_frames']]
L.append("Persistent jumps kept as in v8_g10b (not smoothed across): " + '; '.join(f"ep{int(e[-6:])} {n} at f{','.join(map(str, c))}" for e, n, c in kept) + ".\n")
L.append("## Limits\n")
L.append("- Not yet retargeted: no robot rollout evidence for this bundle. The smoother removes tracker jitter/spikes; real fast motion stays because the image supports it.")
L.append("- No public clip has a spoon, so spoon changes are checked only GT-free (IoU, depth, balls).")
if strong:
    L.append("- The strong rule accepts a bounded per-frame image cost (IoU <= 0.1, depth <= 6 mm, ball <= 6 px per frame) for smoother motion. Largest object-level costs: "
             "ep5 spoon ball clip median +0.7 px (p90 +4.4 px), ep5 pot IoU -0.010, ep14 blue cup |depth| +1.4 mm.")
L.append("\n## Files\n- `manifest.json`: changed episodes/objects with change size, kept jumps, steps before/after, sha256 before/after; sha256 of every file; parameters; public scores (per object); eval QC (per object).")
L.append("- Code: `/home/hansol/v2d/outputs/variants_20261030/A_work/code/` (slib.py, guard_clip.py, build_bundle.py, compare.py, qc_compare.py).")
(B / 'README.md').write_text('\n'.join(L) + '\n')
print('README', B / 'README.md')
