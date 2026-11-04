"""Stream R ep32: write diag/evidence_summary.json = the reference-correction decision (none applied) with the evidence numbers
and sha256 of every evidence file. Reads diag/audit_orig_v2_s1.json (stereo ICP per frame), diag/rest_spoon_fit.json,
diag/marker_roll_orig.json and the v19 rollout."""
import json, hashlib, numpy as np
from pathlib import Path
S = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32'); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
au = json.load(open(S / 'diag/audit_orig_v2_s1.json'))['rows']; m = lambda k, sel=lambda r: True: float(np.nanmean([r[k] for r in au if k in r and sel(r)]))
lifted = lambda r: r['ref_cup_bottom_z_cm'] > 2.0
rest = json.load(open(S / 'diag/rest_spoon_fit.json'))['rows']
out = dict(episode=32, decision='NO reference correction applied: independent evidence does not show the original reference poses to be worse where the robot disagrees with them',
  reference_source='/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032/object_poses.npz', reference_sha256=sha('/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000032/object_poses.npz'),
  contradictions=dict(
    a_cup_lift_tilt=dict(claim='reference cup lifts ~1 cm and tilts 10-12 deg with ~1 N hand force', finding='the ~1 N is the ROBOT left-hand force; the human grasps and lifts the cup 4-11 cm and tilts it up to ~58 deg (RGB). Stereo ICP of the cup mesh agrees with the reference: axis %.1f deg, bottom height diff %.2f cm mean over all frames; while lifted %.1f deg' % (m('cup_fit_axis_vs_ref_deg'), float(np.nanmean([abs(r['fit_cup_bottom_z_cm'] - r['ref_cup_bottom_z_cm']) for r in au if 'fit_cup_bottom_z_cm' in r])), m('cup_fit_axis_vs_ref_deg', lifted)),
      cup_depth_residual_cm=dict(reference=m('cup_res_ref_cm'), icp_fit=m('cup_res_fit_cm')), cup_silhouette_iou_reference=m('cup_iou_ref'),
      only_disagreement='0.4-1.1 s (grasp-and-lift, fast motion): stereo fit differs by up to 13 deg; the robot is even farther from the fit there, and nothing is physically impossible -> not corrected',
      v19_constant_offset='robot cup body sits ~1 cm lower than the reference at rest because the shipped 3c3c cup is 1.78 cm shorter at the bottom than the 57bb mesh the reference was fitted with (stereo shows 57bb bottom 0.8-1.1 cm inside the table at rest -> mesh height, not pose)'),
    b_spoon_bowl_misses_cup=dict(claim='spoon bowl misses the cup 2.6-7 cm at 3-6 s', finding='stereo ICP spoon positions agree with the reference within %.2f cm (mean); the miss is the robot bowl relative to the reference, not a reference artefact; no spoon-into-cup-wall penetration > %.2f cm' % (m('spoon_fit_shift_cm'), max(r['spoon_in_wall_pen_cm'] for r in au))),
    c_in_hand_roll=dict(claim='in-hand roll 22-37 deg at 7-10.5 s', finding='not resolvable: spoon roll from stereo is ill-conditioned (multi-start ICP at rest flips +-15 deg with equal residual; fit spread p90 18 deg), mocap balls sit only 1.0-2.4 cm off the long axis (roll noise 10-17 deg per 3 mm, single vs pair estimates inconsistent)'),
    rest_spoon=dict(finding='reference spoon jitters 1-7 deg while static 0-2.3 s; physics settles it ~10 deg away but that settled pose fits the stereo depth WORSE than the reference (residual %.2f vs %.2f cm) -> not corrected' % (float(np.mean([r['res_sim_cm'] for r in rest])), float(np.mean([r['res_ref_cm'] for r in rest]))))),
  three_way_v19='diag/threeway_v19.txt: robot vs reference cup 9.56 deg/1.29 cm, robot vs stereo 10.17/1.46, reference vs stereo 2.78/0.34; spoon 1.54/1.67/0.38 cm',
  evidence_files={str(p): sha(p) for p in [S / 'diag/audit_orig_v2_s1.json', S / 'diag/audit_orig_bins.txt', S / 'diag/threeway_v19.txt', S / 'diag/rest_spoon_fit.json', S / 'diag/markers_orig.json', S / 'diag/marker_roll_orig.json',
                                             S / 'code/audit_ep32_v2.py', S / 'code/threeway.py', S / 'code/rest_spoon_fit.py', S / 'code/spoon_markers.py', S / 'code/marker_roll.py']},
  stereo_source='/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000032/depth_rgb', masks='/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000032/{blue_cup,wooden_spoon}/masks')
(S / 'diag/evidence_summary.json').write_text(json.dumps(out, indent=2)); print(json.dumps(out['contradictions'], indent=1)[:2500])
