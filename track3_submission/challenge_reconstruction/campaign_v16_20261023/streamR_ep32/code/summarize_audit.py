"""Print 0.5 s bins of an audit JSON (audit_ep32.py)."""
import sys, json, numpy as np
d = json.load(open(sys.argv[1])); rows = d['rows']; step = int(sys.argv[2]) if len(sys.argv) > 2 else 10
keys = ['ref_cup_tilt_vs_table_deg', 'fit_cup_tilt_vs_table_deg', 'cup_fit_axis_vs_ref_deg', 'ref_cup_bottom_z_cm', 'fit_cup_bottom_z_cm', 'table_z_at_cup_cm', 'cup_res_ref_cm', 'cup_res_fit_cm',
        'cup_iou_ref', 'bowl_below_rim_cm', 'bowl_radial_cm', 'fit_bowl_below_rim_cm', 'fit_bowl_radial_cm', 'spoon_res_ref_cm', 'spoon_res_fit_cm', 'spoon_fit_shift_cm', 'spoon_fit_rot_deg', 'spoon_fit_roll_deg', 'spoon_iou_ref', 'spoon_in_wall_pen_cm', 'ref_spoon_min_z_cm']
hdr = ['tiltR', 'tiltF', 'axRF', 'botR', 'botF', 'tabz', 'cResR', 'cResF', 'cIoU', 'bDepR', 'bRadR', 'bDepF', 'bRadF', 'sResR', 'sResF', 'sShft', 'sRot', 'sRoll', 'sIoU', 'wallP', 'sMinZ']
print('   t  ' + ' '.join(f'{h:>6}' for h in hdr))
fr = np.array([r['frame'] for r in rows])
for s in range(0, fr.max() + 1, step):
    g = [r for r in rows if s <= r['frame'] < s + step]
    print(f'{s/20:5.1f} ' + ' '.join(f"{np.nanmean([r.get(k, np.nan) for r in g]):6.2f}" for k in keys))
