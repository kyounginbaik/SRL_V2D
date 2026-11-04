"""list refit results sorted by a combined score sum_obj(cm/3 + deg/30) (passing only), v19 = 1.050/12.56 + 1.986/15.68"""
import json, glob, sys
rows = []
for f in glob.glob('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/refit/*/evaluation/result.json'):
    m = json.load(open(f))['metrics']['official_style_reference_proxy']; o = m['objects']
    c, s = o['blue_cup'], o['wooden_spoon']; cc, cd, sc, sd = c['mean_position_error_cm'][0], c['mean_orientation_error_degrees'][0], s['mean_position_error_cm'][0], s['mean_orientation_error_degrees'][0]
    rows.append((m['mp_sr'], cc / 3 + cd / 30 + sc / 3 + sd / 30, f.split('/')[-3], cc, cd, sc, sd))
v19 = 1.050 / 3 + 12.56 / 30 + 1.986 / 3 + 15.68 / 30
print(f'v19 score {v19:.3f}')
for r in sorted(rows, key=lambda r: (-r[0], r[1]))[:int(sys.argv[1]) if len(sys.argv) > 1 else 25]:
    print(f'{r[0]:.0f} {r[1]:.3f} {r[2]:40s} cup {r[3]:.3f}/{r[4]:5.2f} spoon {r[5]:.3f}/{r[6]:5.2f}' + ('  <- spoon<=v19' if r[5] <= 1.986 and r[0] == 1 else ''))
