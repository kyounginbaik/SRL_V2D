import json, sys, glob
best = None
for d in sorted(glob.glob(sys.argv[1] + '/trial*')):
    try: r = json.load(open(d + '/result.json'))
    except Exception: continue
    o = r['metrics']['official_style_reference_proxy']['objects']; ratio = max(max(v['mean_position_error_cm'][0] / 3, v['mean_orientation_error_degrees'][0] / 30) for v in o.values())
    if best is None or ratio < best[0]: best = (ratio, d)
print(best[1] if best else '')
