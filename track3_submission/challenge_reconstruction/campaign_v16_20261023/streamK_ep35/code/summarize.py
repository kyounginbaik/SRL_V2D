"""Print MP-SR proxy numbers for every result.json under the given dirs (recursive, depth-limited). usage: summarize.py DIR [DIR..]"""
import json, sys, glob, os
for root in sys.argv[1:]:
    for f in sorted(glob.glob(root+'/**/result.json', recursive=True)):
        try: r=json.load(open(f))
        except Exception as e: print(f, 'ERR', e); continue
        m=r.get('metrics')
        if not m or 'official_style_reference_proxy' not in m: continue
        o=m['official_style_reference_proxy']['objects']
        print(os.path.relpath(os.path.dirname(f), root), 'MP-SR', m['reference_proxy_mp_sr'], {k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()}, ('inhand=%s'%{k:round(v,1) for k,v in r['inhand'].items() if 'drift' in k}) if isinstance(r.get('inhand'),dict) and 'drift_cm_at_last_hold' in r['inhand'] else '')
