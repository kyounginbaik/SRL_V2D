import json,os
R='/scratch2/exaFLOPs/srl_yam_run/comparison/results'
for a in 'UAC':
    p=f'{R}/upper_test_{a}.json'
    if not os.path.exists(p): print(a,'MISSING upper test'); continue
    rows=json.load(open(p))
    print('ARM',a)
    for r in rows:
        print(f"  {r['steps']/1e6:5.0f}M seed={r['seed']:5d} kind={r['kind']:11s} {r['kinds']} frontier={r['frontier']:3d} min={r['frontier_min']:3d} slip={r['slip']:3d} cm={r['cm']:.3f} deg={r['deg']:.2f} pass={r['passed']:.2f} rise={r['object_max_rise_cm']:.2f} [{r['object_max_rise_cm_worst']:.2f},{r['object_max_rise_cm_best']:.2f}] move={r['object_max_move_cm']:.2f}")
keys=['chord_sr','maniptrans_sr','mppe_cm','mean_add_m','add_auc','object_position_error_m','object_orientation_error_deg','spider_position_error_centered_m','spider_sr_uncentered','world_count','environment_steps','cause_fraction/object','cause_fraction/wrist']
print('HARNESS')
for a in 'UAC':
    p=f'{R}/harness_{a}.json'
    if not os.path.exists(p): print(a,'MISSING harness'); continue
    d=json.load(open(p)); s=d.get('scalars',d)
    g=lambda k:s.get('evaluation/'+k,s.get(k))
    print(' ',a,{k:(round(g(k),4) if isinstance(g(k),float) else g(k)) for k in keys})
    if a=='U': print('  harness keys',list(d)[:20])
