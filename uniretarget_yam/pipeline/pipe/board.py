import glob,json,os,re
H='/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/';R='/scratch2/exaFLOPs/srl_yam_run/'
for d in sorted(glob.glob(H+'episodes/recon_*')):
    ep=d.split('_')[-1];n=int(ep);row=[f'ep{n}']
    try:
        m=json.load(open(d+'/recon_meta.json'));mv=max(m['movers'],key=lambda x:x['move_cm']);row+= [mv['object'],mv['side'],f"move {mv['move_cm']}cm",f"frames {m['frames']}",f"movers {len(m['movers'])}"]
        m.get('untracked') and row.append('UNTRACKED '+str(m['untracked']));hs=m.get('holds') or [];hs and row.append(str(m['hold'].get('kinds'))+' '+' '.join(f"{x['object'][:8]}:{x['side'][0].upper()}[{x['start']}-{x['end']}]" for x in hs));h=m.get('hold');c=m.get('conditioning');h and row.append(f"hold {h['side']} to {h['end']}/{h['full_frames']-1} cover {h['coverage']}");c and row.append(f"smooth {c['sigma']} dev {c['dev_deg_mean']}/{c['dev_deg_max']}d {c['dev_cm_mean']}cm")
    except Exception as e: row.append('no meta')
    p=d+'/yam/flash/mink/report.json'
    if os.path.exists(p):
        r=json.load(open(p));row.append(f"mink tips {r['fingertip_cm']['mean']:.1f}cm")
    L=sorted(glob.glob(R+f'pipe/p{n}-*.log'),key=os.path.getmtime)
    if L:
        t=open(L[-1],errors='ignore').read();pb=re.findall(r'(mink|flash_pass|flash_harden_hardened) playback seed 42 (\{.*?\}) fr (\d+) pass ([\d.]+) cm ([\d.]+) deg ([\d.]+) rise ([\d.-]+)',t);row.append(' | '.join(f"{a[:7]}:{e}pass {x} rise {g}" for a,b,c,x,e2,f,g in [(q[0],q[1],q[2],q[3],q[4],q[5],q[6]) for q in pb] for e in ['']));
        if 'STEP_FAILED' in t: row.append(re.findall(r'STEP_FAILED \w+',t)[-1])
        s=re.findall(r'"stage": "(yam_passed|stalled|no_reachable_proposal)"',t);row.append('search:'+(s[-1] if s else '-'))
    print(' ; '.join(map(str,row)))
    for f in sorted(glob.glob(H+f'out/recon_{ep}/U-seed*/upper_updates.jsonl')):
        s=[]
        for l in open(f):
            u=json.loads(l);b=u['baseline'];s.append(f"{u['step']//10**6}M{'*' if u['accepted'] else ''}:{b['passed']:.2f}/{b['cm']:.2f}cm/{b['deg']:.0f}d")
        print('   upper',f.split('/')[-2],' '.join(s))
    for f in sorted(glob.glob(R+f'comparison/results/upper_test_recon_{ep}_s*.json')):
        for u in json.load(open(f)):
            if u['seed']==42:print('   test',u['steps']//10**6,'M',u['kinds'],'pass',u['passed'],f"{u['cm']:.2f}cm {u['deg']:.1f}d rise {u['object_max_rise_cm']:.1f}")
