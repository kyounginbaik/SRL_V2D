import sys
sys.argv = ['x']
exec(open('/data1/hansol/retarget33/code/r33cupspoon.py').read().split('runs = [')[0])
for lab, tg in (('target 11/02 HGA (cwHGAfix)', K / 'targets/K33_cwHGAfix'), ('target latest r6', T / 'targets/R33_HGA_r6'), ('team v14 target', Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17'))):
    (pt, rt, ht, it), cg = target(tg, 348); dep = 100 * (cg['rim'] - ht)
    print(f'\n== {lab}: R {100*cg["R"]:.1f} cm height {100*(cg["rim"]-cg["bottom"]):.1f} cm; in-cup frames {int(it.sum())}; max depth below rim {dep.max():.1f} cm at f{int(dep.argmax())}')
    print('  ' + ' '.join(f'f{f}:{"IN" if it[f] else "--"}{dep[f]:+.1f}/{100*rt[f]:.1f}' for f in range(60, 300, 6)))
