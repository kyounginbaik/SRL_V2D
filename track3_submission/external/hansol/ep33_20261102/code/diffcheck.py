import numpy as np
for b,m in (("HGA","HGAR8s15"),("HGB","HGBR8s15")):
    a=np.load(f"/home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_{b}/episode_000033/object_poses.npz")
    n=np.load(f"/data1/hansol/ep33_20261102/roll/recon_E14_best_B2A_S1_{m}/episode_000033/object_poses.npz")
    names=[str(x) for x in a['names']]; d=np.abs(a['world_from_object']-n['world_from_object']).max(axis=(2,3))
    s=names.index('wooden_spoon'); c=names.index('blue_cup'); ch=np.where(d[:,s]>0)[0]
    print(m, "f0 identical:", bool(d[0].max()==0), "| spoon changed stored frames", int(ch.min()), "-", int(ch.max()), "n", len(ch), "| cup max diff", float(d[:,c].max()), "| other keys same:", all(np.array_equal(a[k],n[k]) for k in a.files if k not in ('world_from_object','camera_from_object')))
