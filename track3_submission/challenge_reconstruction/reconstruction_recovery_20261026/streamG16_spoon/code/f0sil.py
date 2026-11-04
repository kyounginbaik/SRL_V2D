import glob, numpy as np, json
G='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG16_spoon'
out={}
for EP in [3,5,6,15,16,32,33,35,36,37]:
    Z=[np.load(f) for f in sorted(glob.glob(f'{G}/out/scan_ep{EP}_*.npz'))]; iou=np.concatenate([z['iou'][:,0] for z in Z])[:24]; dz=np.concatenate([z['dz'][:,0] for z in Z])[:24]; thi=Z[0]['thi']
    idx=[list(thi).index(t) for t in (330,345,0,15,30)]
    out[EP]=dict(iou=np.round(np.nanmean(iou[:,idx],0),3).tolist(), dz_mm=np.round(np.nanmedian(dz[:,idx],0),2).tolist())
    print(EP,'roll -30,-15,0,15,30  IoU',out[EP]['iou'],' depth mm',out[EP]['dz_mm'])
json.dump(out,open(f'{G}/out/f0_sil_depth.json','w'),indent=1)
