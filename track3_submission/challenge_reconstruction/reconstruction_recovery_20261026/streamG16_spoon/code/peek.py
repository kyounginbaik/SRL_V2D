import sys, glob, numpy as np
G='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG16_spoon'
EP=int(sys.argv[1]); a,b=int(sys.argv[2]),int(sys.argv[3])
Z=[np.load(f) for f in sorted(glob.glob(f'{G}/out/scan_ep{EP}_*.npz'))]; cat=lambda k: np.concatenate([z[k] for z in Z])
ball,iou,dz,nb,hand,mpx=(cat(k) for k in ('ball','iou','dz','nb','hand','mask_px')); thb=Z[0]['thb']; thi=Z[0]['thi']
A=np.load(glob.glob(f'{G}/out/ana_ep{EP}_l0.05_p20_b2_r1.npz')[0]); OFF=float(A['offset']); jA=int(A['jA']); H=A['H']
ji=int(np.argmin(np.abs((thi-OFF+180)%360-180)))
for f in range(a,b+1):
    B=ball[f].reshape(-1); j=np.nanargmin(B) if np.isfinite(B).any() else -1
    b0=np.nanargmin(ball[f,0]) if np.isfinite(ball[f,0]).any() else -1
    print(f, 'nb',nb[f],'hand',hand[f],'mask',mpx[f], 'pxA %.1f'%B[jA] if j>=0 else '', 'best', (tuple(H[j]), '%.1f'%B[j]) if j>=0 else '', 'bestflip0', (thb[b0], '%.1f'%ball[f,0,b0]) if b0>=0 else '',
          'iouA %.2f'%iou[f,0,ji], 'iou0max %.2f@%d'%(np.nanmax(iou[f,0]), thi[np.nanargmax(iou[f,0])]) if np.isfinite(iou[f,0]).any() else '', 'iou1max %.2f@%d'%(np.nanmax(iou[f,1]), thi[np.nanargmax(iou[f,1])]) if np.isfinite(iou[f,1]).any() else '',
          'dzA %.1f'%dz[f,0,ji], 'path',tuple(H[A['path'][f]]))
