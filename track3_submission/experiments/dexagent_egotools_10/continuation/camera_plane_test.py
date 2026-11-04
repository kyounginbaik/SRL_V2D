import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from tool_library.common.rigcam import R_cv_from_angles
from tool_library.skills.fit_camera_to_table_plane_cv import fit_camera_to_table_plane_cv

for pitch,roll,height in [(30.,0.,.55),(55.,12.,.7),(70.,-10.,.9)]:
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory);(path/'masks').mkdir()
        h=w=96;focal=100.
        up=R_cv_from_angles(90.,pitch,roll)[2,:]
        yy,xx=np.indices((h,w));rays=np.stack([(xx-w/2)/focal,(yy-h/2)/focal,np.ones((h,w))],-1)
        depths=-height/(rays@up)
        depths[depths<=0]=0
        np.savez(path/'depth.npz',depths=depths[None].astype(np.float32))
        mask=np.zeros((h,w),np.uint8);mask[38:58,38:58]=1
        Image.fromarray(mask).save(path/'masks/frame00000.png')
        fit=fit_camera_to_table_plane_cv(path/'depth.npz',path/'masks',focal,n_frames=1,stride=2,grow=15,keep_out=3,tol=.001)
        assert fit is not None
        assert abs(fit['pitch_deg']-pitch)<.01,(fit,pitch)
        assert abs(fit['roll_deg']-roll)<.01,(fit,roll)
        assert abs(fit['camera_height_m']-height)<.0002
        assert np.linalg.norm(np.asarray(fit['up'])-up)<1e-5
        print(f'[PASS] depth plane pitch {pitch} roll {roll} height {height}: orientation and metric height recovered')
