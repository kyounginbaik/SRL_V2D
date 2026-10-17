"""Determine whether intersecting pot/lid visual patches are depth-observed."""
from pathlib import Path
import argparse,json
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT

def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def run(version):
    src=B.parent/'final_recon/episode_000001';cfg=json.loads((src/'config.json').read_text());fit=json.loads((OUT/version/'fit.json').read_text())
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text());T=np.array(support['proposed_setup_from_source'])
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    meshes=[]
    for path,pose in [(OUT/version/'white_pot_visual.obj',fit['pose']),(src/'assets/white_pot_lid_visual.obj',fit['unchanged_lid_pose'])]:
        mesh=trimesh.load(path,force='mesh',process=False);R=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
        mesh.vertices=(mesh.vertices@R.T+pose[:3]-shift)@T[:3,:3].T+T[:3,3];meshes.append(mesh)
    found=[]
    for height in np.arange(.075,.135,.00075):
        segs=[trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,height])[:,:,:2] for mesh in meshes]
        if not len(segs[0]) or not len(segs[1]):continue
        a,b=segs;p=a[:,None,0];q=b[None,:,0];r=a[:,None,1]-p;s=b[None,:,1]-q
        denominator=cross(r,s);safe=np.where(abs(denominator)>1e-12,denominator,np.nan)
        u=cross(q-p,s)/safe;v=cross(q-p,r)/safe;valid=(u>1e-6)&(u<1-1e-6)&(v>1e-6)&(v<1-1e-6)
        xy=(p+u[...,None]*r)[valid]
        found.extend(np.c_[xy,np.full(len(xy),height)])
    points=np.asarray(found);world=(points-T[:3,3])@T[:3,:3]
    camera=np.load(src/'camera_motion.npz')['world_from_rgb'][0];K=np.load(EP/'camera_trajectory.npz')['K']
    cam=(world-camera[:3,3])@camera[:3,:3];uv=cam@K.T;uv=uv[:,:2]/uv[:,2:3]
    rgb=cv2.imread(str(OUT/'source_000.jpg'));h,w=rgb.shape[:2]
    raw=cv2.imread(str(Path(cfg['depth'])/'000000.png'),-1)
    with np.errstate(divide='ignore',invalid='ignore'):depth=65535/raw.astype(float)-1
    depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
    pix=np.rint(uv).astype(int);inside=(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    discrepancy=np.full(len(points),np.nan);discrepancy[inside]=cam[inside,2]-depth[pix[inside,1],pix[inside,0]]
    visible=inside&np.isfinite(discrepancy)&(abs(discrepancy)<.004);behind=inside&np.isfinite(discrepancy)&(discrepancy>.004)
    for pixel,color in zip(pix,np.where(visible[:,None],np.array([0,0,255]),np.where(behind[:,None],np.array([0,255,255]),np.array([255,0,0])))):
        cv2.circle(rgb,tuple(pixel),3,tuple(map(int,color)),-1,cv2.LINE_AA)
    cv2.putText(rgb,'Actual visual intersections: red depth-visible; yellow behind observed surface',(500,700),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
    cv2.imwrite(str(OUT/version/'intersection_visibility.jpg'),rgb[700:1350,650:1450])
    result=dict(points=len(points),depth_visible_within_4mm=int(visible.sum()),behind_observed_surface=int(behind.sum()),
        depth_delta_quantiles_m=np.nanquantile(discrepancy,[0,.1,.5,.9,1]).tolist(),
        conclusion='Visibility test only. Hidden geometry is not measured ground truth; no mesh edits are made here.')
    (OUT/version/'intersection_visibility.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/version/'intersection_visibility.npz',points_source=world,pixels=pix,depth_delta_m=discrepancy,visible=visible)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',default='pot_geometry_rgb_depth_v3');run(p.parse_args().version)
