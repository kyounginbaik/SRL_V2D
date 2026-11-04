"""Bounded registered lateral-seat audit; immutable observed lid and table."""
import json
import cv2,mujoco,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT

def run():
    cfg=json.loads((B.parent/'final_recon/episode_000001/config.json').read_text())
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text());tangent=np.array(support['proposed_setup_from_source'])[:2,:3]
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    obs=np.load(OUT/'multiframe_clouds/white_pot_000000.npz');pc=obs['points_source']+shift;pixels=obs['pixels'];rgb=cv2.imread(str(OUT/'multiframe_clouds/rgb_000000.jpg'))
    lm=cv2.imread(cfg['object_masks']['white_pot_lid']+'/000000.png',0);lm=cv2.dilate(cv2.resize(lm,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST),np.ones((5,5),np.uint8))
    valid=lm[pixels[:,1],pixels[:,0]]==0;pc=pc[valid][::6]
    out=OUT/'seated_xy_audit';out.mkdir(exist_ok=True);reports=[]
    for height in [.86,.90,.94,.98]:
        suffix=f'_height{round(height*100):03d}';version='pot_geometry_rgb_depth'+suffix;fit=json.loads((OUT/version/'fit.json').read_text())
        scene=OUT/('warped32'+suffix)/'scene_repaired.xml';m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
        addr=int(m.joint('white_pot_free').qposadr[0]);initial=d.qpos[addr:addr+3].copy();R=Rotation.from_quat(d.qpos[addr+3:addr+7],scalar_first=True).as_matrix()
        mesh=trimesh.load(OUT/version/'white_pot_visual.obj',force='mesh',process=False);sample,_=trimesh.sample.sample_surface(mesh,100000,seed=47);tree=cKDTree(sample)
        ids={m.body('white_pot').id,m.body('white_pot_lid').id};rows=[]
        for x in np.linspace(-.012,.012,17):
            for y in np.linspace(-.012,.012,17):
                p=initial+np.array([x,y])@tangent;d.qpos[addr:addr+3]=p;mujoco.mj_forward(m,d)
                pen=max([max(0.,-float(c.dist)) for c in d.contact if {int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])}==ids]+[0.])
                dist=tree.query((pc-p)@R)[0];rows.append(dict(x=float(x),y=float(y),penetration_m=pen,median_stereo_m=float(np.median(dist)),p80_stereo_m=float(np.quantile(dist,.8))))
        feasible=[r for r in rows if r['penetration_m']<.0002]
        best=min(feasible,key=lambda r:r['p80_stereo_m']) if feasible else min(rows,key=lambda r:r['penetration_m'])
        report=dict(height_scale=height,feasible_count=len(feasible),best=best,all=rows);reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='all'}),flush=True)
        (out/'report.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':run()
