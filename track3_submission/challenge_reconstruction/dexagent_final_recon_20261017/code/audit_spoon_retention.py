"""Summarize physical retention without mistaking ballistic height for grasp."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco

def audit(report_path):
    report=json.loads(report_path.read_text());root=report_path.parent/'execution'
    model=mujoco.MjModel.from_xml_path(str(root/'scene_physics.xml'))
    rec=np.load(root/'trajectory.npz');oc=int(model.joint('wooden_spoon_free').qposadr[0])
    t=rec['substep_time_s'];z=rec['substep_qpos'][:,oc+2]
    passive=float(z[(t>1.1)&(t<1.25)].mean())
    geometry=json.loads((root/'actual_contact_geometry.json').read_text())['samples']
    gt=np.array([r['time_seconds'] for r in geometry]);ok=np.array([r['pair'] is not None and r['pair']['geometry_qualified'] for r in geometry])
    ids=np.clip(np.searchsorted(t,gt),0,len(t)-1)
    events=report['execution']['feedback_events'];trans=next((r['time_seconds'] for r in events if r['phase']=='transport'),None)
    def ranges(mask):
        x=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(x==1),np.flatnonzero(x==-1)))
    blocks=ranges(ok & (gt>=trans)) if trans is not None else []
    longest=max(blocks,key=lambda ab:ab[1]-ab[0]) if blocks else None
    loss=None
    if trans is not None:
        for a,b in ranges(~ok & (gt>=trans)):
            if (b-a)*model.opt.timestep>=.06:loss=float(gt[a]);break
    proxy=report['metrics']['declared_frame_reference_proxy']
    out={'report':str(report_path),'physics_verified':report['execution']['physics_verified'],
         'proxy_mp_sr':proxy['mp_sr'],'objects':proxy['objects'],'transport_start_s':trans,
         'first_geometry_loss_sustained_60ms_after_transport_s':loss,
         'longest_geometry_qualified_interval_after_transport_s':[float(gt[longest[0]]),float(gt[longest[1]-1]+model.opt.timestep)] if longest else None,
         'peak_object_origin_lift_2_to_4s_m':float(z[(t>=2)&(t<=4)].max()-passive),
         'peak_object_origin_lift_while_geometry_qualified_m':float((z[ids][ok]-passive).max()) if np.any(ok) else None,
         'interpretation':'Antipodal pair is a contact diagnostic, not proof of stable full wrench support. Unqualified peak height may be ballistic.'}
    baseline=report_path.parents[2].parent/'contact_retention_measured/episode_000003/geometry/execution/trajectory.npz'
    if baseline.exists() and trans is not None:
        old=np.load(baseline);mask=t<=trans
        out['acquisition_prefix_qpos_bitwise_equal_measured_baseline']=bool(np.array_equal(rec['substep_qpos'][mask],old['substep_qpos'][mask]))
        out['acquisition_prefix_ctrl_bitwise_equal_measured_baseline']=bool(np.array_equal(rec['substep_ctrl'][mask],old['substep_ctrl'][mask]))
    (root/'retention_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reports',type=Path,nargs='+');a=p.parse_args()
    for path in a.reports:print(json.dumps(audit(path),indent=2))
