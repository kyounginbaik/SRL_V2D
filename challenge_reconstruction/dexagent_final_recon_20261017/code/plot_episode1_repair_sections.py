"""Compare visual surfaces and convex pieces in the measured table frame."""
from pathlib import Path
import json
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,OUT,EP,load

m,d,camera,meshes=load()
support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text())
T=np.array(support['proposed_setup_from_source']);shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
report=json.loads((OUT/'pot_table_candidate.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(13,10))
for row,(label,potpose) in enumerate([('Original reconstruction',report['original_pot_pose']),('Pot corrected to stereo table',report['candidate_pot_pose'])]):
    for name,color in [('white_pot','tab:blue'),('white_pot_lid','tab:red')]:
        pose=np.array(potpose if name=='white_pot' else report['unchanged_lid_pose']);R=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
        paths=[B.parent/f'final_recon/episode_000001/assets/{name}_visual.obj',*sorted((B.parent/f'final_recon/episode_000001/assets/{name}_convex').glob('*.obj'))]
        for k,path in enumerate(paths):
            mesh=trimesh.load(path,force='mesh',process=False)
            mesh.vertices=(mesh.vertices@R.T+pose[:3]-shift)@T[:3,:3].T+T[:3,3]
            for col,axis in enumerate([0,1]):
                normal=np.eye(3)[axis];segs=trimesh.intersections.mesh_plane(mesh,normal,[0,0,0])
                if len(segs):axs[row,col].add_collection(LineCollection(segs[:,:,[1-axis,2]]*100,colors=color,linewidths=1.6 if k==0 else .6,alpha=1 if k==0 else .25))
    for col in range(2):
        ax=axs[row,col];ax.set_aspect('equal');ax.set_xlim(-12,12);ax.set_ylim(-2,16);ax.axhline(0,color='black');ax.grid(alpha=.2)
        ax.set_title(f'{label}: {"Y" if col==0 else "X"}/height section');ax.set_ylabel('height from measured tabletop (cm)');ax.set_xlabel('horizontal coordinate (cm)')
fig.suptitle('Exact source visual surfaces (bold) and convex collision pieces (faint): pot blue, lid red')
fig.tight_layout();fig.savefig(OUT/'table_frame_vertical_sections.png',dpi=150)
