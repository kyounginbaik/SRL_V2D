import numpy as np, json
X='/data1/hansol/ep33_20261102/xfer/ep33'
CAN='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
def show(p):
    z=np.load(p, allow_pickle=True); print('==', p.split('/')[-2:], {k:(z[k].shape, str(z[k].dtype)) for k in z.files})
    return z
op=show(X+'/bundle/episode_000033/object_poses.npz'); print('names', op['names'], 'fps', op['fps'])
W=op['world_from_rgb']; print('cam0 pos', W[0,:3,3].round(3), 'cam0 z-axis(view) in world', W[0,:3,2].round(3), 'cam0 y-axis(down) in world', W[0,:3,1].round(3))
wf=op['world_from_object']; print('spoon pos f0', wf[0,1,:3,3].round(3) if wf.shape[1]>1 else None, 'cup pos f0', wf[0,0,:3,3].round(3))
print('object z range (world z)', wf[:,:,2,3].min().round(3), wf[:,:,2,3].max().round(3))
hw=show(X+'/mink/hands_world.npz')
for k in hw.files:
    if hw[k].dtype!=object and hw[k].size<20: print(' ', k, hw[k])
sq=show(X+'/mink/sharpa_ik_qpos.npz')
hp=show(X+'/canonical/hand_motion_prior_v2.npz')
sp=show(CAN+'/support_plane.npz'); print({k: sp[k] for k in sp.files if sp[k].size<10})
print(json.load(open(CAN+'/support_plane_report.json')) if __import__('os').path.exists(CAN+'/support_plane_report.json') else '')
cam=show(X+'/canonical/camera.npz')
