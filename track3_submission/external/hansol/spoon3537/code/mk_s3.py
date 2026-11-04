"""ep36 tgt_s3 = tgt_s1 with the spoon roll re-fitted over a late window (agent front/back marks + marker balls + IoU, GT-free,
no human marks). The roll fit runs spoon_annot_roll3 (free end) on the FINAL bundle (v14_g17); its per-frame object-frame
correction Rel = inv(W_old) @ W_new is applied on the right of tgt_s1. usage: mk_s3.py FIT_OUT_DIR"""
import sys, json, shutil, numpy as np
from pathlib import Path
E = 'episode_000036'
B = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17') / E
FIT = Path(sys.argv[1]) / 'bundle' / E
S1 = Path('/data1/hansol/spoon3537/targets/tgt_s1') / E; S3 = Path('/data1/hansol/spoon3537/targets/tgt_s3') / E
a = np.load(B / 'object_poses.npz')['world_from_object'][:, 0]; b = np.load(FIT / 'object_poses.npz')['world_from_object'][:, 0]
Rel = np.einsum('tij,tjk->tik', np.linalg.inv(a), b)
ch = np.where(np.abs(Rel - np.eye(4)).max(axis=(1, 2)) > 1e-9)[0]
T1 = np.load(S1 / 'wooden_spoon_trajectory.npz')['T_world_obj']; T3 = np.einsum('tij,tjk->tik', T1, Rel)
if S3.exists(): shutil.rmtree(S3)
shutil.copytree(S1, S3); np.savez_compressed(S3 / 'wooden_spoon_trajectory.npz', T_world_obj=T3)
ang = np.degrees(np.arccos(np.clip((np.trace(Rel[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
print(json.dumps(dict(target=str(S3), frames_changed=[int(ch.min()), int(ch.max()), int(len(ch))] if len(ch) else [], max_roll_change_deg=float(ang.max()),
                      roll_change_by_second={f'{t / 20:.1f}': round(float(ang[t]), 1) for t in range(0, len(ang), 10)})))
