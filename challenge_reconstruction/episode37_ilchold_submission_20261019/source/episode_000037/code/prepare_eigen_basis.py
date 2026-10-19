from pathlib import Path
import json,hashlib,subprocess
import numpy as np,yaml,newton
from safetensors.numpy import load_file
from flash_chord.embodiments.sharpa_hands import SharpaHands
b=Path(__file__).resolve().parents[1];v=b/'vendor/eigendexplore';src=v/'src/action_bench/assets/sharpa'
m=yaml.safe_load((src/'manifest.yaml').read_text());payload=load_file(str(src/'pca.safetensors'));print('payload',[(k,a.shape) for k,a in payload.items()],flush=True)
layout=SharpaHands().build(newton.ModelBuilder())
factor=np.zeros((56,18),np.float32);orders={}
for side,start,col in [('right',6,0),('left',34,9)]:
 hand=layout.hand(side);names=list(hand.finger_joint_names);deployment=next(d for d in m['deployment_layouts'] if d['hand_layout']==f'sharpa_{side}')
 source=deployment['joint_names'];assert set(names)==set(source) and len(names)==22
 permutation=[source.index(n) for n in names];sign=np.asarray(deployment['joint_signs'])
 block=(payload['components'][:9]*sign[None,:])[:,permutation].T*payload['component_std'][:9][None,:]/.15
 block*=np.sqrt(22/np.square(block).sum());factor[start:start+22,col:col+9]=block
 orders[side]=names
out=b/'priors';out.mkdir(exist_ok=True);np.savez(out/'sharpa_bimanual_residual_basis.npz',factor=factor)
provenance={'source':'https://github.com/hgupt3/eigendexplore','commit':subprocess.check_output(['git','-C',str(v),'rev-parse','HEAD'],text=True).strip(),'manifest':m,'joint_order':orders,'rank_per_hand':9,'action_order':'right wrist6/fingers22 then left wrist6/fingers22','residual_radians_per_normalized_unit_at_zero':.15,'normalization':'each hand squared Frobenius norm 22','basis_sha256':hashlib.sha256((out/'sharpa_bimanual_residual_basis.npz').read_bytes()).hexdigest()}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2));print('BASIS_COMPLETE',provenance['basis_sha256'])
