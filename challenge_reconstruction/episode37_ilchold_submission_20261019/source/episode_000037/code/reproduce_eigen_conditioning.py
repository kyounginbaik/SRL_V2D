from pathlib import Path
import json
import numpy as np,jax.numpy as jnp
from flash_chord.training.flash_sac.eigen import correlated_scale_tril
b=Path(__file__).resolve().parents[1];basis=np.load(b/'priors/sharpa_bimanual_residual_basis.npz')['factor'];rows=[]
for eigen_std in [.07,.1,.3,1.]:
 for std in [.01,.001,.0001,float(np.exp(-10)*np.sqrt(.5))]:
  L=np.asarray(correlated_scale_tril(jnp.full((1,56),std),jnp.asarray(basis*eigen_std)))
  rows.append({'eigen_std':eigen_std,'diagonal_std':std,'finite':bool(np.isfinite(L).all())})
print(json.dumps(rows));(b/'eigen_conditioning_reproduction.json').write_text(json.dumps(rows,indent=2))
