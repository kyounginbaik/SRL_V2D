"""Identify native wrist translation response with previous displacement.

Row i holds command i and its AFTER pose. This is a local diagonal ARX
model, not a contact model or an end-to-end reference/policy gradient.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionResponse:
    gain: np.ndarray
    momentum: np.ndarray
    offset: np.ndarray

    def __post_init__(self):
        for name in ('gain','momentum','offset'):
            value=np.array(getattr(self,name),float,copy=True)
            if value.shape!=(3,) or not np.isfinite(value).all():
                raise ValueError('three finite coefficients per WORLD position axis required')
            value.flags.writeable=False;object.__setattr__(self,name,value)
        if np.any(self.gain<=0):raise ValueError('positive command sensitivity required')
        for a,b in zip(self.gain,self.momentum):
            if np.max(np.abs(np.roots([1.,-(1-a+b),b])))>=1:
                raise ValueError('unforced position response must be asymptotically stable')

    @staticmethod
    def _values(*values):
        arrays=np.broadcast_arrays(*[np.asarray(v,float) for v in values])
        if arrays[0].ndim<1 or arrays[0].shape[-1]!=3 or any(not np.isfinite(x).all() for x in arrays):
            raise ValueError('matching finite WORLD XYZ arrays required')
        return arrays

    def predict(self,previous,before,command):
        previous,before,command=self._values(previous,before,command)
        return before+self.gain*(command-before)+self.momentum*(before-previous)+self.offset

    def inverse(self,previous,before,after):
        previous,before,after=self._values(previous,before,after)
        return before+(after-before-self.momentum*(before-previous)-self.offset)/self.gain


def _transitions(actual_after,commands,supported):
    h,c=np.asarray(actual_after,float),np.asarray(commands,float);support=np.asarray(supported)
    if h.ndim!=3 or h.shape[-1]!=3 or c.shape!=h.shape or support.shape!=h.shape[:2] or support.dtype!=bool or len(h)<4:
        raise ValueError('matched [post-control-frame,world,3] positions and Boolean support required')
    finite=np.isfinite(h).all(-1)&np.isfinite(c).all(-1)
    good=support[2:]&support[1:-1]&support[:-2]&finite[2:]&finite[1:-1]&finite[:-2]
    return (h[:-2][good],h[1:-1][good],h[2:][good],c[2:][good],
        dict(total_transitions=int(good.size),used_transitions=int(good.sum()),
            supported_transition_fraction=float(good.mean()),invalid_position_fraction=float((~finite).mean()),
            row_semantics='command i / after i; before=after i-1; previous=after i-2'))


def fit_position_response(actual_after,commands,supported):
    previous,before,after,command,report=_transitions(actual_after,commands,supported)
    if len(before)<18:return None,dict(report,reason='insufficient_supported_transitions')
    coefficients=[]
    for axis in range(3):
        design=np.c_[command[:,axis]-before[:,axis],before[:,axis]-previous[:,axis],np.ones(len(before))]
        # Normalize columns before checking rank: translation and intercept
        # have different units, and position/velocity must be independently excited.
        spread=design[:,:2].std(0)
        if np.any(spread<1e-5):return None,dict(report,reason='insufficient_axis_excitation',axis=axis)
        normalized=np.c_[(design[:,:2]-design[:,:2].mean(0))/spread,np.ones(len(before))]
        if np.linalg.cond(normalized)>1e4:return None,dict(report,reason='unidentifiable_position_velocity_response',axis=axis)
        coefficients.append(np.linalg.lstsq(design,after[:,axis]-before[:,axis],rcond=None)[0])
    values=np.asarray(coefficients)
    try:model=PositionResponse(values[:,0],values[:,1],values[:,2])
    except ValueError as exc:return None,dict(report,reason=str(exc))
    return model,dict(report,reason='identified_local_inertial_position_response')


def validate_position_response(model,actual_after,commands,supported):
    previous,before,after,command,report=_transitions(actual_after,commands,supported)
    if not len(before):return dict(report,reason='no_supported_validation_transitions')
    prediction=model.predict(previous,before,command)
    error=np.linalg.norm(prediction-after,axis=-1)
    return dict(report,reason='one_step_prediction_only',position_rmse_m=float(np.sqrt(np.mean(error**2))),
        position_p90_m=float(np.percentile(error,90)))
