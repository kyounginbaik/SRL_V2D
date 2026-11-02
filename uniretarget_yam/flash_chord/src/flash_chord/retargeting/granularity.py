"""Matched temporal parameterizations for robot-reference corrections."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def granularity_basis(frames, variant, *, fps, chunk_seconds=.5, smoothing_seconds=.1,
                      boundary_seconds=.25):
    """One shared coefficient, one per frame, or one per fixed time chunk.

    All variants use the same time-domain smoothing and endpoint envelope.
    Chunks are fixed by time, independent of motion/contact labels. The basis
    covers interior frames; no coefficient is allocated solely to an endpoint.
    """
    if frames < 3 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('at least three frames and positive finite fps required')
    if not all(np.isfinite(v) and v >= 0 for v in (smoothing_seconds, boundary_seconds)):
        raise ValueError('smoothing and boundary durations must be finite and nonnegative')
    interior = frames - 2
    if variant == 'episode':
        labels = np.zeros(interior, dtype=int)
    elif variant == 'timestep':
        labels = np.arange(interior)
    elif variant == 'chunk':
        if not np.isfinite(chunk_seconds) or chunk_seconds <= 0:
            raise ValueError('positive finite chunk duration required')
        labels = np.arange(interior) // max(1, int(round(chunk_seconds * fps)))
    else:
        raise ValueError(f'unknown temporal granularity {variant}')
    basis = np.zeros((frames, int(labels[-1])+1), dtype=np.float32)
    basis[np.arange(1, frames-1), labels] = 1
    # Extend endpoint values before applying the same filter to every variant.
    basis[0] = basis[1]
    basis[-1] = basis[-2]
    if smoothing_seconds > 0:
        basis = gaussian_filter1d(basis, smoothing_seconds*fps, axis=0, mode='nearest')
    ramp = max(1, min(int(round(boundary_seconds*fps)), (frames-1)//2))
    distance = np.minimum(np.arange(frames), np.arange(frames)[::-1])
    envelope = .5-.5*np.cos(np.pi*np.minimum(distance/ramp, 1))
    return (basis*envelope[:,None]).astype(np.float32)
