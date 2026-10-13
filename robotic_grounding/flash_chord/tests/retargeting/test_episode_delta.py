"""The single-chunk probe must truly have one shared temporal coefficient."""
import numpy as np
from flash_chord.retargeting.granularity import granularity_basis


def test_single_chunk_is_episode_basis():
    episode = granularity_basis(335, 'episode', fps=20)
    chunk = granularity_basis(335, 'chunk', fps=20, chunk_seconds=60)
    np.testing.assert_array_equal(episode, chunk)
    assert chunk.shape == (335, 1)
    np.testing.assert_array_equal(chunk[[0, -1]], 0)
    np.testing.assert_array_equal(chunk[5:-5], 1)


def test_factorial_dimensions():
    episode = granularity_basis(335, 'episode', fps=20)
    chunks = granularity_basis(335, 'chunk', fps=20, chunk_seconds=1)
    assert [b.shape[1]*d for b in (episode, chunks) for d in (12, 56)] == [12,56,204,952]
