import numpy as np

from src.faiss_index import l2_normalize


def test_l2_normalize_unit_length():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype="float32")
    normed = l2_normalize(x)
    norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_l2_normalize_handles_zero_vector_without_crashing():
    x = np.array([[0.0, 0.0]], dtype="float32")
    normed = l2_normalize(x)
    assert not np.isnan(normed).any()


def test_l2_normalize_preserves_direction():
    x = np.array([[2.0, 0.0]], dtype="float32")
    normed = l2_normalize(x)
    assert np.allclose(normed, [[1.0, 0.0]])
