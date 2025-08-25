import numpy as np
from snakegame import model
from snakegame import config as C

def test_forward_shapes():
    x = np.zeros((5, C.N_X), dtype=np.float32)
    g = model.random_genome(np.random.default_rng(0))
    y = model.forward(x, g)
    assert y.shape == (5, C.N_Y)
    assert np.allclose(y.sum(axis=1), 1, atol=1e-5)
