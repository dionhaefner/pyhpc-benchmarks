import math
import importlib


def generate_inputs(size):
    import numpy as np
    np.random.seed(17)

    shape = (
        math.ceil(2 * size ** (1/3)),
        math.ceil(2 * size ** (1/3)),
        math.ceil(0.25 * size ** (1/3)),
    )

    s = np.random.uniform(1e-2, 10, size=shape)
    t = np.random.uniform(-12, 20, size=shape)
    p = np.random.uniform(0, 1000, size=(1, 1, shape[-1]))
    return s, t, p


def try_import(backend):
    try:
        return importlib.import_module(f'.eos_{backend}', __name__)
    except ImportError:
        return None


def get_callable(backend, size):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    return lambda: backend_module.run(*inputs)


__implementations__ = (
    'bohrium',
    'jax',
    'numba',
    'numpy',
    'pytorch',
    'tensorflow',
    'theano',
)
