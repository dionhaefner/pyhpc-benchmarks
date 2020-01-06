import math
import importlib
import functools


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


def import_backend(backend):
    return importlib.import_module(f'.eos_{backend}', __name__)


def get_callable(backend, size, device='cpu'):
    backend_module = import_backend(backend)
    inputs = generate_inputs(size)
    if hasattr(backend_module, 'prepare_inputs'):
        inputs = backend_module.prepare_inputs(*inputs, device=device)
    return functools.partial(backend_module.run, *inputs, device=device)


__implementations__ = (
    'bohrium',
    'cupy',
    'jax',
    'mxnet',
    'numba',
    'numpy',
    'pytorch',
    'tensorflow',
    'theano',
)
