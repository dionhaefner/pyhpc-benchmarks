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

    # masks
    maskT, maskU, maskV, maskW = ((np.random.rand(*shape) < 0.8).astype('float64') for _ in range(4))

    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw, zt = (np.random.randn(shape[2]) for _ in range(3))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))

    # 3d arrays
    K_iso, K_iso_steep, K_11, K_22, K_33 = (np.random.randn(*shape) for _ in range(5))

    # 4d arrays
    salt, temp = (np.random.randn(*shape, 3) for _ in range(2))

    # 5d arrays
    Ai_ez, Ai_nz, Ai_bx, Ai_by = (np.zeros((*shape, 2, 2)) for _ in range(4))

    return (
        maskT, maskU, maskV, maskW,
        dxt, dxu, dyt, dyu, dzt, dzw,
        cost, cosu,
        salt, temp, zt,
        K_iso, K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by
    )


def try_import(backend):
    try:
        return importlib.import_module(f'.isoneutral_{backend}', __name__)
    except ImportError:
        return None


def get_callable(backend, size, gpu=False):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    return functools.partial(backend_module.run, *inputs, gpu=gpu)


__implementations__ = (
    'bohrium',
    'numba',
    'numpy',
    'jax',
    'pytorch',
    # 'tensorflow',
    'theano',
)
