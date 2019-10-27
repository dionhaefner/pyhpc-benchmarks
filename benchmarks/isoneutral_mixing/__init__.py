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

    # masks
    maskT, maskU, maskV, maskW = ((np.random.rand(*shape) < 0.8).astype('float64') for _ in range(4))

    # 1d arrays
    dxt, dxu = (np.random.rand(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.rand(shape[1]) for _ in range(2))
    dzt, dzw, zt = (np.random.rand(shape[2]) for _ in range(3))
    cost, cosu = (np.random.rand(shape[1]) for _ in range(2))

    # 3d arrays
    K_iso, K_iso_steep, K_11, K_22, K_33 = (np.random.rand(*shape) for _ in range(5))

    # 4d arrays
    salt, temp = (np.random.rand(*shape, 3) for _ in range(2))

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


def get_callable(backend, size):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    return lambda: backend_module.run(*inputs)


__implementations__ = (
    'bohrium',
    'numba',
    'numpy',
    'jax',
    'pytorch',
    # 'tensorflow',
    'theano',
)
