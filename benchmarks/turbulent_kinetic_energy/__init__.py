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
    maskU, maskV, maskW = ((np.random.rand(*shape) < 0.8).astype('float64') for _ in range(3))

    # 1d arrays
    dxt, dxu = (np.random.randn(shape[0]) for _ in range(2))
    dyt, dyu = (np.random.randn(shape[1]) for _ in range(2))
    dzt, dzw = (np.random.randn(shape[2]) for _ in range(2))
    cost, cosu = (np.random.randn(shape[1]) for _ in range(2))

    # 2d arrays
    kbot = np.random.randint(0, shape[2], size=shape[:2])
    forc_tke_surface = np.random.randn(*shape[:2])

    # 3d arrays
    kappaM, mxl, forc = (np.random.randn(*shape) for _ in range(3))

    # 4d arrays
    u, v, w, tke, dtke = (np.random.randn(*shape, 3) for _ in range(5))

    return (
        u, v, w,
        maskU, maskV, maskW,
        dxt, dxu, dyt, dyu, dzt, dzw,
        cost, cosu,
        kbot,
        kappaM, mxl, forc,
        forc_tke_surface,
        tke, dtke
    )


def try_import(backend):
    try:
        return importlib.import_module(f'.tke_{backend}', __name__)
    except ImportError:
        return None


def get_callable(backend, size, device='cpu'):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    if hasattr(backend_module, 'prepare_inputs'):
        inputs = backend_module.prepare_inputs(*inputs, device=device)
    return functools.partial(backend_module.run, *inputs, device=device)


__implementations__ = (
    'bohrium',
    'cupy',
    'numba',
    'numpy',
    'mxnet',
    'jax',
)
