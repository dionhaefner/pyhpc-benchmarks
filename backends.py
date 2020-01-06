import os
import importlib

import numpy


def convert_to_numpy(arr, backend, device='cpu'):
    """Converts an array or collection of arrays to np.ndarray"""
    if isinstance(arr, (list, tuple)):
        return [convert_to_numpy(subarr, backend, device) for subarr in arr]

    if backend == 'bohrium':
        arr = arr.copy2numpy()

    if backend == 'cupy':
        arr = arr.get()

    if backend == 'jax':
        arr = numpy.asarray(arr)

    if backend == 'pytorch':
        if device == 'gpu':
            arr = numpy.asarray(arr.cpu())
        else:
            arr = numpy.asarray(arr)

    if backend == 'tensorflow':
        arr = numpy.asarray(arr)

    if backend == 'theano':
        arr = numpy.asarray(arr)

    if backend == 'mxnet':
        arr = arr.asnumpy()

    if type(arr) is not numpy.ndarray:
        raise RuntimeError(f'Failed to copy array to NumPy (backend {backend}, type {type(arr)})')

    return arr


class BackendNotSupported(Exception):
    pass


class SetupContext:
        def __init__(self, f):
            self._f = f
            self._f_args = (tuple(), dict())

        def __call__(self, *args, **kwargs):
            self._f_args = (args, kwargs)
            return self

        def __enter__(self):
            self._env = os.environ.copy()
            args, kwargs = self._f_args
            self._f_iter = iter(self._f(*args, **kwargs))

            try:
                next(self._f_iter)
            except Exception as e:
                raise BackendNotSupported(str(e)) from None

            return self

        def __exit__(self, *args, **kwargs):
            try:
                next(self._f_iter)
            except StopIteration:
                pass
            os.environ = self._env


setup_function = SetupContext


# setup function definitions

@setup_function
def setup_numpy(device='cpu'):
    os.environ.update(
        OMP_NUM_THREADS='1',
    )
    yield


@setup_function
def setup_bohrium(device='cpu'):
    os.environ.update(
        OMP_NUM_THREADS='1',
        BH_STACK='opencl' if device == 'gpu' else 'openmp',
    )
    try:
        import bohrium  # noqa: F401
        yield
    finally:
        # bohrium does things to numpy
        importlib.reload(numpy)


@setup_function
def setup_theano(device='cpu'):
    os.environ.update(
        OMP_NUM_THREADS='1',
    )
    if device == 'gpu':
        os.environ.update(
            THEANO_FLAGS='device=cuda',
        )
    import theano  # noqa: F401
    yield


@setup_function
def setup_numba(device='cpu'):
    os.environ.update(
        OMP_NUM_THREADS='1',
    )
    import numba  # noqa: F401
    yield


@setup_function
def setup_cupy(device='cpu'):
    if device != 'gpu':
        raise RuntimeError('cupy requires GPU mode')
    import cupy  # noqa: F401
    yield


@setup_function
def setup_jax(device='cpu'):
    os.environ.update(
        XLA_FLAGS=(
            '--xla_cpu_multi_thread_eigen=false '
            'intra_op_parallelism_threads=1 '
            'inter_op_parallelism_threads=1 '
        ),
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
    )

    if device in ('cpu', 'gpu'):
        os.environ.update(JAX_PLATFORM_NAME=device)

    import jax
    from jax.config import config

    if device == 'tpu':
        config.update('jax_xla_backend', 'tpu_driver')
        config.update('jax_backend_target', os.environ.get('JAX_BACKEND_TARGET'))

    if device != 'tpu':
        # use 64 bit floats (not supported on TPU)
        config.update('jax_enable_x64', True)

    if device == 'gpu':
        assert len(jax.devices()) > 0

    yield


@setup_function
def setup_pytorch(device='cpu'):
    os.environ.update(
        OMP_NUM_THREADS='1',
    )
    import torch
    if device == 'gpu':
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0
    yield


@setup_function
def setup_tensorflow(device='cpu'):
    os.environ.update(
        OMP_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
    )
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    if device == 'gpu':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        assert gpus
    else:
        tf.config.experimental.set_visible_devices([], 'GPU')
    yield


@setup_function
def setup_mxnet(device='cpu'):
    from mxnet import npx
    npx.set_np()

    if device == 'gpu':
        assert npx.num_gpus() > 0

    yield


__backends__ = {
    'numpy': setup_numpy,
    'bohrium': setup_bohrium,
    'cupy': setup_cupy,
    'jax': setup_jax,
    'theano': setup_theano,
    'numba': setup_numba,
    'pytorch': setup_pytorch,
    'tensorflow': setup_tensorflow,
    'mxnet': setup_mxnet,
}
