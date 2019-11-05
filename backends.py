import os
import importlib

import numpy


def convert_to_numpy(arr, backend, gpu=False):
    if isinstance(arr, numpy.ndarray):
        return arr

    if backend == 'bohrium':
        return arr.copy2numpy()

    if backend == 'cupy':
        return arr.get()

    if backend == 'jax':
        return numpy.asarray(arr)

    if backend == 'pytorch':
        if gpu:
            return numpy.asarray(arr.cpu())
        else:
            return numpy.asarray(arr)

    if backend == 'tensorflow':
        return numpy.asarray(arr)

    if backend == 'theano':
        return numpy.asarray(arr)

    raise RuntimeError(f'Got unexpected array / backend combination: {type(arr)} / {backend}')


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
def setup_numpy(gpu=False):
    import numpy
    yield


@setup_function
def setup_bohrium(gpu=False):
    try:
        os.environ.update(
            OMP_NUM_THREADS='1',
            BH_STACK='opencl' if gpu else 'openmp',
            NUMPY_EXPERIMENTAL_ARRAY_FUNCTION='1',
        )
        import bohrium
        yield
    finally:
        # bohrium does things to numpy
        importlib.reload(numpy)


@setup_function
def setup_theano(gpu=False):
    os.environ.update(
        OMP_NUM_THREADS='1',
    )
    if gpu:
        os.environ.update(
            THEANO_FLAGS='device=cuda',
        )
    import theano
    yield


@setup_function
def setup_numba(gpu=False):
    import numba
    yield


@setup_function
def setup_cupy(gpu=False):
    if not gpu:
        raise RuntimeError('cupy requires GPU mode')
    import cupy
    yield


@setup_function
def setup_jax(gpu=False):
    os.environ.update(
        XLA_FLAGS=(
            '--xla_cpu_multi_thread_eigen=false '
            'intra_op_parallelism_threads=1 '
            'inter_op_parallelism_threads=1 '
        ),
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAX_PLATFORM_NAME='gpu' if gpu else 'cpu',
    )

    import jax
    from jax.config import config
    # use 64 bit floats
    config.update('jax_enable_x64', True)

    if gpu:
        assert len(jax.devices()) > 0

    yield


@setup_function
def setup_pytorch(gpu=False):
    os.environ.update(
        OMP_NUM_THREADS='1',
    )
    import torch
    if gpu:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0
    yield


@setup_function
def setup_tensorflow(gpu=False):
    os.environ.update(
        OMP_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
    )
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    if gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        assert gpus
    else:
        tf.config.experimental.set_visible_devices([], 'GPU')
    yield


__backends__ = {
    'numpy': setup_numpy,
    'bohrium': setup_bohrium,
    'cupy': setup_cupy,
    'jax': setup_jax,
    'theano': setup_theano,
    'numba': setup_numba,
    'pytorch': setup_pytorch,
    'tensorflow': setup_tensorflow
}
