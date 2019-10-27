import os
import contextlib
import importlib

import numpy


class BackendNotSupported(Exception):
    pass


def try_import(module):
    try:
        importlib.import_module(module)
    except ImportError:
        raise BackendNotSupported()


def setup_function(f):
    return contextlib.contextmanager(f)


# setup function definitions

@setup_function
def setup_numpy():
    try_import('numpy')
    yield


@setup_function
def setup_bohrium():
    oldenv = os.environ.get('OMP_NUM_THREADS')
    try:
        os.environ.update(
            OMP_NUM_THREADS='1',
            BH_STACK='openmp',
            NUMPY_EXPERIMENTAL_ARRAY_FUNCTION='1',
        )
        try_import('bohrium')
        yield
    finally:
        if oldenv is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = oldenv
        # bohrium does things to numpy
        importlib.reload(numpy)


@setup_function
def setup_theano():
    oldenv = os.environ.get('OMP_NUM_THREADS')
    try:
        os.environ.update(
            OMP_NUM_THREADS='1',
        )
        try_import('theano')
        yield
    finally:
        if oldenv is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = oldenv


@setup_function
def setup_numba():
    try_import('numba')
    yield


@setup_function
def setup_jax():
    oldenv = os.environ.get('XLA_FLAGS')
    try:
        os.environ.update(
            XLA_FLAGS=(
                "--xla_cpu_multi_thread_eigen=false "
                "intra_op_parallelism_threads=1 "
                "inter_op_parallelism_threads=1 "
            )
        )

        try_import('jax')
        from jax.config import config
        # use 64 bit floats
        config.update('jax_enable_x64', True)
        yield
    finally:
        if oldenv is None:
            del os.environ['XLA_FLAGS']
        else:
            os.environ['XLA_FLAGS'] = oldenv


@setup_function
def setup_pytorch():
    oldenv = os.environ.get('OMP_NUM_THREADS')
    try:
        os.environ.update(
            OMP_NUM_THREADS='1',
        )
        try_import('torch')
        yield
    finally:
        if oldenv is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = oldenv


@setup_function
def setup_tensorflow():
    oldenv = os.environ.get('OMP_NUM_THREADS')
    try:
        os.environ.update(
            OMP_NUM_THREADS='1',
        )
        try_import('tensorflow')
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        yield
    finally:
        if oldenv is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = oldenv


__backends__ = {
    'numpy': setup_numpy,
    'bohrium': setup_bohrium,
    'jax': setup_jax,
    'theano': setup_theano,
    'numba': setup_numba,
    'pytorch': setup_pytorch,
    'tensorflow': setup_tensorflow
}
