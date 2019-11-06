# HPC benchmarks for Python

This is a suite of benchmarks to test the *sequential CPU* and GPU performance of various computational backends with Python frontends.

Specifically, we want to test which high-performance backend is best for *geophysical* (finite-difference based) *simulations*.

**Contents**

- [FAQ](#faq)
- [Installation](#environment-setup)
- [Usage](#usage)
- [Example results](#example-results)
- [Contributing](#contributing)

## FAQ

### Why?

The scientific Python ecosystem is thriving, but high-performance computing in Python isn't really a thing yet.
We try to change this [with our pure Python ocean simulator Veros](https://github.com/dionhaefner/veros), but which backend should we use for computations?

Tremendous amounts of time and resources go into the development of Python frontends to high-performance backends,
but those are usually tailored towards deep learning. We wanted to see whether we can profit from those advances, by
(ab-)using these libraries for geophysical modelling.

### Why do the benchmarks look so weird?

These are more or less verbatim copies from [Veros](https://github.com/dionhaefner/veros) (i.e., actual parts of a physical model).
Most earth system and climate model components are based on finite-difference schemes to compute derivatives. This can be represented
in vectorized form by index shifts of arrays (such as `0.5 * (arr[1:] + arr[:-1])`, the first-order derivative of `arr` at every point).

Now, maths is difficult, and numerics are weird. When many different physical quantities (defined on different grids) interact, things
get messy very fast.

### Why only test sequential CPU performance?

Two reasons:
- I was curious to see how good the compilers are without being able to fall back to thread parallelism.
- In many physical models, it is pretty straightforward to parallelize the model "by hand" via MPI.
  Therefore, we are not really dependent on good parallel performance out of the box.

### Which backends are currently supported?

- [NumPy](https://numpy.org) (CPU only)
- [Numba](https://numba.pydata.org) (CPU only)
- [Jax](https://github.com/google/jax)
- [Tensorflow](https://www.tensorflow.org)
- [Pytorch](https://pytorch.org)
- [Theano](http://deeplearning.net/software/theano/)
- [Bohrium](http://www.bh107.org)
- [CuPy](https://cupy.chainer.org/) (GPU only)

(not every backend is available for every benchmark)

## Environment setup

For CPU:

```bash
$ conda create -f environment_cpu.yml
$ conda activate pyhpc-bench-cpu
```

GPU:

```bash
$ conda create -f environment_gpu.yml
$ conda activate pyhpc-bench-gpu
```

If you prefer to install things by hand, just have a look at the environment files to see what you need.

## Usage

Your entrypoint is the script `run.py`:

```bash
$ python run.py --help
Usage: run.py [OPTIONS] BENCHMARK

  HPC benchmarks for Python

  Usage:

      $ python run.py benchmarks/<BENCHMARK_FOLDER>

  Examples:

      $ taskset -c 0 python run.py benchmarks/equation_of_state

      $ python run.py benchmarks/equation_of_state -b numpy -b jax --gpu

  More information:

      https://github.com/dionhaefner/pyhpc-benchmarks

Options:
  -s, --size INTEGER              Run benchmark for this array size
                                  (repeatable)  [default: 4096, 16384, 65536,
                                  262144, 1048576, 4194304]
  -b, --backend [numpy|bohrium|cupy|jax|theano|numba|pytorch|tensorflow]
                                  Run benchmark with this backend (repeatable)
                                  [default: run all backends]
  -r, --repetitions INTEGER       Fixed number of iterations to run for each
                                  size and backend [default: auto-detect]
  --burnin INTEGER                Number of initial iterations that are
                                  disregarded for final statistics  [default:
                                  1]
  --gpu / --cpu                   Run benchmarks on GPU where supported by the
                                  backend [default: CPU]
  --help                          Show this message and exit.
```

Benchmarks are run for all combinations of the chosen sizes (`-s`) and backends (`-b`), in random order.

### CPU

Some backends refuse to be confined to a single thread, so I recommend you wrap your benchmarks
in `taskset` to set processor affinity to a single core (only works on Linux):

```bash
$ conda activate pyhpc-bench-cpu
$ taskset -c 0 python run.py benchmarks/<benchmark_name>
```

### GPU

Some backends use all available GPUs by default, some don't. If you have multiple GPUs, you can set the
one to be used through `CUDA_VISIBLE_DEVICES`, so keep things fair.

```bash
$ conda activate pyhpc-bench-gpu
$ export CUDA_VISIBLE_DEVICES="0"
$ python run.py benchmarks/<benchmark_name> --gpu
```

Some backends are pretty greedy with allocating memory. For large problem sizes, it can be a good idea to
only run one backend at a time (and NumPy for reference):

```bash
$ conda activate pyhpc-bench-gpu
$ export CUDA_VISIBLE_DEVICES="0"
$ for backend in bohrium jax cupy pytorch tensorflow; do
...    python run benchmarks/<benchmark_name> --gpu -b $backend -b numpy -s 10_000_000
...    done
```

## Example results

I ran these on the following **hardware**:

- Intel Xeon E5-2650 v4 @ 2.20 GHz
- 512GB DDR4 memory (not that we would need it)
- NVidia Tesla P100 (16GB memory)

### CPU

```bash
```

### GPU

```bash
```

## Contributing

Community contributions are encouraged! Whether you want to donate another benchmark, share your experience, optimize an implementation, or suggest another backend - [feel free to ask](https://github.com/dionhaefner/pyhpc-benchmarks/issues) or [open a PR](https://github.com/dionhaefner/pyhpc-benchmarks/pulls).
