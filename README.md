[![DOI](https://zenodo.org/badge/212333820.svg)](https://zenodo.org/badge/latestdoi/212333820)

# HPC benchmarks for Python

This is a suite of benchmarks to test the *sequential CPU* and GPU performance of various computational backends with Python frontends.

Specifically, we want to test which high-performance backend is best for *geophysical* (finite-difference based) *simulations*.

**Contents**

- [FAQ](#faq)
- [Installation](#environment-setup)
- [Usage](#usage)
- [Example results](#example-results)
- [Conclusion](#conclusion)
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
in vectorized form by index shifts of arrays (such as `0.5 * (arr[1:] + arr[:-1])`, the first-order derivative of `arr` at every point). The most common index range is `[2:-2]`, which represents the full domain (the two outermost grid cells are overlap / "ghost cells" that allow us to shift the array across the boundary).

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
- [Aesara](https://github.com/aesara-devs/aesara) (CPU only)
- [Jax](https://github.com/google/jax)
- [Tensorflow](https://www.tensorflow.org)
- [Pytorch](https://pytorch.org)
- [CuPy](https://cupy.chainer.org/) (GPU only)

(not every backend is available for every benchmark)

### What is included in the measurements?

Pure time spent number crunching. Preparing the inputs, copying stuff from and to GPU, compilation time, time it takes to check results etc. are excluded.
This is based on the assumption that these things are only done a few times per simulation (i.e., that their cost is
amortized during long-running simulations).

### How does this compare to a low-level implementation?

As a rule of thumb (from our experience with Veros), the performance of a Fortran implementation is very close to that of the Numba backend, or ~3 times faster than NumPy.


## Environment setup

For CPU:

```bash
$ conda env create -f environment-cpu.yml
$ conda activate pyhpc-bench-cpu
```

GPU:

```bash
$ conda env create -f environment-gpu.yml
$ conda activate pyhpc-bench-gpu
```

If you prefer to install things by hand, just have a look at the environment files to see what you need. You don't need to install all backends; if a module is unavailable, it is skipped automatically.

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

      $ python run.py benchmarks/equation_of_state -b numpy -b jax --device
      gpu

  More information:

      https://github.com/dionhaefner/pyhpc-benchmarks

Options:
  -s, --size INTEGER              Run benchmark for this array size
                                  (repeatable)  [default: 4096, 16384, 65536,
                                  262144, 1048576, 4194304]
  -b, --backend [numpy|cupy|jax|aesara|numba|pytorch|tensorflow]
                                  Run benchmark with this backend (repeatable)
                                  [default: run all backends]
  -r, --repetitions INTEGER       Fixed number of iterations to run for each
                                  size and backend [default: auto-detect]
  --burnin INTEGER                Number of initial iterations that are
                                  disregarded for final statistics  [default:
                                  1]
  --device [cpu|gpu|tpu]          Run benchmarks on given device where
                                  supported by the backend  [default: cpu]
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

Some backends are greedy with allocating memory. On GPU, you can only run one backend at a time (add NumPy for reference):

```bash
$ conda activate pyhpc-bench-gpu
$ export CUDA_VISIBLE_DEVICES="0"
$ for backend in jax cupy pytorch tensorflow; do
...    python run benchmarks/<benchmark_name> --device gpu -b $backend -b numpy -s 10_000_000
...    done
```

## Example results

### Summary

#### Equation of state

<p align="middle">
  <img src="results/magni-plots/bench-equation_of_state-CPU.png?raw=true" width="400">
  <img src="results/magni-plots/bench-equation_of_state-GPU.png?raw=true" width="400">
</p>
  
#### Isoneutral mixing

<p align="middle">
  <img src="results/magni-plots/bench-isoneutral_mixing-CPU.png?raw=true" width="400">
  <img src="results/magni-plots/bench-isoneutral_mixing-GPU.png?raw=true" width="400">
</p>

#### Turbulent kinetic energy

<p align="middle">
  <img src="results/magni-plots/bench-turbulent_kinetic_energy-CPU.png?raw=true" width="400">
  <img src="results/magni-plots/bench-turbulent_kinetic_energy-GPU.png?raw=true" width="400">
</p>

### Full reports

- [Example results on bare metal with Tesla P100 GPU](/results/magni.md) (more reliable)
- [Example results on Google Colab](/results/colab.md) (easier to reproduce)

## Conclusion

Lessons I learned by assembling these benchmarks: (your mileage may vary)

- The performance of JAX is very competitive, both on GPU and CPU. It is consistently among the top implementations on both platforms.
- Pytorch performs very well on GPU for large problems (slightly better than JAX), but its CPU performance is not great for tasks with many slicing operations.
- Numba is a great choice on CPU if you don't mind writing explicit for loops (which can be more readable than a vectorized implementation), being slightly faster than JAX with little effort.
- JAX performance on GPU seems to be quite hardware dependent. JAX performancs significantly better (relatively speaking) on a Tesla P100 than a Tesla K80.
- If you have embarrasingly parallel workloads, speedups of > 1000x are easy to achieve on high-end GPUs.
- TPUs are catching up to GPUs. We can now get similar performance to a high-end GPU on these workloads.
- Tensorflow is not great for applications like ours, since it lacks tools to apply partial updates to tensors (such as `tensor[2:-2] = 0.`).
- If you use Tensorflow on CPU, make sure to use XLA (`experimental_compile`) for tremendous speedups.
- CuPy is nice! Often you don't need to change anything in your NumPy code to have it run on GPU (with decent, but not outstanding performance).
- Reaching Fortran performance on CPU for non-trivial tasks is hard :)

## Contributing

Community contributions are encouraged! Whether you want to donate another benchmark, share your experience, optimize an implementation, or suggest another backend - [feel free to ask](https://github.com/dionhaefner/pyhpc-benchmarks/issues) or [open a PR](https://github.com/dionhaefner/pyhpc-benchmarks/pulls).

### Adding a new backend

Adding a new backend is easy!

Let's assume that you want to add support for a library called `speedygonzales`. All you need to do is this:

- Implement a benchmark to use your library, e.g. `benchmarks/equation_of_state/eos_speedygonzales.py`.
- Register the benchmark in the respective `__init__.py` file (`benchmarks/equation_of_state/__init__.py`), by adding `"speedygonzales"` to its `__implementations__` tuple.
- Register the backend, by adding its setup function to the `__backends__` dict in [`backends.py`](https://github.com/dionhaefner/pyhpc-benchmarks/blob/master/backends.py).

   A setup function is what is called before every call to your benchmark, and can be used for custom setup and teardown. In the simplest case, it is just

   ```python
   def setup_speedygonzales(device='cpu'):
       # code to run before benchmark
       yield
       # code to run after benchmark
   ```

Then, you can run the benchmark with your new backend:

```bash
$ python run.py benchmarks/equation_of_state -b speedygonzales
```
