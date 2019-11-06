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
- [Jax](https://github.com/google/jax)
- [Tensorflow](https://www.tensorflow.org)
- [Pytorch](https://pytorch.org)
- [Theano](http://deeplearning.net/software/theano/)
- [Bohrium](http://www.bh107.org)
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
$ conda create -f environment_cpu.yml
$ conda activate pyhpc-bench-cpu
```

GPU:

```bash
$ conda create -f environment_gpu.yml
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

I ran these on the following hardware:

- Intel Xeon E5-2650 v4 @ 2.20 GHz
- 512GB DDR4 memory (not that we would need it)
- NVidia Tesla P100 (16GB memory)

Software stack:
- CentOS 7
- GNU compiler tookit 8.3.0
- Python 3.7.3
- CUDA 10.1
- Most packages pulled from conda-forge (exceptions see below)
- Backend versions:

    ```bash
    bohrium==0.11.0.post19  # built from source
    cupy-cuda101==6.5.0
    jax==0.1.49  # built from source
    jaxlib==0.1.32  # built from source
    llvmlite==0.30.0
    numba==0.46.0
    numpy==1.17.2
    tensorflow==2.0.0
    theano==1.0.4
    torch==1.2.0
    ```

### Equation of state

#### CPU

```bash
$ taskset -c 23 python run.py benchmarks/equation_of_state/
Setup for backend "cupy" failed (skipping), reason: cupy requires GPU mode
Estimating repetitions...
Running 81642 benchmarks...  [####################################]  100%

benchmarks.equation_of_state
============================
Running on CPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  jax           10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.006     3.060
       4,096  theano        10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.011     2.834
       4,096  numba         10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.011     2.826
       4,096  numpy         10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.006     1.000
       4,096  tensorflow    10,000     0.002     0.001     0.002     0.002     0.002     0.002     0.013     0.923
       4,096  pytorch        1,000     0.003     0.000     0.002     0.003     0.003     0.003     0.007     0.711
       4,096  bohrium          100     0.058     0.000     0.058     0.058     0.058     0.058     0.059     0.032

      16,384  jax           10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.013     3.754
      16,384  theano         1,000     0.003     0.000     0.003     0.003     0.003     0.003     0.007     3.150
      16,384  numba         10,000     0.003     0.000     0.003     0.003     0.003     0.003     0.013     2.867
      16,384  tensorflow     1,000     0.007     0.001     0.006     0.007     0.007     0.007     0.018     1.174
      16,384  pytorch        1,000     0.008     0.000     0.008     0.008     0.008     0.008     0.009     1.013
      16,384  numpy          1,000     0.008     0.000     0.008     0.008     0.008     0.008     0.012     1.000
      16,384  bohrium          100     0.061     0.000     0.060     0.060     0.061     0.061     0.062     0.137

      65,536  jax            1,000     0.007     0.000     0.007     0.007     0.007     0.007     0.018     4.462
      65,536  theano         1,000     0.010     0.000     0.010     0.010     0.010     0.010     0.014     3.259
      65,536  numba          1,000     0.011     0.000     0.011     0.011     0.011     0.011     0.016     2.889
      65,536  tensorflow     1,000     0.030     0.004     0.028     0.028     0.028     0.028     0.048     1.113
      65,536  pytorch          100     0.031     0.001     0.031     0.031     0.031     0.031     0.036     1.061
      65,536  numpy            100     0.033     0.001     0.033     0.033     0.033     0.033     0.038     1.000
      65,536  bohrium          100     0.069     0.000     0.068     0.069     0.069     0.069     0.070     0.477

     262,144  jax            1,000     0.028     0.001     0.026     0.028     0.028     0.029     0.038     4.752
     262,144  theano           100     0.037     0.000     0.036     0.037     0.037     0.037     0.037     3.693
     262,144  numba            100     0.042     0.000     0.041     0.042     0.042     0.042     0.042     3.246
     262,144  bohrium          100     0.100     0.000     0.099     0.100     0.100     0.100     0.101     1.351
     262,144  pytorch          100     0.118     0.003     0.116     0.117     0.117     0.118     0.139     1.145
     262,144  numpy            100     0.135     0.005     0.132     0.133     0.133     0.137     0.163     1.000
     262,144  tensorflow       100     0.149     0.004     0.146     0.148     0.148     0.148     0.171     0.906

   1,048,576  jax              100     0.117     0.001     0.116     0.117     0.117     0.117     0.122     6.752
   1,048,576  theano           100     0.150     0.001     0.148     0.151     0.151     0.151     0.153     5.245
   1,048,576  numba            100     0.170     0.002     0.165     0.170     0.170     0.171     0.173     4.641
   1,048,576  bohrium          100     0.231     0.002     0.225     0.231     0.232     0.232     0.234     3.418
   1,048,576  tensorflow        10     0.684     0.003     0.681     0.682     0.682     0.686     0.689     1.154
   1,048,576  numpy             10     0.789     0.006     0.783     0.784     0.788     0.793     0.801     1.000
   1,048,576  pytorch           10     0.852     0.011     0.843     0.845     0.847     0.850     0.881     0.927

   4,194,304  jax               10     0.446     0.000     0.445     0.446     0.446     0.446     0.447     8.303
   4,194,304  theano            10     0.593     0.005     0.583     0.594     0.594     0.596     0.598     6.244
   4,194,304  numba             10     0.667     0.003     0.661     0.666     0.666     0.669     0.674     5.553
   4,194,304  bohrium           10     0.740     0.003     0.730     0.740     0.740     0.742     0.743     5.005
   4,194,304  tensorflow        10     3.002     0.013     2.994     2.996     2.997     2.998     3.031     1.233
   4,194,304  numpy             10     3.703     0.006     3.696     3.698     3.703     3.705     3.716     1.000
   4,194,304  pytorch           10     4.791     0.004     4.784     4.789     4.792     4.793     4.798     0.773

(time in wall seconds, less is better)

$ taskset -c 23 python run.py benchmarks/equation_of_state/ -s 16777216
Setup for backend "cupy" failed (skipping), reason: cupy requires GPU mode
Estimating repetitions...
Running 77 benchmarks...  [####################################]  100%

benchmarks.equation_of_state
============================
Running on CPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
  16,777,216  jax               10     1.864     0.007     1.857     1.860     1.861     1.867     1.880     7.793
  16,777,216  theano            10     2.389     0.007     2.378     2.384     2.391     2.394     2.397     6.081
  16,777,216  numba             10     2.681     0.010     2.667     2.676     2.679     2.683     2.707     5.419
  16,777,216  bohrium           10     3.191     0.016     3.153     3.184     3.193     3.203     3.208     4.553
  16,777,216  tensorflow        10    11.881     0.055    11.844    11.850    11.863    11.874    12.035     1.223
  16,777,216  numpy             10    14.527     0.074    14.454    14.481    14.500    14.530    14.678     1.000
  16,777,216  pytorch           10    22.358     0.124    22.152    22.278    22.363    22.447    22.577     0.650

(time in wall seconds, less is better)
```

#### GPU

```bash
$ for backend in bohrium cupy jax pytorch tensorflow theano; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/equation_of_state/ --gpu -b $backend -b numpy; done
Estimating repetitions...
Running 11832 benchmarks...  [####################################]  100%

benchmarks.equation_of_state
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  numpy         10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.019     1.000
       4,096  bohrium          100     0.059     0.001     0.058     0.059     0.059     0.059     0.069     0.031

      16,384  numpy          1,000     0.008     0.001     0.008     0.008     0.008     0.008     0.017     1.000
      16,384  bohrium          100     0.059     0.001     0.058     0.059     0.059     0.059     0.066     0.137

      65,536  numpy            100     0.034     0.003     0.032     0.033     0.033     0.033     0.045     1.000
      65,536  bohrium          100     0.059     0.003     0.058     0.059     0.059     0.059     0.088     0.564

     262,144  bohrium          100     0.060     0.002     0.059     0.059     0.059     0.059     0.075     2.259
     262,144  numpy            100     0.135     0.018     0.125     0.126     0.126     0.132     0.234     1.000

   1,048,576  bohrium          100     0.060     0.000     0.059     0.059     0.059     0.060     0.062    13.984
   1,048,576  numpy             10     0.832     0.011     0.825     0.827     0.827     0.828     0.860     1.000

   4,194,304  bohrium          100     0.061     0.001     0.060     0.061     0.061     0.061     0.067    60.235
   4,194,304  numpy             10     3.666     0.011     3.658     3.661     3.662     3.663     3.697     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 16332 benchmarks...  [####################################]  100%

benchmarks.equation_of_state
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  numpy         10,000     0.002     0.001     0.002     0.002     0.002     0.002     0.016     1.000
       4,096  cupy           1,000     0.012     0.001     0.011     0.011     0.011     0.011     0.025     0.164

      16,384  numpy          1,000     0.009     0.002     0.008     0.008     0.008     0.011     0.021     1.000
      16,384  cupy           1,000     0.012     0.001     0.011     0.011     0.011     0.011     0.024     0.800

      65,536  cupy           1,000     0.012     0.001     0.011     0.011     0.011     0.011     0.025     4.321
      65,536  numpy            100     0.050     0.002     0.041     0.049     0.051     0.052     0.055     1.000

     262,144  cupy           1,000     0.012     0.001     0.011     0.011     0.011     0.011     0.025    16.872
     262,144  numpy            100     0.196     0.003     0.193     0.194     0.195     0.196     0.214     1.000

   1,048,576  cupy           1,000     0.016     0.001     0.016     0.016     0.016     0.016     0.030    51.724
   1,048,576  numpy             10     0.851     0.003     0.847     0.848     0.850     0.853     0.857     1.000

   4,194,304  cupy             100     0.061     0.002     0.060     0.060     0.060     0.060     0.074    60.524
   4,194,304  numpy             10     3.674     0.004     3.666     3.672     3.675     3.677     3.680     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 71232 benchmarks...  [####################################]  100%

benchmarks.equation_of_state
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  jax           10,000     0.000     0.000     0.000     0.000     0.000     0.000     0.012     9.063
       4,096  numpy         10,000     0.002     0.001     0.002     0.002     0.002     0.002     0.016     1.000

      16,384  jax           10,000     0.000     0.000     0.000     0.000     0.000     0.000     0.014    50.354
      16,384  numpy          1,000     0.011     0.001     0.008     0.011     0.011     0.012     0.020     1.000

      65,536  jax           10,000     0.000     0.000     0.000     0.000     0.000     0.000     0.012   218.568
      65,536  numpy            100     0.050     0.002     0.032     0.050     0.050     0.051     0.057     1.000

     262,144  jax           10,000     0.000     0.000     0.000     0.000     0.000     0.000     0.012   793.155
     262,144  numpy            100     0.193     0.007     0.126     0.193     0.194     0.195     0.196     1.000

   1,048,576  jax           10,000     0.001     0.001     0.001     0.001     0.001     0.001     0.012  1036.462
   1,048,576  numpy             10     0.844     0.005     0.836     0.838     0.845     0.847     0.851     1.000

   4,194,304  jax           10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.015  2811.487
   4,194,304  numpy             10     3.646     0.018     3.618     3.628     3.655     3.659     3.662     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 431232 benchmarks...  [####################################]  100%

benchmarks.equation_of_state
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  pytorch      100,000     0.000     0.000     0.000     0.000     0.000     0.000     0.017    31.673
       4,096  numpy         10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.015     1.000

      16,384  pytorch      100,000     0.000     0.000     0.000     0.000     0.000     0.000     0.015   132.561
      16,384  numpy          1,000     0.008     0.000     0.008     0.008     0.008     0.008     0.011     1.000

      65,536  pytorch      100,000     0.000     0.000     0.000     0.000     0.000     0.000     0.015   457.425
      65,536  numpy            100     0.040     0.006     0.032     0.032     0.044     0.045     0.050     1.000

     262,144  pytorch      100,000     0.000     0.000     0.000     0.000     0.000     0.000     0.018  1036.713
     262,144  numpy            100     0.186     0.006     0.175     0.179     0.185     0.191     0.198     1.000

   1,048,576  pytorch       10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.015  1376.581
   1,048,576  numpy             10     0.798     0.004     0.791     0.796     0.797     0.800     0.804     1.000

   4,194,304  pytorch       10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.013  1854.139
   4,194,304  numpy             10     3.695     0.001     3.693     3.694     3.694     3.696     3.697     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 16332 benchmarks...  [####################################]  100%

benchmarks.equation_of_state
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  numpy         10,000     0.002     0.001     0.002     0.002     0.002     0.002     0.014     1.000
       4,096  tensorflow     1,000     0.006     0.007     0.003     0.004     0.004     0.005     0.056     0.324

      16,384  tensorflow     1,000     0.006     0.007     0.003     0.004     0.004     0.005     0.055     1.436
      16,384  numpy          1,000     0.009     0.001     0.007     0.008     0.008     0.008     0.020     1.000

      65,536  tensorflow     1,000     0.006     0.007     0.003     0.004     0.004     0.005     0.054     6.195
      65,536  numpy            100     0.039     0.004     0.033     0.036     0.037     0.043     0.053     1.000

     262,144  tensorflow     1,000     0.006     0.005     0.003     0.005     0.005     0.005     0.049    33.489
     262,144  numpy            100     0.188     0.003     0.184     0.185     0.187     0.190     0.195     1.000

   1,048,576  tensorflow     1,000     0.008     0.003     0.005     0.008     0.008     0.008     0.032    99.634
   1,048,576  numpy             10     0.827     0.001     0.825     0.826     0.827     0.828     0.829     1.000

   4,194,304  tensorflow       100     0.017     0.001     0.013     0.017     0.017     0.017     0.020   214.069
   4,194,304  numpy             10     3.682     0.003     3.678     3.679     3.682     3.685     3.687     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 52332 benchmarks...  [####################################]  100%

benchmarks.equation_of_state
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  theano        10,000     0.000     0.001     0.000     0.000     0.000     0.000     0.012     7.404
       4,096  numpy         10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.013     1.000

      16,384  theano        10,000     0.000     0.000     0.000     0.000     0.000     0.000     0.012    23.923
      16,384  numpy          1,000     0.008     0.001     0.008     0.008     0.008     0.008     0.020     1.000

      65,536  theano        10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.012    66.858
      65,536  numpy            100     0.042     0.002     0.033     0.041     0.042     0.045     0.046     1.000

     262,144  theano        10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.013   118.739
     262,144  numpy            100     0.196     0.002     0.192     0.195     0.196     0.198     0.208     1.000

   1,048,576  theano         1,000     0.008     0.001     0.006     0.006     0.008     0.009     0.009   114.621
   1,048,576  numpy             10     0.860     0.001     0.858     0.859     0.860     0.861     0.863     1.000

   4,194,304  theano           100     0.032     0.000     0.032     0.032     0.032     0.032     0.033   115.181
   4,194,304  numpy             10     3.737     0.003     3.735     3.735     3.736     3.737     3.745     1.000

(time in wall seconds, less is better)
```

### Isoneutral mixing

#### CPU

```bash
$ taskset -c 23 python run.py benchmarks/isoneutral_mixing/
Setup for backend "cupy" failed (skipping), reason: cupy requires GPU mode
Estimating repetitions...
Running 30456 benchmarks...  [####################################]  100%

benchmarks.isoneutral_mixing
============================
Running on CPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  numba         10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.012     3.155
       4,096  jax           10,000     0.002     0.000     0.001     0.001     0.002     0.002     0.013     2.856
       4,096  theano         1,000     0.003     0.000     0.003     0.003     0.003     0.003     0.015     1.529
       4,096  numpy          1,000     0.004     0.000     0.004     0.004     0.004     0.004     0.012     1.000
       4,096  pytorch        1,000     0.007     0.000     0.007     0.007     0.007     0.007     0.013     0.657
       4,096  bohrium          100     0.079     0.003     0.078     0.078     0.078     0.079     0.105     0.057

      16,384  jax            1,000     0.006     0.000     0.006     0.006     0.006     0.006     0.016     2.535
      16,384  numba          1,000     0.007     0.000     0.007     0.007     0.007     0.007     0.012     2.342
      16,384  theano         1,000     0.011     0.000     0.011     0.011     0.011     0.011     0.017     1.478
      16,384  pytorch        1,000     0.016     0.001     0.016     0.016     0.016     0.016     0.023     1.019
      16,384  numpy          1,000     0.016     0.000     0.016     0.016     0.016     0.016     0.023     1.000
      16,384  bohrium          100     0.086     0.001     0.085     0.085     0.086     0.086     0.089     0.191

      65,536  jax            1,000     0.029     0.001     0.028     0.029     0.029     0.029     0.036     2.269
      65,536  numba            100     0.031     0.001     0.031     0.031     0.031     0.031     0.035     2.103
      65,536  theano           100     0.043     0.001     0.043     0.043     0.043     0.043     0.048     1.513
      65,536  pytorch          100     0.049     0.000     0.048     0.049     0.049     0.049     0.051     1.337
      65,536  numpy            100     0.065     0.001     0.065     0.065     0.065     0.065     0.069     1.000
      65,536  bohrium          100     0.117     0.002     0.115     0.115     0.116     0.119     0.124     0.559

     262,144  numba            100     0.123     0.004     0.118     0.120     0.121     0.129     0.132     2.127
     262,144  jax              100     0.130     0.001     0.128     0.129     0.131     0.131     0.132     2.019
     262,144  theano           100     0.195     0.007     0.178     0.189     0.196     0.201     0.205     1.350
     262,144  pytorch          100     0.195     0.006     0.187     0.190     0.195     0.200     0.207     1.344
     262,144  bohrium          100     0.235     0.005     0.229     0.230     0.232     0.239     0.262     1.120
     262,144  numpy            100     0.263     0.008     0.249     0.256     0.264     0.270     0.281     1.000

   1,048,576  numba             10     0.558     0.005     0.554     0.556     0.557     0.558     0.573     2.260
   1,048,576  jax               10     0.579     0.001     0.578     0.578     0.579     0.580     0.582     2.178
   1,048,576  bohrium           10     0.725     0.004     0.718     0.724     0.725     0.726     0.736     1.740
   1,048,576  theano            10     0.828     0.005     0.823     0.825     0.826     0.827     0.839     1.525
   1,048,576  pytorch           10     1.058     0.001     1.056     1.057     1.057     1.059     1.060     1.193
   1,048,576  numpy             10     1.262     0.004     1.257     1.259     1.260     1.265     1.270     1.000

   4,194,304  numba             10     2.306     0.010     2.295     2.296     2.305     2.316     2.320     2.251
   4,194,304  jax               10     2.481     0.003     2.477     2.479     2.480     2.483     2.485     2.092
   4,194,304  bohrium           10     2.702     0.013     2.683     2.691     2.701     2.711     2.728     1.921
   4,194,304  theano            10     3.337     0.017     3.286     3.340     3.342     3.344     3.347     1.555
   4,194,304  numpy             10     5.190     0.010     5.167     5.188     5.193     5.198     5.201     1.000
   4,194,304  pytorch           10     5.412     0.136     5.133     5.337     5.459     5.522     5.580     0.959

$ taskset -c 23 python run.py benchmarks/isoneutral_mixing/ -s 16777216
Setup for backend "cupy" failed (skipping), reason: cupy requires GPU mode
Estimating repetitions...
Running 66 benchmarks...  [####################################]  100%

benchmarks.isoneutral_mixing
============================
Running on CPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
  16,777,216  numba             10     9.374     0.038     9.314     9.349     9.377     9.409     9.426     2.583
  16,777,216  jax               10     9.748     0.013     9.732     9.739     9.748     9.751     9.781     2.484
  16,777,216  bohrium           10    11.219     0.085    11.100    11.139    11.228    11.299    11.331     2.159
  16,777,216  theano            10    14.132     0.058    14.023    14.098    14.130    14.166    14.246     1.714
  16,777,216  numpy             10    24.218     0.069    24.134    24.158    24.204    24.285    24.326     1.000
  16,777,216  pytorch           10    38.636     0.178    38.357    38.537    38.636    38.765    38.935     0.627

(time in wall seconds, less is better)
```

#### GPU

```bash
$ for backend in bohrium cupy jax pytorch theano; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/isoneutral_mixing/ --gpu -b $backend -b numpy; done

Estimating repetitions...
Running 2832 benchmarks...  [####################################]  100%

benchmarks.isoneutral_mixing
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  numpy          1,000     0.005     0.001     0.004     0.004     0.004     0.005     0.009     1.000
       4,096  bohrium          100     0.081     0.003     0.079     0.079     0.080     0.081     0.093     0.058

      16,384  numpy          1,000     0.017     0.002     0.016     0.016     0.017     0.017     0.054     1.000
      16,384  bohrium          100     0.081     0.004     0.077     0.079     0.080     0.081     0.113     0.209

      65,536  numpy            100     0.068     0.005     0.065     0.065     0.066     0.070     0.101     1.000
      65,536  bohrium          100     0.086     0.011     0.079     0.080     0.082     0.084     0.117     0.796

     262,144  bohrium          100     0.086     0.004     0.081     0.082     0.086     0.088     0.102     3.276
     262,144  numpy            100     0.281     0.009     0.264     0.275     0.278     0.282     0.312     1.000

   1,048,576  bohrium          100     0.104     0.006     0.095     0.097     0.104     0.108     0.130    12.345
   1,048,576  numpy             10     1.278     0.020     1.258     1.266     1.274     1.280     1.334     1.000

   4,194,304  bohrium          100     0.196     0.006     0.185     0.191     0.197     0.200     0.212    26.890
   4,194,304  numpy             10     5.262     0.032     5.241     5.243     5.248     5.269     5.349     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 7332 benchmarks...  [####################################]  100%

benchmarks.isoneutral_mixing
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  numpy          1,000     0.005     0.001     0.004     0.004     0.005     0.005     0.009     1.000
       4,096  cupy           1,000     0.016     0.001     0.016     0.016     0.016     0.016     0.025     0.283

      16,384  cupy           1,000     0.016     0.001     0.016     0.016     0.016     0.016     0.024     1.029
      16,384  numpy          1,000     0.017     0.001     0.016     0.017     0.017     0.017     0.021     1.000

      65,536  cupy           1,000     0.017     0.001     0.016     0.016     0.016     0.016     0.025     4.072
      65,536  numpy            100     0.067     0.003     0.064     0.065     0.066     0.068     0.081     1.000

     262,144  cupy           1,000     0.017     0.001     0.016     0.016     0.016     0.017     0.028    16.638
     262,144  numpy            100     0.280     0.008     0.267     0.274     0.279     0.281     0.305     1.000

   1,048,576  cupy           1,000     0.023     0.001     0.022     0.022     0.022     0.023     0.028    56.042
   1,048,576  numpy             10     1.278     0.029     1.249     1.259     1.268     1.283     1.333     1.000

   4,194,304  cupy             100     0.086     0.001     0.085     0.085     0.085     0.086     0.092    61.664
   4,194,304  numpy             10     5.283     0.040     5.232     5.257     5.271     5.315     5.354     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 34332 benchmarks...  [####################################]  100%

benchmarks.isoneutral_mixing
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  jax           10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.006     4.137
       4,096  numpy          1,000     0.005     0.000     0.004     0.005     0.005     0.005     0.009     1.000

      16,384  jax           10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.008    13.834
      16,384  numpy          1,000     0.017     0.001     0.016     0.016     0.016     0.017     0.025     1.000

      65,536  jax           10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.007    36.303
      65,536  numpy            100     0.066     0.003     0.063     0.064     0.064     0.067     0.078     1.000

     262,144  jax            1,000     0.005     0.000     0.005     0.005     0.005     0.005     0.010    51.533
     262,144  numpy            100     0.274     0.009     0.265     0.267     0.273     0.276     0.307     1.000

   1,048,576  jax            1,000     0.018     0.000     0.017     0.017     0.018     0.018     0.022    71.795
   1,048,576  numpy             10     1.261     0.045     1.226     1.230     1.236     1.286     1.345     1.000

   4,194,304  jax              100     0.065     0.000     0.065     0.065     0.065     0.065     0.065    80.580
   4,194,304  numpy             10     5.227     0.121     5.129     5.153     5.162     5.262     5.513     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 7332 benchmarks...  [####################################]  100%

benchmarks.isoneutral_mixing
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  numpy          1,000     0.005     0.001     0.004     0.004     0.005     0.005     0.009     1.000
       4,096  pytorch        1,000     0.008     0.001     0.008     0.008     0.008     0.008     0.015     0.578

      16,384  pytorch        1,000     0.008     0.001     0.008     0.008     0.008     0.008     0.014     2.074
      16,384  numpy          1,000     0.017     0.001     0.016     0.016     0.017     0.017     0.021     1.000

      65,536  pytorch        1,000     0.008     0.000     0.008     0.008     0.008     0.008     0.013     8.367
      65,536  numpy            100     0.068     0.004     0.063     0.064     0.068     0.070     0.081     1.000

     262,144  pytorch        1,000     0.008     0.001     0.008     0.008     0.008     0.008     0.013    32.975
     262,144  numpy            100     0.279     0.010     0.259     0.273     0.276     0.286     0.304     1.000

   1,048,576  pytorch        1,000     0.020     0.000     0.020     0.020     0.020     0.020     0.024    63.237
   1,048,576  numpy             10     1.255     0.026     1.231     1.237     1.242     1.276     1.305     1.000

   4,194,304  pytorch          100     0.074     0.001     0.074     0.074     0.074     0.074     0.086    69.963
   4,194,304  numpy             10     5.190     0.061     5.126     5.135     5.168     5.248     5.296     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 15342 benchmarks...  [####################################]  100%

benchmarks.isoneutral_mixing
============================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  theano        10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.005     2.255
       4,096  numpy          1,000     0.004     0.000     0.004     0.004     0.004     0.004     0.007     1.000

      16,384  theano         1,000     0.003     0.000     0.003     0.003     0.003     0.003     0.005     5.453
      16,384  numpy          1,000     0.016     0.000     0.016     0.016     0.016     0.016     0.028     1.000

      65,536  theano         1,000     0.007     0.000     0.007     0.007     0.007     0.007     0.011     9.539
      65,536  numpy            100     0.065     0.000     0.064     0.065     0.065     0.065     0.066     1.000

     262,144  theano         1,000     0.019     0.001     0.018     0.018     0.019     0.019     0.033    13.021
     262,144  numpy            100     0.250     0.003     0.246     0.249     0.250     0.250     0.261     1.000

   1,048,576  theano           100     0.107     0.005     0.089     0.105     0.107     0.110     0.123    11.684
   1,048,576  numpy             10     1.254     0.009     1.240     1.248     1.253     1.258     1.271     1.000

   4,194,304  theano            10     0.432     0.010     0.413     0.426     0.435     0.441     0.444    11.863
   4,194,304  numpy             10     5.127     0.015     5.110     5.117     5.119     5.140     5.154     1.000

(time in wall seconds, less is better)
```

### Turbulent kinetic energy

#### CPU

```bash
$ taskset -c 23 python run.py benchmarks/turbulent_kinetic_energy/
Estimating repetitions...
Running 46074 benchmarks...  [####################################]  100%

benchmarks.turbulent_kinetic_energy
===================================
Running on CPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  jax           10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.005     3.590
       4,096  numba         10,000     0.001     0.000     0.001     0.001     0.001     0.001     0.005     1.961
       4,096  numpy         10,000     0.003     0.000     0.002     0.003     0.003     0.003     0.007     1.000
       4,096  bohrium          100     0.052     0.001     0.050     0.051     0.051     0.052     0.056     0.051

      16,384  jax           10,000     0.002     0.000     0.002     0.002     0.002     0.002     0.014     3.865
      16,384  numba          1,000     0.004     0.000     0.004     0.004     0.004     0.004     0.007     1.876
      16,384  numpy          1,000     0.008     0.000     0.008     0.008     0.008     0.008     0.014     1.000
      16,384  bohrium          100     0.054     0.001     0.052     0.053     0.054     0.054     0.058     0.156

      65,536  jax            1,000     0.010     0.001     0.009     0.009     0.009     0.010     0.013     3.310
      65,536  numba          1,000     0.015     0.001     0.014     0.014     0.014     0.015     0.018     2.153
      65,536  numpy          1,000     0.032     0.001     0.029     0.031     0.031     0.032     0.037     1.000
      65,536  bohrium          100     0.062     0.002     0.060     0.060     0.062     0.063     0.068     0.508

     262,144  numba            100     0.051     0.002     0.047     0.048     0.050     0.052     0.058     2.703
     262,144  jax              100     0.057     0.004     0.049     0.055     0.056     0.056     0.068     2.416
     262,144  bohrium          100     0.092     0.005     0.088     0.089     0.093     0.095     0.125     1.481
     262,144  numpy            100     0.137     0.007     0.124     0.134     0.137     0.140     0.156     1.000

   1,048,576  numba            100     0.197     0.003     0.187     0.195     0.196     0.198     0.208     3.075
   1,048,576  bohrium          100     0.219     0.005     0.212     0.214     0.219     0.223     0.234     2.767
   1,048,576  jax              100     0.294     0.008     0.286     0.290     0.291     0.295     0.322     2.059
   1,048,576  numpy             10     0.606     0.012     0.595     0.600     0.603     0.605     0.633     1.000

   4,194,304  bohrium           10     0.703     0.004     0.699     0.701     0.702     0.703     0.711     3.150
   4,194,304  numba             10     0.749     0.004     0.743     0.746     0.749     0.750     0.757     2.958
   4,194,304  jax               10     1.239     0.021     1.220     1.228     1.233     1.241     1.295     1.788
   4,194,304  numpy             10     2.215     0.026     2.195     2.200     2.204     2.210     2.267     1.000

(time in wall seconds, less is better)

$ taskset -c 23 python run.py benchmarks/turbulent_kinetic_energy/ -s 16777216
Estimating repetitions...
Running 44 benchmarks...  [####################################]  100%

benchmarks.turbulent_kinetic_energy
===================================
Running on CPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
  16,777,216  bohrium           10     2.611     0.007     2.601     2.607     2.609     2.615     2.623     4.187
  16,777,216  numba             10     2.927     0.061     2.904     2.905     2.908     2.910     3.109     3.735
  16,777,216  jax               10     4.733     0.003     4.726     4.732     4.733     4.735     4.738     2.310
  16,777,216  numpy             10    10.933     0.067    10.875    10.885    10.892    11.004    11.037     1.000

(time in wall seconds, less is better)
```

#### GPU

```bash
$ for backend in bohrium jax; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/turbulent_kinetic_energy/ --gpu -b $backend -b numpy; done

Estimating repetitions...
Running 12282 benchmarks...  [####################################]  100%

benchmarks.turbulent_kinetic_energy
===================================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  numpy         10,000     0.003     0.000     0.002     0.002     0.002     0.002     0.016     1.000
       4,096  bohrium           10     0.057     0.000     0.056     0.057     0.057     0.057     0.057     0.044

      16,384  numpy          1,000     0.008     0.000     0.008     0.008     0.008     0.008     0.012     1.000
      16,384  bohrium           10     0.057     0.000     0.056     0.057     0.057     0.057     0.057     0.143

      65,536  numpy          1,000     0.030     0.001     0.027     0.030     0.030     0.030     0.034     1.000
      65,536  bohrium           10     0.058     0.001     0.056     0.057     0.058     0.059     0.060     0.520

     262,144  bohrium           10     0.061     0.003     0.057     0.058     0.060     0.064     0.065     2.177
     262,144  numpy            100     0.132     0.005     0.113     0.133     0.133     0.133     0.148     1.000

   1,048,576  bohrium          100     0.064     0.003     0.062     0.063     0.063     0.063     0.076     9.116
   1,048,576  numpy             10     0.584     0.001     0.582     0.584     0.584     0.585     0.585     1.000

   4,194,304  bohrium           10     0.113     0.014     0.091     0.098     0.119     0.123     0.128    19.375
   4,194,304  numpy             10     2.183     0.005     2.179     2.181     2.182     2.184     2.198     1.000

(time in wall seconds, less is better)

Estimating repetitions...
Running 43332 benchmarks...  [######################--------------]   61%  00:19:28^C  # lost patience

benchmarks.turbulent_kinetic_energy
===================================
Running on GPU

size          backend     calls     mean      stdev     min       25%       median    75%       max       Δ
------------------------------------------------------------------------------------------------------------------
       4,096  jax            6,375     0.001     0.000     0.001     0.001     0.001     0.001     0.007     1.990
       4,096  numpy          6,297     0.003     0.000     0.002     0.002     0.002     0.003     0.007     1.000

      16,384  jax            6,304     0.001     0.000     0.001     0.001     0.001     0.001     0.006     5.801
      16,384  numpy            637     0.008     0.000     0.008     0.008     0.008     0.008     0.012     1.000

      65,536  jax            6,370     0.002     0.000     0.002     0.002     0.002     0.002     0.007    17.927
      65,536  numpy             55     0.033     0.002     0.029     0.030     0.034     0.035     0.036     1.000

     262,144  jax              618     0.004     0.000     0.003     0.003     0.004     0.004     0.007    37.319
     262,144  numpy             64     0.134     0.001     0.133     0.134     0.134     0.135     0.140     1.000

   1,048,576  jax              656     0.013     0.000     0.012     0.012     0.013     0.013     0.015    46.704
   1,048,576  numpy              7     0.585     0.002     0.582     0.583     0.585     0.587     0.588     1.000

   4,194,304  jax               69     0.048     0.001     0.047     0.047     0.048     0.048     0.052    45.636
   4,194,304  numpy              3     2.176     0.005     2.172     2.172     2.172     2.178     2.183     1.000

(time in wall seconds, less is better)
```

## Conclusion

Lessons I learned by assembling these benchmarks: (your mileage may vary)

- The performance of Jax seems very competitive, both on GPU and CPU.
- Numba is a great choice on CPU if you don't mind writing explicit for loops (which can be more readable than a vectorized implementation).
- If you have embarrasingly parallel workloads, speedups of > 1000x are easy to achieve on high-end GPUs.
- Tensorflow is not great for applications like ours, since it lacks tools to apply partial updates to tensors (in the sense of `tensor[2:-2] = 0.`).
- Don't bother using Pytorch or Tensorflow on CPU.
- CuPy is nice! Often you don't need to change anything in your NumPy code to have it run on GPU (with decent, but not outstanding performance).
- Reaching Fortran performance on CPU with vectorized implementations is hard :)

## Contributing

Community contributions are encouraged! Whether you want to donate another benchmark, share your experience, optimize an implementation, or suggest another backend - [feel free to ask](https://github.com/dionhaefner/pyhpc-benchmarks/issues) or [open a PR](https://github.com/dionhaefner/pyhpc-benchmarks/pulls).
