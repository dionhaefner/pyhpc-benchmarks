# pyhpc-benchmarks

This is a suite of benchmarks to test the *sequential CPU* and GPU performance of various computational backends for Python.

## Why?



## Supported backends

- NumPy
- Numba
- Jax
- Tensorflow
- Pytorch
- Theano
- Bohrium
- CuPy

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

## Usage

Your entrypoint is the script `run.py`:

```bash
$ python run.py --help
```

Benchmarks are run for all combinations of the chosen sizes (`-s`) and backends (`-b`), in random order.

### CPU

```bash
$ conda activate pyhpc-bench-cpu
$ taskset -c 0 python run.py benchmarks/<benchmark_name>
```

### GPU

```bash
$ conda activate pyhpc-bench-gpu
$ CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/<benchmark_name> --gpu
```

## Example results
