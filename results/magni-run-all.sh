#!/bin/bash -l

set -e

ml load nvtoolkit
conda activate pyhpc-bench-gpu
conda list

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/groups/ocean/software/software_gcc2020/nvtoolkit/11.2.2"

cd `git rev-parse --show-toplevel`

CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/equation_of_state/ --device cpu
CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/equation_of_state/ --device cpu -s 16777216
for backend in cupy jax pytorch tensorflow; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/equation_of_state/ --device gpu -b $backend -b numpy; done

CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/isoneutral_mixing/ --device cpu
CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/isoneutral_mixing/ --device cpu -s 16777216
for backend in cupy jax pytorch; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/isoneutral_mixing/ --device gpu -b $backend -b numpy; done

CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/turbulent_kinetic_energy/ --device cpu
CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/turbulent_kinetic_energy/ --device cpu -s 16777216
for backend in jax pytorch; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/turbulent_kinetic_energy/ --device gpu -b $backend -b numpy; done
