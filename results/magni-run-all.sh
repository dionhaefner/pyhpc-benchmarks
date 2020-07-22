#!/bin/bash -l

set -e

ml load bohrium
ml unload python
conda activate bench-2020
conda list

export BH_CONFIG=$HOME/.bohrium/config.ini
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/groups/ocean/software/software/nvtoolkit/cuda-10.1"

cd `git rev-parse --show-toplevel`

CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/equation_of_state/ --device cpu
CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/equation_of_state/ --device cpu -s 16777216
for backend in bohrium cupy jax pytorch tensorflow theano; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/equation_of_state/ --device gpu -b $backend -b numpy; done

CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/isoneutral_mixing/ --device cpu
CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/isoneutral_mixing/ --device cpu -s 16777216
for backend in bohrium cupy jax pytorch theano; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/isoneutral_mixing/ --device gpu -b $backend -b numpy; done

CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/turbulent_kinetic_energy/ --device cpu
CUDA_VISIBLE_DEVICES="" taskset -c 23 python run.py benchmarks/turbulent_kinetic_energy/ --device cpu -s 16777216
for backend in bohrium jax; do CUDA_VISIBLE_DEVICES="0" python run.py benchmarks/turbulent_kinetic_energy/ --device gpu -b $backend -b numpy; done

