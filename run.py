#!/usr/bin/env python3

import random
import itertools

import click

from backends import __backends__ as setup_functions, BackendNotSupported, convert_to_numpy
from utilities import (
    Timer, estimate_repetitions, format_output, compute_statistics,
    get_benchmark_module, check_consistency
)


DEFAULT_SIZE = tuple(2 ** i for i in range(12, 23, 2))


@click.command('run')
@click.argument(
    'BENCHMARK',
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True),
)
@click.option(
    '-s', '--size',
    required=False,
    multiple=True,
    default=DEFAULT_SIZE,
    show_default=True,
    type=click.INT,
    help=f'Run benchmark for this array size (repeatable)',
)
@click.option(
    '-b', '--backend',
    required=False,
    multiple=True,
    default=None,
    type=click.Choice(setup_functions.keys()),
    help='Run benchmark with this backend (repeatable) [default: run all backends]',
)
@click.option(
    '-r', '--repetitions',
    required=False,
    default=None,
    type=click.INT,
    help='Fixed number of iterations to run for each size and backend [default: auto-detect]',
)
@click.option(
    '--burnin',
    required=False,
    default=1,
    type=click.INT,
    show_default=True,
    help='Number of initial iterations that are disregarded for final statistics',
)
@click.option(
    '--device',
    required=False,
    default='cpu',
    type=click.Choice(['cpu', 'gpu', 'tpu']),
    show_default=True,
    help='Run benchmarks on given device where supported by the backend',
)
def main(benchmark, size=None, backend=None, repetitions=None, burnin=1, device='cpu'):
    """HPC benchmarks for Python

    Usage:

        $ python run.py benchmarks/<BENCHMARK_FOLDER>

    Examples:

        $ taskset -c 0 python run.py benchmarks/equation_of_state

        $ python run.py benchmarks/equation_of_state -b numpy -b jax --device gpu

    More information:

        https://github.com/dionhaefner/pyhpc-benchmarks

    """
    try:
        bm_module, bm_identifier = get_benchmark_module(benchmark)
    except ImportError as e:
        click.echo(
            f'Error while loading benchmark {benchmark}: {e!s}',
            err=True
        )
        raise click.Abort()

    available_backends = set(bm_module.__implementations__)

    if len(backend) == 0:
        backend = available_backends.copy()
    else:
        backend = set(backend)

    unsupported_backends = [b for b in backend if b not in available_backends]

    for b in unsupported_backends:
        click.echo(
            f'Backend "{b}" is not supported by chosen benchmark (skipping)',
            err=True
        )
        backend.remove(b)

    for b in backend.copy():
        try:
            with setup_functions[b](device=device):
                pass
        except BackendNotSupported as e:
            click.echo(
                f'Setup for backend "{b}" failed (skipping), reason: {e!s}',
                err=True
            )
            backend.remove(b)

    runs = sorted(itertools.product(backend, size))

    if len(runs) == 0:
        click.echo('Nothing to do')
        return

    timings = {run: [] for run in runs}

    if repetitions is None:
        click.echo('Estimating repetitions...')
        repetitions = {}
        for b, s in runs:
            with setup_functions[b](device=device):
                run = bm_module.get_callable(b, s, device=device)
                repetitions[(b, s)] = estimate_repetitions(run)
    else:
        repetitions = {(b, s): repetitions for b, s in runs}

    all_runs = list(itertools.chain.from_iterable(
        [run] * (repetitions[run] + burnin) for run in runs
    ))
    random.shuffle(all_runs)

    results = {}
    checked = {r: False for r in runs}

    pbar = click.progressbar(
        label=f'Running {len(all_runs)} benchmarks...', length=len(runs)
    )

    callables = {}

    try:
        with pbar:
            for (b, size) in all_runs:
                if (b, size) not in callables:
                    with setup_functions[b](device=device):
                        callables[(b, size)] = bm_module.get_callable(b, size, device=device)

                with setup_functions[b](device=device):
                    with Timer() as t:
                        res = callables[(b, size)]()

                # YOWO (you only warn once)
                if not checked[(b, size)]:
                    if size in results:
                        is_consistent = check_consistency(
                            results[size],
                            convert_to_numpy(res, b, device)
                        )
                        if not is_consistent:
                            click.echo(
                                f'\nWarning: inconsistent results for size {size}',
                                err=True
                            )
                    else:
                        results[size] = convert_to_numpy(res, b, device)
                    checked[(b, size)] = True

                timings[(b, size)].append(t.elapsed)
                pbar.update(1. / (repetitions[(b, size)] + burnin))

            # push pbar to 100%
            pbar.update(1.)

        for run in runs:
            assert len(timings[run]) == repetitions[run] + burnin

    finally:
        stats = compute_statistics(timings)
        click.echo(format_output(stats, bm_identifier, device=device))


if __name__ == '__main__':
    main()
