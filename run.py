#!/usr/bin/env python3

import importlib
import random
import os
import itertools

import click
import numpy as np

from backends import __backends__, BackendNotSupported
from utilities import Timer, estimate_repetitions


def compute_statistics(timings, burnin=1):
    stats = []

    for (backend, size), t in timings.items():
        t = t[burnin:]
        repetitions = len(t)

        if repetitions:
            mean = np.mean(t)
            stdev = np.std(t)
            percentiles = np.percentile(t, [0, 25, 50, 75, 100])
        else:
            mean = stdev = float('nan')
            percentiles = [float('nan')] * 5

        stats.append(
            (size, backend, repetitions, mean, stdev, *percentiles)
        )

    stats = np.array(stats, dtype=[
        ('size', 'i8'),
        ('backend', object),
        ('calls', 'i8'),
        ('mean', 'f4'),
        ('stdev', 'f4'),
        ('min', 'f4'),
        ('25%', 'f4'),
        ('median', 'f4'),
        ('75%', 'f4'),
        ('max', 'f4'),
    ])

    return stats


def format_output(stats, benchmark_title):
    stats = np.sort(stats, axis=0, order=['size', 'mean', 'max', 'median'])

    header = stats.dtype.names
    col_width = max(len(x) for x in header) + 2

    def format_col(s, is_time=False):
        if np.issubdtype(type(s), np.integer):
            typecode = 'd'
        else:
            typecode = '.3f'
        if is_time:
            format_string = f'{{s:>{col_width}{typecode}}}'
        else:
            format_string = f'{{s:<{col_width}}}'
        return format_string.format(s=s)

    out = [
        '',
        benchmark_title,
        '=' * len(benchmark_title),
        '',
        '  '.join(format_col(s) for s in header)
    ]

    out.append('-' * len(out[-1]))

    for row in stats:
        out.append(
            '  '.join(format_col(s, not isinstance(s, str)) for s in row)
        )

    out.extend([
        '',
        '(timings in seconds)',
        ''
    ])

    return '\n'.join(out)


@click.command('run')
@click.argument(
    'BENCHMARK',
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True)
)
@click.option(
    '-s', '--size',
    required=False,
    multiple=True,
    default=None,
    type=click.INT
)
@click.option(
    '-b', '--backend',
    required=False,
    multiple=True,
    default=None
)
@click.option(
    '-r', '--repetitions',
    required=False,
    default=None,
    type=click.INT
)
def main(benchmark, size=None, backend=None, repetitions=None):
    if len(size) == 0:
        size = [2 ** i for i in range(10, 18)]

    try:
        file_path = benchmark
        base_path = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.relpath(file_path, base_path)
        import_path = '.'.join(os.path.split(module_path))
        bm_module = importlib.import_module(import_path)
    except ImportError as e:
        click.echo(f'Error while loading benchmark {benchmark}: {e!s}', err=True)
        raise click.Abort()

    available_backends = set(bm_module.__implementations__)

    if len(backend) == 0:
        backend = available_backends.copy()
    else:
        backend = set(backend)

    unsupported_backends = [b for b in backend if b not in available_backends]

    for b in unsupported_backends:
        click.echo(f'Backend "{b}" is not supported by chosen benchmark', err=True)
        backend.remove(b)

    runs = sorted(itertools.product(backend, size))

    if len(runs) == 0:
        click.echo('Nothing to do')
        return

    timings = {run: [] for run in runs}
    run_benchmark = bm_module.run
    setup_functions = __backends__

    if repetitions is None:
        click.echo('Estimating repetitions...')
        repetitions = {}
        for b, s in runs:
            with setup_functions[b]():
                repetitions[(b, s)] = estimate_repetitions(run_benchmark, [b, s])
    else:
        repetitions = {(b, s): repetitions for b, s in runs}

    all_runs = list(itertools.chain.from_iterable(
        [run] * (repetitions[run] + 1) for run in runs
    ))
    all_runs = random.sample(all_runs, k=len(all_runs))

    results = {}

    click.echo(f'Running {len(all_runs)} benchmarks...')
    try:
        with click.progressbar(all_runs) as pbar:
            for (b, size) in pbar:
                with setup_functions[b](), Timer() as t:
                    res = run_benchmark(b, size)

                if (b, size) in results:
                    if not np.allclose(results[(b, size)], res):
                        click.echo(f'Warning: inconsistent results for {fun.__name__}({size})', err=True)

                results[(b, size)] = res
                timings[(b, size)].append(t.elapsed)

        for run in runs:
            assert len(timings[run]) == repetitions[run] + 1

    finally:
        stats = compute_statistics(timings)
        click.echo(format_output(stats, import_path))


if __name__ == '__main__':
    main()
