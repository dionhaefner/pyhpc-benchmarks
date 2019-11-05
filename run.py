#!/usr/bin/env python3

import importlib
import random
import os
import itertools
import collections

import click
import numpy as np

from backends import __backends__ as setup_functions, BackendNotSupported, convert_to_numpy
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
            (size, backend, repetitions, mean, stdev, *percentiles, float('nan'))
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
        ('Δ', 'f4'),
    ])

    # add deltas
    sizes = np.unique(stats['size'])
    for s in sizes:
        mask = stats['size'] == s

        # measure relative to NumPy if present, otherwise worst backend
        if 'numpy' in stats['backend'][mask]:
            reference_time = stats['mean'][mask & (stats['backend'] == 'numpy')]
        else:
            reference_time = np.nanmax(stats['mean'][mask])

        stats['Δ'][mask] = reference_time / stats['mean'][mask]

    return stats


def format_output(stats, benchmark_title, gpu=False):
    stats = np.sort(stats, axis=0, order=['size', 'mean', 'max', 'median'])

    header = stats.dtype.names
    col_widths = collections.defaultdict(lambda: 8)
    col_widths.update(size=12, backend=10)

    def format_col(col_name, value, is_time=False):
        col_width = col_widths[col_name]

        if np.issubdtype(type(value), np.integer):
            typecode = ','
        else:
            typecode = '.3f'

        if is_time:
            format_string = f'{{value:>{col_width}{typecode}}}'
        else:
            format_string = f'{{value:<{col_width}}}'

        return format_string.format(value=value)

    out = [
        '',
        benchmark_title,
        '=' * len(benchmark_title),
        f'Running on {"GPU" if gpu else "CPU"}',
        '',
        '  '.join(format_col(s, s) for s in header)
    ]

    out.append('-' * len(out[-1]))

    current_size = None
    for row in stats:
        # print empty line on size change
        size = row[0]
        if current_size is not None and size != current_size:
            out.append('')
        current_size = size

        out.append(
            '  '.join(
                format_col(n, s, not isinstance(s, str))
                for n, s in zip(header, row)
            )
        )

    out.extend([
        '',
        '(time in wall seconds, less is better)',
    ])

    return '\n'.join(out)


def get_benchmark_module(file_path):
    base_path = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.relpath(file_path, base_path)
    import_path = '.'.join(os.path.split(module_path))
    bm_module = importlib.import_module(import_path)
    return bm_module, import_path


def check_consistency(res1, res2):
    if isinstance(res1, (tuple, list)):
        if not len(res1) == len(res2):
            return False

        return all(check_consistency(r1, r2) for r1, r2 in zip(res1, res2))

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    return np.allclose(res1, res2)


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
@click.option(
    '--burnin',
    required=False,
    default=1,
    type=click.INT
)
@click.option(
    '--gpu',
    required=False,
    default=False,
    is_flag=True
)
def main(benchmark, size=None, backend=None, repetitions=None, burnin=1, gpu=False):
    if len(size) == 0:
        size = [2 ** i for i in range(12, 23, 2)]

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
            with setup_functions[b](gpu=gpu):
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
            with setup_functions[b](gpu=gpu):
                run = bm_module.get_callable(b, s, gpu=gpu)
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

    try:
        with pbar:
            for (b, size) in all_runs:
                with setup_functions[b](gpu=gpu):
                    run = bm_module.get_callable(b, size, gpu=gpu)
                    with Timer() as t:
                        res = run()

                if not checked[(b, size)]:
                    if size in results:
                        is_consistent = check_consistency(
                            results[size],
                            convert_to_numpy(res, b, gpu)
                        )
                        if not is_consistent:
                            click.echo(
                                f'\nWarning: inconsistent results for size {size}',
                                err=True
                            )
                    else:
                        results[size] = convert_to_numpy(res, b, gpu)
                    checked[(b, size)] = True

                timings[(b, size)].append(t.elapsed)
                pbar.update(1. / (repetitions[(b, size)] + burnin))

            # push pbar to 100%
            pbar.update(1.)

        for run in runs:
            assert len(timings[run]) == repetitions[run] + burnin

    finally:
        stats = compute_statistics(timings)
        click.echo(format_output(stats, bm_identifier, gpu=gpu))


if __name__ == '__main__':
    main()
