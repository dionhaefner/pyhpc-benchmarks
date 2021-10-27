import time
import math
import collections
import importlib
import os

import numpy as np


class Timer:
    def __init__(self):
        self.elapsed = float('nan')

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if value is None:
            self.elapsed = time.perf_counter() - self._start


def estimate_repetitions(func, args=(), target_time=10, powers_of=10):
    # call function once for warm-up
    func(*args)

    # some backends need an extra nudge (looking at you, PyTorch)
    func(*args)

    # call again and measure time
    with Timer() as t:
        func(*args)

    time_per_rep = t.elapsed
    exponent = math.log(target_time / time_per_rep, powers_of)
    num_reps = int(powers_of ** round(exponent))
    return max(powers_of, num_reps)


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


def format_output(stats, benchmark_title, device='cpu'):
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
        f'Running on {device.upper()}',
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
