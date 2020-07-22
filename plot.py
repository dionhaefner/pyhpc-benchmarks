#!/usr/bin/env python3

import os
import re
from collections import defaultdict

import click
import matplotlib
matplotlib.use('Agg')

# stupid regex matching ahead
RE_RESULT = re.compile(r''.join([
    r'\s*',
    r'(?P<size>(?:\d|,)+)\s*',
    r'(?P<backend>\w+)\s*',
    *(
        rf'(?P<{name}>(?:\d|\.|,)+)\s*' for name in
        ('calls', 'mean', 'stdev', 'min', 'p25', 'median', 'p75', 'max', 'delta')
    ),
]))
RE_BENCHMARK = re.compile(r'benchmarks\.(?P<name>\w+)')
RE_PLATFORM = re.compile(r'Running on (?P<platform>\w+)')

BACKEND_COLORS = {
    'numpy': 'C0',
    'bohrium': 'C1',
    'cupy': 'C2',
    'jax': 'C3',
    'numba': 'C4',
    'pytorch': 'C5',
    'tensorflow': 'C6',
    'theano': 'C7',
}


def plot_results(records, benchmark, platform, outfile, plot_delta=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=80)

    this_record = records[(benchmark, platform)]
    last_coords = {}

    for backend, backend_values in this_record.items():
        x = backend_values['size']
        if plot_delta:
            y = backend_values['delta']
            ylabel = 'Relative speedup'
        else:
            y = backend_values['mean']
            ylabel = 'Mean runtime (s)'

        x, y = zip(*sorted(zip(x, y), key=lambda ix: ix[0]))

        plt.plot(
            x, y, 'o--', label=backend,
            color=BACKEND_COLORS[backend]
        )
        last_coords[backend] = (x[-1], y[-1])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.xlabel('Problem size (# elements)')
        plt.ylabel(ylabel)

        plt.xscale('log')
        plt.yscale('log')

    plt.title(f'Benchmark "{benchmark}" on {platform.upper()}')
    fig.canvas.draw()

    # add annotations, make sure they don't overlap
    last_text_pos = 0
    for backend, (x, y) in sorted(last_coords.items(), key=lambda k: k[1][1]):
        trans = ax.transData
        _, tp = trans.transform((0, y))
        tp = max(tp, last_text_pos + 15)
        _, y = trans.inverted().transform((0, tp))

        plt.annotate(
            backend, (x, y), xytext=(10, 0), textcoords='offset points',
            annotation_clip=False, color=BACKEND_COLORS[backend], va='center'
        )

        last_text_pos = tp

    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def _parse_int(string):
    return int(string.replace(',', '_'))


@click.command('plot')
@click.argument('INFILE', type=click.File('r'))
@click.option(
    '-o', '--outdir', required=True, type=click.Path(file_okay=False, writable=True),
    help='Output directory for plots'
)
@click.option(
    '--plot-delta', is_flag=True,
    help='Plot relative speedup instead of absolute runtime'
)
def main(infile, outdir, plot_delta):
    """Read a benchmark report from file or stdin and plot the results

    Example:

        $ python run.py benchmarks/equation_of_state > bench.txt

        $ python plot.py bench.txt -o plots

    """
    records = {}

    for line in infile:
        bench_match = RE_BENCHMARK.match(line)
        if bench_match:
            current_benchmark = bench_match.group('name')
            continue

        platform_match = RE_PLATFORM.match(line)
        if platform_match:
            current_platform = platform_match.group('platform')
            continue

        result_match = RE_RESULT.match(line)
        if not result_match:
            continue

        result_line = result_match.groupdict()
        backend = result_line['backend']

        key = (current_benchmark, current_platform)
        if key not in records:
            records[key] = {}

        if backend not in records[key]:
            records[key][backend] = defaultdict(list)

        record = records[key][backend]

        if _parse_int(result_line['size']) in record['size']:
            click.echo(
                f'Warning: duplicate entry for benchmark {current_benchmark} '
                f'on {current_platform}, backend {backend}, size {result_line["size"]} '
                '- skipping'
            )
            continue

        for rkey, rval in result_line.items():
            if rkey in ('calls', 'size'):
                rval = _parse_int(rval)
            elif rkey in ('mean', 'stdev', 'min', 'p25', 'median', 'p75', 'max', 'delta'):
                rval = float(rval)

            record[rkey].append(rval)

    os.makedirs(outdir, exist_ok=True)

    for benchmark, platform in records.keys():
        outfile = os.path.join(outdir, f'bench-{benchmark}-{platform}.png')
        plot_results(records, benchmark, platform, outfile, plot_delta)
        click.echo(f'Wrote {outfile}')


if __name__ == '__main__':
    main()
