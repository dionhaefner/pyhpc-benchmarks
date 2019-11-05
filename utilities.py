import time
import math


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

    # call again and measure time
    with Timer() as t:
        func(*args)

    time_per_rep = t.elapsed
    exponent = math.log(target_time / time_per_rep, powers_of)
    num_reps = int(powers_of ** round(exponent))
    return max(powers_of, num_reps)
