from itertools import product, repeat
import os


def mtscomp_perf(compression_algorithm='zlib', compression_level=None, chunk_duration=None, do_diff=True):
    return {'read_time': 0, 'write_time': 0, 'ratio': 1.}


params = {
    'compression_level': {
        'title': 'Compression level',
        'values': [1, 3, 6, 9],
        'plotdim': '',
    },
    'chunk_duration': {
        'title': 'Chunk duration (s)',
        'values': [.25, 1., 5.],
        'plotdim': '',
    },
    'do_diff': {
        'title': 'Differentiate',
        'values': [True, False],
    },
    'dataset': {
        'title': 'Dataset',
        'values': ['artif1', 'artif2', 'real', 'imec1', 'imec2'],
    },
}

targets = {
    'values': ['read_time', 'write_time', 'ratio'],
    'labels': ['Read time', 'Write time', 'Compression ratio'],
    'plotdim': 'plot',
}


def _iter_param_set(params):
    """Iterate over all combinations of parameters as dictionaries {param: value}."""
    yield from map(dict, product(*(zip(repeat(param), info['values']) for param, info in params.items())))


def benchmark_plots(fun, params=None, targets=None, output_dir=None):
    for param_values in _iter_param_set(params):
        print(param_values)


benchmark_plots(mtscomp_perf, params, targets)
