from itertools import product, repeat
from pathlib import Path
import time

from tabulate import tabulate
import numpy as np
from joblib import Memory
from tqdm import tqdm

from mtscomp import compress, decompress, load_raw_data


dtype = np.uint16


def mtscomp_perf(**kwargs):
    ds = kwargs.pop('ds', None)
    assert ds

    name, n_channels, sample_rate, duration = ds

    # Compress the file.
    path = Path('data/' + name)
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'
    t0 = time.perf_counter()
    compress(
        path, out, outmeta, sample_rate=sample_rate, n_channels=n_channels, dtype=dtype,
        check_after_compress=False, **kwargs)
    t1 = time.perf_counter()
    wt = t1 - t0

    # Decompress the file and write it to disk.
    out2 = path.with_suffix('.decomp.bin')
    t0 = time.perf_counter()
    decompress(out, outmeta, out2, check_after_decompress=False)
    t1 = time.perf_counter()
    rtc = t1 - t0

    # Read the uncompressed file.
    t0 = time.perf_counter()
    x = load_raw_data(path, n_channels=n_channels, dtype=dtype, mmap=False)
    assert x.size
    t1 = time.perf_counter()
    rtdec = t1 - t0

    orig_size = path.stat().st_size
    compressed_size = out.stat().st_size

    return {
        'read_time_compressed': rtc,
        'read_time_decompressed': rtdec,
        'write_time': wt,
        'ratio': 100 - 100 * compressed_size / orig_size,
    }


params = {
    'ds': {
        'title': 'dataset',
        'values': [
            # ('imec_385_1s.bin', 385, 3e4, 1.),
            # ('imec_385_10s.bin', 385, 3e4, 10.),
            ('imec_385_100s.bin', 385, 3e4, 100.),
            # ('pierre_10s.bin', 256, 2e4, 10.),
        ],
    },
    'n_threads': {
        'title': 'n_threads',
        'values': [1, 4, 8]
    },
    # 'do_time_diff': {
    #     'title': 'time diff',
    #     'values': [False, True],
    # },
    # 'do_spatial_diff': {
    #     'title': 'spatial diff',
    #     'values': [False, True],
    # },

    # 'compression_level': {
    #     'title': 'Compression level',
    #     'values': [-1],
    # },
    # 'chunk_duration': {
    #     'title': 'Chunk duration (s)',
    #     'values': [.1, 1, 10],
    # },
}

targets = {
    'values': ['write_time', 'read_time_compressed', 'read_time_decompressed', 'ratio'],
}


def _iter_param_set(params):
    """Iterate over all combinations of parameters as dictionaries {param: value}."""
    yield from map(
        dict, product(*(zip(repeat(param), info['values']) for param, info in params.items())))


class PlotParams:
    def __init__(self, fun, params, targets):
        self.fun = fun
        self.params = params
        self.targets = targets

        self.plot_param = self._get_param_for_plotdim('plot')
        self.row_param = self._get_param_for_plotdim('row')
        self.column_param = self._get_param_for_plotdim('column')
        self.group_param = self._get_param_for_plotdim('group')
        self.bar_param = self._get_param_for_plotdim('bar')

        self.target_plotdim = self.targets.get('plotdim', '')
        self.target_values = self.targets.get('values', [])

        self.plot_values = self._get_param_values(self.plot_param)
        self.row_values = self._get_param_values(self.row_param)
        self.column_values = self._get_param_values(self.column_param)
        self.group_values = self._get_param_values(self.group_param)
        self.bar_values = self._get_param_values(self.bar_param)

        self.n_plots = len(self.plot_values) or 1
        self.n_rows = len(self.row_values) or 1
        self.n_columns = len(self.column_values) or 1
        self.n_groups = len(self.group_values) or 1
        self.n_bars = len(self.bar_values) or 1

    def _get_param_values(self, param):
        if param == 'target':
            return self.target_values
        return self.params.get(param, {}).get('values', [])

    def _get_param_for_plotdim(self, plotdim):
        if self.targets.get('plotdim', '') == plotdim:
            return 'target'
        try:
            return next(p for p, i in self.params.items() if i.get('plotdim', '') == plotdim)
        except StopIteration:
            return

    def _get_target(self, plot_idx=0, row=0, column=0, group_idx=0, bar_idx=0):
        if self.target_plotdim == 'plot':
            target_idx = plot_idx
        elif self.target_plotdim == 'row':
            target_idx = row
        elif self.target_plotdim == 'column':
            target_idx = column
        elif self.target_plotdim == 'group':
            target_idx = group_idx
        return target_idx

    def get_plot_value(self, plot_idx=0, row=0, column=0, group_idx=0, bar_idx=0):
        target_idx = self._get_target(
            plot_idx=plot_idx, row=row, column=column, group_idx=group_idx, bar_idx=bar_idx)
        params = {
            self.row_param: self.row_values[row],
            self.column_param: self.column_values[column],
            self.group_param: self.group_values[group_idx],
            self.plot_param: self.plot_values[plot_idx],
            self.bar_param: self.bar_values[bar_idx],
        }
        params.pop('target', None)
        return self.fun(**params).get(self.target_values[target_idx], 0)

    def make(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(
            self.n_rows, self.n_columns,
        )
        index = np.arange(self.n_groups)
        bar_width = .75 / self.n_bars

        for row in range(self.n_rows):
            for column in range(self.n_columns):
                if self.n_columns == self.n_rows == 1:
                    ax = axes
                elif self.n_columns == 1:
                    ax = axes[row]
                else:
                    ax = axes[row, column]
                target_idx = self._get_target(
                    plot_idx=0, row=row, column=column)
                for bar in range(self.n_bars):
                    values = [
                        self.get_plot_value(
                            row=row, column=column, group_idx=group, bar_idx=bar)
                        for group in range(self.n_groups)
                    ]
                    label = self.bar_values[bar]
                    ax.bar(index + bar_width * bar, values, bar_width, label=label)

                ax.set_xlabel(self._get_param_for_plotdim('group'))
                ax.set_ylabel(self.target_values[target_idx])
                ax.set_xticks(index + bar_width, map(str, self.group_values))
        return fig


def benchmark_plots(fun, params=None, targets=None, output_dir=None):
    params = params or {}
    pp = PlotParams(fun, params, targets)
    fig = pp.make()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    location = '.cache'
    memory = Memory(location, verbose=0)
    fun = memory.cache(mtscomp_perf)
    N = len(list(_iter_param_set(params)))
    table = []
    for param_set in tqdm(_iter_param_set(params), total=N):
        output = fun(**param_set)
        d = {k: v for k, v in param_set.items() if len(params[k]['values']) > 1}
        d.update(output)
        table.append(d)

    print(tabulate(table, headers='keys'))
