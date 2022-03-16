# -*- coding: utf-8 -*-

"""mtscomp tests."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path

import numpy as np

from mtscomp import compress, decompress, lossy as ml

logger = logging.getLogger('mtscomp')
# add_default_handler('DEBUG')


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

_INT16_MAX = 32766
ERROR_THRESHOLD = .08  # the error is the % of values that differ by more than this percent

n_channels = 19
sample_rate = 1234.
duration = 5.67
normal_std = .25
time = np.arange(0, duration, 1. / sample_rate)
n_samples = len(time)
dtype = np.int16
np.random.seed(0)


def randn():
    return np.random.normal(loc=0, scale=normal_std, size=(n_samples, n_channels))


def white_sine():
    return np.sin(10 * time)[:, np.newaxis] + randn()


def colored_sine():
    arr = white_sine()
    try:
        from scipy.signal import filtfilt, butter
    except ImportError:
        logger.debug("Skip the filtering as SciPy is not available.")
        return arr
    b, a = butter(3, 0.05)
    arr = filtfilt(b, a, arr, axis=0)
    assert arr.shape == (n_samples, n_channels)
    return arr


def _write_arr(path, arr):
    """Write an array."""
    with open(path, 'wb') as f:
        arr.tofile(f)


def _to_int16(arr, M=None):
    M = M or np.abs(arr).max()
    arr = arr / M if M > 0 else arr
    assert np.all(np.abs(arr) <= 1.)
    arr16 = (arr * _INT16_MAX).astype(np.int16)
    return arr16


def _from_int16(arr, M):
    return arr * float(M / _INT16_MAX)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _import_mpl():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.grid"] = False
    sns.set_theme(style="white")
    return plt


def _show_img(ax, x, title, vmin=None, vmax=None):
    ax.imshow(x, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)


def _prepare_compare(lossless, lossy, t0, t1):
    assert isinstance(lossy, ml.LossyReader)

    sr = lossless.sample_rate
    i0 = int(round(t0 * sr))
    i1 = int(round(t1 * sr))

    lossless_img = ml._preprocess_default(lossless[i0:i1])
    lossy_img = lossy.get(t0, t1).T

    mM = lossy._svd.quantile
    return lossless_img, lossy_img, mM


def _compute_error(lossless_img, lossy_img, threshold=ERROR_THRESHOLD):
    x = lossless_img - lossy_img
    return (np.abs(x).ravel() > lossless_img.max() * threshold).mean()


def show_compare(lossless, lossy, t0, t1, threshold=ERROR_THRESHOLD, do_show=True):
    assert isinstance(lossy, ml.LossyReader)

    lossless_img, lossy_img, (m, M) = _prepare_compare(lossless, lossy, t0, t1)

    err = _compute_error(lossless_img, lossy_img, threshold=threshold)
    print(f"Relative error is {err * 100:.1f}%.")

    title = f"rank={lossy.rank}, {lossy.compression:.1f}x compression, error {err * 100:.1f}%"

    nrows = 2
    plt = _import_mpl()
    fix, axs = plt.subplots(nrows, 1, sharex=True)
    _show_img(axs[0], lossless_img, 'original', vmin=m, vmax=M)
    _show_img(axs[1], lossy_img, title, vmin=m, vmax=M)
    # _show_img(axs[2], lossless_img - lossy_img, 'residual', vmin=0, vmax=v)

    n_ticks = 5
    ticks = np.linspace(0, lossless_img.shape[1], n_ticks)
    labels = ['%.1f ms' % (t * 1000) for t in np.linspace(t0, t1, n_ticks)]
    plt.xticks(ticks, labels)
    if do_show:
        plt.show()

    return err


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_lossy_artificial(tmp_path):
    path_bin = tmp_path / 'sine.bin'
    path_cbin = tmp_path / 'sine.cbin'

    # Generate an artificial binary file.
    arr = colored_sine()
    assert arr.shape == (n_samples, n_channels)
    M = np.abs(arr).max()
    _write_arr(path_bin, _to_int16(arr, M))

    # Compress it (lossless).
    compress(path_bin, out=path_cbin, sample_rate=sample_rate, n_channels=n_channels, dtype=dtype)
    assert path_cbin.exists()

    # Compress it (lossy).
    rank = 8
    path_lossy = ml.compress_lossy(path_cbin=path_cbin, rank=rank)
    assert path_lossy.exists()
    assert np.load(path_lossy).shape == (n_samples // ml.DOWNSAMPLE_FACTOR, rank)

    # Decompress.
    lossy = ml.decompress_lossy(path_lossy)
    assert arr.ndim == lossy.ndim
    assert arr.shape[1] == lossy.shape[1]
    assert arr.shape[0] - lossy.shape[0] == arr.shape[0] % ml.DOWNSAMPLE_FACTOR

    lossless = decompress(path_cbin)

    err = show_compare(lossless, lossy, 0, duration, threshold=.1, do_show=False)
    assert err < 1


def test_lossy_local():

    EPHYS_DIR = Path(__file__).parent.resolve()
    path_cbin = EPHYS_DIR / "raw.cbin"
    if not path_cbin.exists():
        logger.warning(f"skip test because {path_cbin} does not exist")
        return

    rank = 40
    max_chunks = 10

    out_lossy = ml.compress_lossy(
        path_cbin=path_cbin,
        chunks_excerpts=5,
        rank=rank,
        max_chunks=max_chunks,
        overwrite=True,
        dry_run=False,
    )

    lossless = decompress(path_cbin)
    lossy = ml.decompress_lossy(out_lossy)

    # plt = _import_mpl()
    # x = lossless_img - lossy_img
    # plt.figure()
    # plt.hist(np.abs(x).ravel(), bins=100)

    hw = .1
    t = hw
    err = show_compare(lossless, lossy, t - hw, t + hw, do_show=True)
    assert err < 1


def preprocess(raw):
    """Default preprocessing function: CAR and 6x downsampling.

    Note: this function transposes the array.

    Parameters
    ----------

    raw : ndarray (n_samples, n_channels)
        Signal.

    Returns
    -------

    pp : ndarray (n_channels, n_samples)

    """

    assert raw.shape[0] > raw.shape[1]
    # raw is (ns, nc)

    pp = ml._car(raw)
    # pp is (ns, nc)
    assert pp.shape[0] > pp.shape[1]

    pp = ml._downsample(pp, factor=ml.DOWNSAMPLE_FACTOR)
    # pp is (nc', ns)
    assert pp.shape[0] < pp.shape[1]

    return pp


def test_lossy_array(tmp_path):
    path_lossy = tmp_path / 'sine.lossy.npy'
    path_svd = tmp_path / 'sine.svd.npz'

    # Generate an artificial binary file.
    arr = colored_sine()
    assert arr.shape == (n_samples, n_channels)
    # M = np.abs(arr).max()

    # mtscomp-like interface for a regular NumPy array.
    reader = ml.ArrayReader(arr, sample_rate=sample_rate)

    # Compress it (lossy).
    rank = 8
    path_lossy = ml.compress_lossy(
        reader=reader, rank=rank, out_lossy=path_lossy, out_svd=path_svd,
        preprocess=preprocess)
    assert path_lossy.exists()
    assert np.load(path_lossy).shape == (n_samples // ml.DOWNSAMPLE_FACTOR, rank)

    # Decompress.
    lossy = ml.decompress_lossy(path_lossy)
    assert arr.ndim == lossy.ndim
    assert arr.shape[1] == lossy.shape[1]
    assert arr.shape[0] - lossy.shape[0] == arr.shape[0] % ml.DOWNSAMPLE_FACTOR

    err = show_compare(reader, lossy, 0, duration, threshold=.1, do_show=False)
    assert err < 1
