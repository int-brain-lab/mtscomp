# -*- coding: utf-8 -*-

"""mtscomp tests."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import product
import logging
from pathlib import Path

import numpy as np
from pytest import fixture, raises, mark

from mtscomp import add_default_handler, Writer, Reader, load_raw_data, compress, uncompress

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

n_channels = 19
sample_rate = 1234.
duration = 5.67
normal_std = .25
time = np.arange(0, duration, 1. / sample_rate)
n_samples = len(time)


add_default_handler('DEBUG')


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def path(tmp_path):
    return Path(tmp_path) / 'data.bin'


def zeros():
    return np.zeros((n_samples, n_channels))


def randn():
    return np.random.normal(loc=0, scale=normal_std, size=(n_samples, n_channels))


def randn_custom(ns, nc):
    return np.random.normal(loc=0, scale=normal_std, size=(ns, nc))


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


@fixture(params=('zeros', 'randn', 'white_sine', 'colored_sine'))
def arr(request):
    return globals()[request.param]()


@fixture(params=('uint8', 'uint16', 'int8', 'int16', 'int32'))
def dtype(request):
    return np.dtype(request.param)


#------------------------------------------------------------------------------
# Test utils
#------------------------------------------------------------------------------

_INT16_MAX = 32766


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


def _round_trip(path, arr, **ckwargs):
    _write_arr(path, arr)
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'
    compress(
        path, out, outmeta, sample_rate=sample_rate, n_channels=arr.shape[1],
        dtype=arr.dtype, **ckwargs)
    unc = uncompress(out, outmeta)
    assert np.allclose(unc[:], arr)
    return unc


#------------------------------------------------------------------------------
# Misc tests
#------------------------------------------------------------------------------

def test_load_raw_data(path):
    arrs = [
        np.zeros((0, 1)),
        np.zeros((1, 1)),
        np.zeros((10, 1)),
        np.zeros((10, 10)),
        np.random.randn(100, 10),

        np.random.randn(100, 10).astype(np.float32),
        (np.random.randn(100, 10) * 100).astype(np.int16),
        np.random.randint(low=0, high=255, size=(100, 10)).astype(np.uint8),
    ]
    for arr in arrs:
        for mmap in (True, False):
            with open(path, 'wb') as f:
                arr.tofile(f)
            n_channels = arr.shape[1] if arr.ndim >= 2 else 1
            loaded = load_raw_data(path=path, n_channels=n_channels, dtype=arr.dtype, mmap=mmap)
            assert np.array_equal(arr, loaded)


def test_int16(arr):
    M = np.abs(arr).max()
    arr16 = _to_int16(arr, M=M)
    arr_ = _from_int16(arr16, M)
    assert np.allclose(arr_, arr, atol=1e-4)


#------------------------------------------------------------------------------
# Read/write tests
#------------------------------------------------------------------------------

def test_low(path, arr):
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'

    # Write the array into a raw data binary file.
    _write_arr(path, arr)

    # Compress the file.
    w = Writer()
    w.open(path, sample_rate=sample_rate, n_channels=arr.shape[1], dtype=arr.dtype)
    w.write(out, outmeta)
    w.close()

    # Load the compressed file.
    r = Reader()
    r.open(out, outmeta)
    uncomp = r[:]
    r.close()

    # Ensure the array are equal.
    assert np.allclose(arr, uncomp)


def test_high(path, arr):
    _round_trip(path, arr)


def test_dtypes(path, dtype):
    # Test various int dtypes.
    arr = np.array(np.random.randint(low=0, high=255, size=(1000, 100)), dtype=dtype).T
    _round_trip(path, arr)


def test_reader_indexing(path, arr):
    # Write the array into a raw data binary file.
    M = np.abs(arr).max()
    arr16 = _to_int16(arr, M)
    unc = _round_trip(path, arr16)
    # Index with many different items.
    N = n_samples

    # First, degenerate slices.
    items = [
        slice(start, stop, step) for start, stop, step in product(
            (None, 0, 1, -1), (None, 0, 1, -1), (None, 2, 3, N // 2, N))]

    # Other slices with random numbers.
    X = np.random.randint(low=-100, high=2 * N, size=(100, 3))
    items.extend([slice(start, stop, step) for start, stop, step in X])

    items.extend([
        (slice(None, None, None),),
        (slice(None, None, None), slice(1, -1, 2)),
        (slice(None, None, None), [1, 5, 3]),
    ])

    # Single integers.
    items.extend([0, 1, N - 2, N - 1])  # N, N + 1, -1, -2])
    items.extend(np.random.randint(low=-N, high=N, size=100).tolist())

    # For every item, check the uncompression.
    for s in items:
        if isinstance(s, slice) and s.step is not None and s.step <= 0:
            continue
        # If the indexing fails, ensures the same indexing fails on the Reader.
        try:
            expected = arr16[s]
        except IndexError:
            with raises(IndexError):
                unc[s]
                continue
        sliced = unc[s]
        assert sliced.dtype == expected.dtype
        assert sliced.shape == expected.shape
        assert np.array_equal(sliced, expected)
        assert np.allclose(_from_int16(sliced, M), arr[s], atol=1e-4)


#------------------------------------------------------------------------------
# Read/write tests with different parameters
#------------------------------------------------------------------------------

@mark.parametrize('chunk_duration', [.01, .1, 1., 10.])
def test_chunk_duration(path, arr, chunk_duration):
    _round_trip(path, arr, chunk_duration=chunk_duration)


@mark.parametrize('ns', [0, 1, 100, 10000])
@mark.parametrize('nc', [0, 1, 10, 100])
def test_n_channels(path, ns, nc):
    arr = randn_custom(ns, nc)
    if 0 in (ns, nc):
        with raises(Exception):
            _round_trip(path, arr)
    else:
        _round_trip(path, arr)


@mark.parametrize('do_diff', [True, False])
@mark.parametrize('compression_level', [1, 3, 6, 9])
def test_compression_levels_do_diff(path, arr, compression_level, do_diff):
    _round_trip(path, arr, compression_level=compression_level, do_diff=do_diff)
