# -*- coding: utf-8 -*-

"""mtscomp tests."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import redirect_stdout
import io
from itertools import product
import json
import hashlib
import logging
import os
import os.path as op
from pathlib import Path
import re

import numpy as np
from pytest import fixture, raises, mark

import mtscomp as mtscomp_mod
from mtscomp import (
    add_default_handler,
    Writer, Reader, load_raw_data, diff_along_axis, cumsum_along_axis,
    mtscomp_parser, mtsdecomp_parser, _args_to_config, read_config,
    compress, decompress, mtsdesc, mtscomp, mtsdecomp, mtschop,
    CHECK_ATOL)

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
def tmp_path_(tmp_path):
    # HACK: do not use user config path in tests
    mtscomp_mod.CONFIG_PATH = tmp_path / '.mtscomp'
    return tmp_path


@fixture
def path(tmp_path_):
    return Path(tmp_path_) / 'data.bin'


def zeros():
    return np.zeros((n_samples, n_channels), dtype=np.float32)


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
    unc = decompress(out, outmeta)
    assert np.allclose(unc[:], arr)
    return unc


def sha1(buf):
    sha = hashlib.sha1()
    sha.update(buf)
    return sha.hexdigest()


#------------------------------------------------------------------------------
# Misc tests
#------------------------------------------------------------------------------

def test_config_1(tmp_path_):
    """Test default options/"""
    config = read_config()
    assert config.check_after_compress
    assert config.check_after_decompress
    assert config.do_time_diff
    assert not config.do_spatial_diff


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
            del loaded  # Close memmap


def test_int16(arr):
    M = np.abs(arr).max()
    arr16 = _to_int16(arr, M=M)
    arr_ = _from_int16(arr16, M)
    assert np.allclose(arr_, arr, atol=1e-4)


@mark.parametrize('ax1', [None, 0, 1])
@mark.parametrize('ax2', [None, 0, 1])
def test_diff_cumsum_1(arr, ax1, ax2):
    if ax1 == ax2 and ax1 is not None:
        # Skip double diff along the same axis.
        return
    arrd = diff_along_axis(arr, axis=ax1)
    arrd = diff_along_axis(arrd, axis=ax2)
    arr2 = cumsum_along_axis(arrd, axis=ax2)
    arr2 = cumsum_along_axis(arr2, axis=ax1)
    assert arr.shape == arr2.shape
    assert arr.dtype == arr2.dtype
    if np.issubdtype(arr.dtype, np.integer):
        assert np.array_equal(arr, arr2)
    else:
        assert np.allclose(arr, arr2, atol=CHECK_ATOL)


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
    # Note: out and outmeta should be set by default to the values specified above.
    w.write(None, None)
    w.close()

    # Load the compressed file.
    r = Reader()
    r.open(out, outmeta)
    decomp = r[:]
    r.close()

    # Ensure the array are equal.
    assert np.allclose(arr, decomp)


def test_high(path, arr):
    _round_trip(path, arr)


def test_dtypes(path, dtype):
    # Test various int dtypes.
    arr = np.array(np.random.randint(low=0, high=255, size=(1000, 100)), dtype=dtype).T
    _round_trip(path, arr)


def test_reader_indexing_1(path, arr):
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
        (slice(None, None, None), 1),
        (1, slice(None, None, None)),
        (2, 1),
    ])

    # Single integers.
    items.extend([0, 1, N - 2, N - 1])  # N, N + 1, -1, -2])
    items.extend(np.random.randint(low=-N, high=N, size=100).tolist())

    # Different integer types
    inttypes = [np.uint64, np.int64, np.int8, int]
    for t1 in inttypes:
        for t2 in inttypes:
            items.append(slice(t1(1), t2(3))) # Slices
            for t3 in inttypes:
                items.append(slice(t1(5), t2(9), t3(2))) # Slices with steps

    # For every item, check the decompression.
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


def test_reader_indexing_2(path, arr):
    # Write the array into a raw data binary file.
    M = np.abs(arr).max()
    arr16 = _to_int16(arr, M)
    unc = _round_trip(path, arr16)

    expected = [
        (-1, 2, 0, 0),
        (0, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 0, 0),
        (2, 1, 0, 0),
        (2, -1, 0, 0),
        (2, 2, 0, 0),
        (2, 2, 0, 0),

        (1233, 1233, 0, 0),
        (1233, 1234, 0, 1),
        (1234, 1234, 1, 1),
        (1234, 1235, 1, 1),


        (1233, 1233, 0, 0),
        (1233, 1234, 0, 1),
        (1234, 1234, 1, 1),
        (1234, 1235, 1, 1),

        (-10000, 10000, 0, 5),
        (0, 10000, 0, 5),

        (1233, 10000, 0, 5),
        (1234, 10000, 1, 5),

        (6996, 10000, 5, 5),
        (6997, 10000, 5, 5),
        (6998, 10000, 5, 5),
    ]

    for i0, i1, c0, c1 in expected:
        assert unc._chunks_for_interval(i0, i1) == (c0, c1)


def test_check_fail(path, arr):
    """Check that compression fails if we change one byte in the original file before finishing
    the write() method."""
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'

    # Write the array into a raw data binary file.
    _write_arr(path, arr)

    def before_check(writer):
        # Change one byte in the original file.
        # First, we need to close the original memmapped file before we can write to it.
        writer.close()
        # Then, we change one byte in it.
        f_size = op.getsize(path)
        # WARNING: must open in r+b mode in order to modify bytes in-place in the middle of
        # the file.
        with open(str(path), 'r+b') as f:
            f.seek(f_size // 2)
            # Also, it's better to change multiple bytes at the same time to be sure that
            # the underlying number (e.g. float64) is fully modified.
            f.write(os.urandom(8))
        assert not np.allclose(np.fromfile(str(path), dtype=arr.dtype), arr.ravel())
        # The file size should be the same, although one byte has been changed.
        assert op.getsize(path) == f_size
        # Finally, we open it again before the check.
        writer.open(path, sample_rate=sample_rate, n_channels=arr.shape[1], dtype=arr.dtype)

    # Compress the file.
    with raises(RuntimeError):
        w = Writer(before_check=before_check)
        w.open(path, sample_rate=sample_rate, n_channels=arr.shape[1], dtype=arr.dtype)
        w.write(out, outmeta)
        w.close()


def test_comp_decomp(path):
    """Compress and decompress a random binary file with integer data type, and check the files
    are byte to byte equal. This would not work for floating-point data types."""
    arr = np.array(np.random.randint(low=0, high=255, size=(1000, 1000)), dtype=np.int16).T
    _write_arr(path, arr)
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'
    compress(
        path, out, outmeta, sample_rate=sample_rate, n_channels=arr.shape[1], dtype=arr.dtype,
    )
    decompressed_path = path.with_suffix('.decomp.bin')
    decompress(out, outmeta, out=decompressed_path)

    # Check the files are equal.
    with open(str(path), 'rb') as f:
        buf1 = f.read()
        sha1_original = sha1(buf1)
    with open(str(decompressed_path), 'rb') as f:
        buf2 = f.read()
        sha1_decompressed = sha1(buf2)
    assert buf1 == buf2

    # Check the SHA1s.
    with open(str(out), 'rb') as f:
        sha1_compressed = sha1(f.read())
    with open(str(outmeta), 'r') as f:
        meta = json.load(f)

    assert meta['sha1_compressed'] == sha1_compressed
    assert meta['sha1_uncompressed'] == sha1_decompressed == sha1_original


def test_decompress_pool(path, arr):
    _write_arr(path, arr)
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'
    compress(
        path, out, outmeta, sample_rate=sample_rate, n_channels=arr.shape[1], dtype=arr.dtype,
        check_after_compress=False)
    reader = decompress(out, outmeta, cache_size=2)

    pool = reader.start_thread_pool()
    d1 = reader.decompress_chunks([0, 1, 2], pool=pool)
    d2 = reader.decompress_chunks([1, 2, 3], pool=pool)
    d3 = reader.decompress_chunks([0, 1, 3], pool=pool)
    reader.stop_thread_pool()

    assert sorted(d1.keys()) == [0, 1, 2]
    assert sorted(d2.keys()) == [1, 2, 3]
    assert sorted(d3.keys()) == [0, 1, 3]


def test_3d(path):
    file_npy = path.parent.joinpath('titi.npy')
    file_cnpy = path.parent.joinpath('titi.cnpy')
    array = np.random.randint(-5000, high=5000, size=(100, 120, 130), dtype=np.int16)
    np.save(file_npy, array)
    # two way trip - makes sure that
    # 1) the sample_rate fed as an int64 doesn't error
    # 2) the initial shape of the array is saved in the meta-data
    mtscomp_mod.compress(file_npy,
                         out=file_cnpy,
                         outmeta=file_cnpy.with_suffix('.ch'),
                         sample_rate=np.prod(array.shape[1:]),  # here needs to cast as float
                         dtype=array.dtype,
                         do_time_diff=False)
    d = mtscomp_mod.decompress(file_cnpy, cmeta=file_cnpy.with_suffix('.ch'))
    assert np.all(np.isclose(d[:, :].reshape(d.cmeta.shape), array))


def test_chop(path):
    arr = np.array(np.random.randint(low=0, high=255, size=(1000, 100)), dtype=np.int16)
    _write_arr(path, arr)
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'
    compress(
        path, out, outmeta, sample_rate=100, n_channels=arr.shape[1], dtype=arr.dtype,
    )

    # Chop with method #1
    r = Reader()
    r.open(out, outmeta)
    out_chopped = out.with_name('data.chopped.cbin')
    assert r.n_chunks == 10
    r.chop(5, out_chopped)
    r.close()
    with open(str(out_chopped), 'rb') as f:
        sha1_chopped = sha1(f.read())

    # Check chopped file.
    r = Reader()
    r.open(out_chopped)
    assert r.n_chunks == 5
    arr_chopped = r[:]
    assert arr_chopped.dtype == arr.dtype
    assert arr_chopped.shape == (500, 100)
    assert np.all(arr_chopped == arr[:500])
    r.close()

    # Chop with method #2
    out_chopped2 = path.parent / 'data.chopped2.cbin'
    outmeta_chopped2 = path.parent / 'data.chopped2.ch'
    _write_arr(path, arr[:500])
    compress(
        path, out_chopped2, outmeta_chopped2, sample_rate=100,
        n_channels=arr.shape[1], dtype=arr.dtype,
    )

    # Check that the chopped file is identical with both methods.
    with open(str(out_chopped2), 'rb') as f:
        sha1_chopped2 = sha1(f.read())
    assert sha1_chopped == sha1_chopped2


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


@mark.parametrize('do_time_diff', [True, False])
@mark.parametrize('do_spatial_diff', [True, False])
@mark.parametrize('comp_level', [1, 6, 9])
def test_comp_levels_do_diff(path, arr, comp_level, do_time_diff, do_spatial_diff):
    _round_trip(
        path, arr, compression_level=comp_level,
        do_time_diff=do_time_diff, do_spatial_diff=do_spatial_diff)


@mark.parametrize('n_threads', [1, 2, 4, None])
def test_n_threads(path, arr, n_threads):
    _round_trip(path, arr, n_threads=n_threads)


#------------------------------------------------------------------------------
# CLI tests
#------------------------------------------------------------------------------

def test_cliargs_0(tmp_path_):
    """Test default parameters."""
    parser = mtscomp_parser()

    args = ['somefile']
    pargs, config = _args_to_config(parser, args)
    assert config.algorithm == 'zlib'
    assert config.check_after_compress
    assert config.check_after_decompress
    assert config.do_time_diff
    assert not config.do_spatial_diff

    pargs, config = _args_to_config(parser, args + ['-p 3'])
    assert config.n_threads == 3
    assert config.check_after_compress
    assert config.check_after_decompress

    pargs, config = _args_to_config(parser, args + ['-c 2', '-s 10000', '-n 123', '-d uint8'])
    assert config.chunk_duration == 2
    assert config.sample_rate == 10000
    assert config.n_channels == 123
    assert config.dtype == 'uint8'
    assert config.check_after_compress
    assert config.check_after_decompress
    assert not pargs.debug

    pargs, config = _args_to_config(parser, args + ['-c 2', '-nc', '--debug'])
    assert not config.check_after_compress
    assert config.check_after_decompress
    assert pargs.debug


def test_cliargs_1(tmp_path_):
    """Test default parameters."""
    parser = mtsdecomp_parser()

    args = ['somefile']
    pargs, config = _args_to_config(parser, args, compress=False)
    assert config.check_after_compress
    assert config.check_after_decompress

    pargs, config = _args_to_config(parser, args + ['-nc'], compress=False)
    assert config.check_after_compress
    assert not config.check_after_decompress


def test_cli_1(path, arr):
    _write_arr(path, arr)
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'
    path2 = path.parent / 'data2.bin'

    with raises(ValueError):
        # Wrong number of channels: should raise an error because the file size is not a
        # multiple of that wrong number of channels.
        mtscomp([
            str(path), '-d', str(arr.dtype), '-s', str(sample_rate), '-n', str(arr.shape[1] + 1)])
    # Compress.
    mtscomp([str(path), '-d', str(arr.dtype), '-s', str(sample_rate), '-n', str(arr.shape[1])])

    # Capture the compressed dataset description.
    f = io.StringIO()
    with redirect_stdout(f):
        mtsdesc([str(out), str(outmeta)])
    desc = f.getvalue()
    dt = np.dtype(re.search(r'dtype[ ]+(\S+)', desc).group(1))
    nc = int(re.search(r'n_channels[ ]+([0-9]+)', desc).group(1))

    # Decompress.
    mtsdecomp([str(out), str(outmeta), '-o', str(path2)])

    # Extract n_channels and dtype from the description.
    decompressed = load_raw_data(path=path2, n_channels=nc, dtype=dt)

    # Check that the decompressed and original arrays match.
    np.allclose(decompressed, arr)


def test_cli_2(path, arr):
    _write_arr(path, arr)
    parser = mtscomp_parser()
    args = [
        str(path), '-d', str(arr.dtype), '-s', str(sample_rate), '-n', str(arr.shape[1]),
        '-nc']

    # Error raised if params are not given.
    with raises(ValueError):
        mtscomp(args[:1] + ['--debug'])
    mtscomp(args)
    with raises(ValueError):
        mtscomp(args[:1] + args[3:])

    # Now, we use --set-default
    mtscomp(args + ['--set-default'])
    # This call should not fail this time.
    mtscomp(args[:1])

    # Check the saved default config.
    pargs, config = _args_to_config(parser, args[:1])
    assert config.dtype == str(arr.dtype)
    assert config.check_after_compress is False
    assert config.n_channels == 19
    assert config.sample_rate == 1234

    mtsdecomp([str(path.with_suffix('.cbin')), '-f', '--debug'])

    # Test override of a parameter despite set_default.
    pargs, config = _args_to_config(parser, args[:1] + ['-s 100'])
    assert config.sample_rate == 100


def test_cli_3(path, arr):
    _write_arr(path, arr)
    parser = mtscomp_parser()
    args = [str(path), '-d', str(arr.dtype), '-s', str(sample_rate)]

    # Error raised if params are not given.
    with raises(ValueError):
        mtscomp(args)
    with raises(ValueError):
        mtscomp(args[:1] + ['-n', '19'])

    # Now, we use --set-default
    with raises(ValueError):
        mtscomp(args + ['--set-default'])
    # This should still fail.
    with raises(ValueError):
        mtscomp(args[:1])
    # Should not fail
    mtscomp(args[:1] + ['-n', '19'])

    # Check the saved default config.
    pargs, config = _args_to_config(parser, args[:1])
    assert config.dtype == str(arr.dtype)
    assert config.check_after_compress is True
    assert config.get('n_channels', None) is None
    assert config.sample_rate == 1234


def test_cli_4(tmp_path_, arr):

    arr = np.stack([arr, arr], axis=2)

    root = Path(tmp_path_)
    path = root / 'data.npy'
    pathu = root / 'data.bin'

    np.save(path, arr)
    args = [str(path), '-s', str(sample_rate)]
    mtscomp(args)
    assert 'data.cnpy' in (p.name for p in path.parent.iterdir())

    assert not pathu.exists()
    pathc = root / 'data.cnpy'
    args = [str(pathc)]
    mtsdecomp(args)
    assert pathu.exists()

    arru = np.fromfile(str(pathu), dtype=arr.dtype).reshape(arr.shape)
    assert np.allclose(arr, arru)


def test_cli_chop(path, arr):
    _write_arr(path, arr)
    out = path.parent / 'data.cbin'
    out_chopped = path.parent / 'data.chopped.cbin'
    out_chopped_decomp = path.parent / 'data.chopped.bin'

    mtscomp([str(path), '-d', str(arr.dtype), '-s', str(sample_rate), '-n', str(arr.shape[1])])

    # Chop 3 chunks of 1 second each.
    mtschop([str(out), '-n', '3', '-o', str(out_chopped)])

    # Decompress the chopped compressed file.
    mtsdecomp([str(out_chopped), '-o', str(out_chopped_decomp)])

    # Make sure it corresponds to the first 3 seconds of the original array.
    arru = np.fromfile(str(out_chopped_decomp), dtype=arr.dtype)
    arru = arru.reshape((-1, arr.shape[1]))
    assert np.allclose(arr[:int(round(3 * sample_rate))], arru)
