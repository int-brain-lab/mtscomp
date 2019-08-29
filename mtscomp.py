# -*- coding: utf-8 -*-

"""mtscomp: multichannel time series lossless compression in Python."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import argparse
import bisect
from functools import lru_cache
import json
import logging
import os.path as op
from pathlib import Path
import sys
import zlib

import numpy as np

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Global variables
#------------------------------------------------------------------------------

__version__ = '0.1.0a1'
FORMAT_VERSION = '0.0'  # development

__all__ = ('load_raw_data', 'Writer', 'Reader', 'compress', 'decompress')


DEFAULT_CHUNK_DURATION = 1.
DEFAULT_COMPRESSION_ALGORITHM = 'zlib'
DEFAULT_CACHE_SIZE = 10  # number of chunks to keep in memory while reading the data
CHECK_AFTER_COMPRESS = True  # check the integrity of the compressed file
CRITICAL_ERROR_URL = \
    "https://github.com/int-brain-lab/mtscomp/issues/new?title=Critical+error"

CHECK_AFTER_DECOMPRESS = True  # check the integrity of the decompressed file saved to disk


#------------------------------------------------------------------------------
# Misc utils
#------------------------------------------------------------------------------

# Set a null handler on the root logger
logger = logging.getLogger('mtscomp')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

_logger_fmt = '%(asctime)s.%(msecs)03d [%(levelname)s] %(caller)s %(message)s'
_logger_date_fmt = '%H:%M:%S'


class _Formatter(logging.Formatter):
    def format(self, record):
        # Only keep the first character in the level name.
        record.levelname = record.levelname[0]
        filename = op.splitext(op.basename(record.pathname))[0]
        record.caller = '{:s}:{:d}'.format(filename, record.lineno).ljust(20)
        message = super(_Formatter, self).format(record)
        color_code = {'D': '90', 'I': '0', 'W': '33', 'E': '31'}.get(record.levelname, '7')
        message = '\33[%sm%s\33[0m' % (color_code, message)
        return message


def add_default_handler(level='INFO', logger=logger):
    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = _Formatter(fmt=_logger_fmt, datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


#------------------------------------------------------------------------------
# I/O utils
#------------------------------------------------------------------------------

def load_raw_data(path=None, n_channels=None, dtype=None, offset=None, mmap=True):
    """Load raw data at a given path."""
    path = Path(path)
    assert path.exists(), "File %s does not exist." % path
    assert dtype, "The data type must be provided."
    n_channels = n_channels or 1
    # Compute the array shape.
    item_size = np.dtype(dtype).itemsize
    offset = offset or 0
    n_samples = (op.getsize(str(path)) - offset) // (item_size * n_channels)
    size = n_samples * n_channels
    if size == 0:
        return np.zeros((0, n_channels), dtype=dtype)
    shape = (n_samples, n_channels)
    # Memmap the file into a NumPy-like array.
    if mmap:
        return np.memmap(str(path), dtype=dtype, shape=shape, offset=offset)
    else:
        if offset > 0:  # pragma: no cover
            raise NotImplementedError()  # TODO
        return np.fromfile(str(path), dtype).reshape(shape)


#------------------------------------------------------------------------------
# Low-level API
#------------------------------------------------------------------------------

class Writer:
    """Handle compression of a raw data file.

    Constructor
    -----------

    chunk_duration : float
        Duration of the chunks, in seconds.
    compression_algorithm : str
        Name of the compression algorithm. Only `zlib` is supported at the moment.
    compression_level : int
        Compression level of the chosen algorithm.
    do_diff : bool
        Whether to compute the time-wise diff of the array before compressing.
    before_check : function
        A callback method that could be called just before the integrity check.

    """
    def __init__(
            self, chunk_duration=DEFAULT_CHUNK_DURATION, compression_algorithm=None,
            compression_level=-1, do_diff=True, before_check=None):
        self.chunk_duration = chunk_duration
        self.compression_algorithm = compression_algorithm or DEFAULT_COMPRESSION_ALGORITHM
        assert self.compression_algorithm == 'zlib', "Only zlib is currently supported."
        self.compression_level = compression_level
        self.do_diff = do_diff
        self.before_check = before_check or (lambda x: None)

    def open(
            self, data_path, sample_rate=None, n_channels=None, dtype=None,
            offset=None, mmap=True):
        """Open a raw data (memmapped) from disk in order to compress it.

        Parameters
        ----------

        data_path : str or Path
            Path to the raw binary array.
        sample_rate : float
            Sample rate of the data.
        n_channels : int
            Number of columns (channels) in the data array.
            The shape of the data is `(n_samples, n_channels)`.
        dtype : dtype
            NumPy data type of the data array.
        offset : int
            Offset, in bytes, of the data within the binary file.
        mmap : bool
            Whether the data should be memmapped.

        """
        self.sample_rate = sample_rate
        assert sample_rate > 0
        assert n_channels > 0
        self.dtype = dtype
        self.data_path = Path(data_path)
        self.data = load_raw_data(data_path, n_channels=n_channels, dtype=dtype)
        self.file_size = self.data.size * self.data.itemsize
        assert self.data.ndim == 2
        self.n_samples, self.n_channels = self.data.shape
        assert self.n_samples > 0
        assert self.n_channels > 0
        assert n_channels == self.n_channels
        logger.debug("Open %s with size %s.", data_path, self.data.shape)
        self._compute_chunk_bounds()

    def _compute_chunk_bounds(self):
        """Compute the chunk bounds, in number of time samples."""
        chunk_size = int(np.round(self.chunk_duration * self.sample_rate))
        chunk_bounds = list(range(0, self.n_samples, chunk_size))
        if chunk_bounds[-1] < self.n_samples:
            chunk_bounds.append(self.n_samples)
        # One element more than the number of chunks, the chunk is in
        # chunk_bounds[i]:chunk_bounds[i+1] (first element included, last element excluded).
        self.chunk_bounds = chunk_bounds
        self.n_chunks = len(self.chunk_bounds) - 1
        assert self.chunk_bounds[0] == 0
        assert self.chunk_bounds[-1] == self.n_samples
        logger.debug("Chunk bounds: %s", self.chunk_bounds)

    def get_cmeta(self):
        """Return the metadata of the compressed file."""
        return {
            'version': FORMAT_VERSION,
            'compression_algorithm': self.compression_algorithm,
            'compression_level': self.compression_level,
            'do_diff': self.do_diff,
            'dtype': str(np.dtype(self.dtype)),
            'n_channels': self.n_channels,
            'sample_rate': self.sample_rate,
            'chunk_bounds': self.chunk_bounds,
            'chunk_offsets': self.chunk_offsets,
        }

    def get_chunk(self, chunk_idx):
        """Return a given chunk as a NumPy array with shape `(n_samples_chk, n_channels)`.

        Parameters
        ----------

        chunk_idx : int
            Index of the chunk, from 0 to `n_chunks - 1`.

        """
        assert 0 <= chunk_idx <= self.n_chunks - 1
        i0 = self.chunk_bounds[chunk_idx]
        i1 = self.chunk_bounds[chunk_idx + 1]
        return self.data[i0:i1, :]

    def write_chunk(self, chunk_idx, fb):
        """Write a given chunk into the output file.

        Parameters
        ----------

        chunk_idx : int
            Index of the chunk, from 0 to `n_chunks - 1`.
        fb : file_handle
            File handle of the compressed binary file to be written.

        Returns
        -------

        length : int
            The number of bytes written in the compressed binary file.

        """
        # Retrieve the chunk data as a 2D NumPy array.
        chunk = self.get_chunk(chunk_idx)
        assert chunk.ndim == 2
        assert chunk.shape[1] == self.n_channels
        # Compute the diff along the time axis.
        if self.do_diff:
            chunkd = np.diff(chunk, axis=0)
            chunkd = np.concatenate((chunk[0, :][np.newaxis, :], chunkd), axis=0)
        else:  # pragma: no cover
            chunkd = chunk
        # The first row is the same (we need to keep the initial values in order to reconstruct
        # the original array from the diff)0
        assert chunkd.shape == chunk.shape
        assert np.array_equal(chunkd[0, :], chunk[0, :])
        # Compress the diff.
        chunkdc = zlib.compress(chunkd.tobytes())
        length = fb.write(chunkdc)
        ratio = 100 - 100 * length / (chunk.size * chunk.itemsize)
        logger.debug("Chunk %d/%d: -%.3f%%.", chunk_idx + 1, self.n_chunks, ratio)
        # Return the number of bytes written.
        return length

    def write(self, out, outmeta):
        """Write the compressed data in a compressed binary file, and a compression header file
        in JSON.

        Parameters
        ----------

        out : str or Path
            Path to the compressed data binary file (typically ̀.cbin` file extension).
        outmeta : str or Path
            Path to the compression header JSON file (typicall `.ch` file extension).

        Returns
        -------

        ratio : float
            The ratio of the size of the compressed binary file versus the size of the
            original binary file.

        """
        # Default file extension for output files.
        if not out:
            out = self.data_path.with_suffix('.c' + self.data_path.suffix[1:])
        if not outmeta:
            outmeta = self.data_path.with_suffix('.ch')
        # Ensure the parent directory exists.
        Path(out).parent.mkdir(exist_ok=True, parents=True)
        # Write all chunks.
        offset = 0
        self.chunk_offsets = [0]
        with open(out, 'wb') as fb:
            for chunk_idx in range(self.n_chunks):
                length = self.write_chunk(chunk_idx, fb)
                offset += length
                self.chunk_offsets.append(offset)
            # Final size of the file.
            csize = fb.tell()
        assert self.chunk_offsets[-1] == csize
        ratio = csize / self.file_size
        logger.info("Wrote %s (-%.3f%%).", out, 100 - 100 * ratio)
        # Write the metadata file.
        with open(outmeta, 'w') as f:
            json.dump(self.get_cmeta(), f, indent=2, sort_keys=True)
        # Check that the written file matches the original file (once decompressed).
        if CHECK_AFTER_COMPRESS:
            # Callback function before checking.
            self.before_check(self)
            try:
                check(self.data, out, outmeta)
            except AssertionError:
                raise RuntimeError(
                    "CRITICAL ERROR: automatic check failed when compressing the data. "
                    "Report immediately to " + CRITICAL_ERROR_URL)
            logger.debug("Automatic integrity check after compression PASSED.")
        return ratio

    def close(self):
        """Close all file handles."""
        self.data._mmap.close()


class Reader:
    """Handle decompression of a compressed data file.

    Constructor
    -----------

    cache_size : int
        Maximum number of chunks to keep in memory while reading. Every chunk kept in cache
        may take a few dozens of MB in RAM.

    """
    def __init__(self, cache_size=DEFAULT_CACHE_SIZE):
        self.cache_size = cache_size or 0

    def open(self, cdata, cmeta):
        """Open a compressed data file.

        Parameters
        ----------

        cdata : str or Path
            Path to the compressed data file.
        cmeta : str or Path or dict
            Path to the compression header JSON file, or its contents as a Python dictionary.

        """
        # Read metadata file.
        if not isinstance(cmeta, dict):
            with open(cmeta, 'r') as f:
                cmeta = json.load(f)
        assert isinstance(cmeta, dict)
        self.cmeta = Bunch(cmeta)
        # Read some values from the metadata file.
        self.n_channels = self.cmeta.n_channels
        self.dtype = np.dtype(self.cmeta.dtype)
        self.chunk_offsets = self.cmeta.chunk_offsets
        self.chunk_bounds = self.cmeta.chunk_bounds
        self.n_samples = self.chunk_bounds[-1]
        self.n_chunks = len(self.chunk_bounds) - 1
        self.shape = (self.n_samples, self.n_channels)
        self.ndim = 2

        # Open data.
        if isinstance(cdata, (str, Path)):
            cdata = open(cdata, 'rb')
        self.cdata = cdata

        if self.cache_size > 0:
            self.read_chunk = lru_cache(maxsize=self.cache_size)(self.read_chunk)

    def iter_chunks(self, first_chunk=0, last_chunk=None):
        """Iterate over the compressed chunks.

        Yield tuples `(chunk_idx, chunk_start, chunk_length)`.

        """
        last_chunk = last_chunk if last_chunk is not None else self.n_chunks - 1
        for idx, (i0, i1) in enumerate(
                zip(self.chunk_offsets[first_chunk:last_chunk + 1],
                    self.chunk_offsets[first_chunk + 1:last_chunk + 2])):
            yield first_chunk + idx, i0, i1 - i0

    def read_chunk(self, chunk_idx, chunk_start, chunk_length):
        """Read a compressed chunk and return a NumPy array."""
        # Load the compressed chunk from the file.
        self.cdata.seek(chunk_start)
        cbuffer = self.cdata.read(chunk_length)
        assert len(cbuffer) == chunk_length
        # Decompress the chunk.
        buffer = zlib.decompress(cbuffer)
        chunk = np.frombuffer(buffer, self.dtype)
        assert chunk.dtype == self.dtype
        # Reshape the chunk.
        i0, i1 = self.chunk_bounds[chunk_idx:chunk_idx + 2]
        assert i0 <= i1
        n_samples_chunk = i1 - i0
        assert chunk.size == n_samples_chunk * self.n_channels
        chunk = chunk.reshape((n_samples_chunk, self.n_channels))
        # Perform a cumsum.
        if self.cmeta.do_diff:
            chunki = np.empty_like(chunk)
            np.cumsum(chunk, axis=0, out=chunki)
        else:
            chunki = chunk
        assert chunki.shape == (n_samples_chunk, self.n_channels)
        return chunki

    def _validate_index(self, i, value_for_none=0):
        if i is None:
            i = value_for_none
        elif i < 0:
            i += self.n_samples
        i = np.clip(i, 0, self.n_samples)
        assert 0 <= i <= self.n_samples
        return i

    def _chunks_for_interval(self, i0, i1):
        """Find the first and last chunks to be loaded in order to get the data between
        time samples `i0` and `i1`."""
        assert i0 <= i1

        first_chunk = max(0, bisect.bisect_left(self.chunk_bounds, i0) - 1)
        assert first_chunk >= 0
        assert self.chunk_bounds[first_chunk] <= i0

        last_chunk = min(
            bisect.bisect_left(self.chunk_bounds, i1, lo=first_chunk),
            self.n_chunks - 1)
        assert i1 <= self.chunk_bounds[last_chunk + 1]

        assert 0 <= first_chunk <= last_chunk <= self.n_chunks - 1
        return first_chunk, last_chunk

    def tofile(self, out):
        """Write the decompressed array to disk."""
        out = Path(out)
        if out.exists():  # pragma: no cover
            raise ValueError("The output file %s already exists." % out)
        # Read all chunks and save them to disk.
        with open(out, 'wb') as fb:
            for chunk_idx, chunk_start, chunk_length in self.iter_chunks():
                self.read_chunk(chunk_idx, chunk_start, chunk_length).tofile(fb)
        if CHECK_AFTER_DECOMPRESS:
            decompressed = load_raw_data(out, n_channels=self.n_channels, dtype=self.dtype)
            check(decompressed, self.cdata, self.cmeta)
            logger.debug("Automatic integrity check after decompression PASSED.")

    def close(self):
        """Close all file handles."""
        self.cdata.close()

    def __getitem__(self, item):
        """Implement NumPy array slicing, return a regular in-memory NumPy array."""
        fallback = np.zeros((0, self.n_channels), dtype=self.dtype)
        if isinstance(item, slice):
            # Slice indexing.
            i0 = self._validate_index(item.start, 0)
            i1 = self._validate_index(item.stop, self.n_samples)
            if i1 <= i0:
                return fallback
            assert i0 < i1
            first_chunk, last_chunk = self._chunks_for_interval(i0, i1)
            chunks = []
            for chunk_idx, chunk_start, chunk_length in self.iter_chunks(first_chunk, last_chunk):
                chunk = self.read_chunk(chunk_idx, chunk_start, chunk_length)
                chunks.append(chunk)
            if not chunks:  # pragma: no cover
                return fallback
            # Concatenate all chunks.
            ns = sum(chunk.shape[0] for chunk in chunks)
            arr = np.empty((ns, self.n_channels), dtype=self.dtype)
            arr = np.concatenate(chunks, out=arr)
            assert arr.ndim == 2
            assert arr.shape[1] == self.n_channels
            assert arr.shape[0] == (
                self.chunk_bounds[last_chunk + 1] - self.chunk_bounds[first_chunk])
            # Subselect in the chunk.
            a = i0 - self.chunk_bounds[first_chunk]
            b = i1 - self.chunk_bounds[first_chunk]
            assert 0 <= a <= b <= arr.shape[0]
            out = arr[a:b:item.step, :]
            # Ensure the shape of the output is the expected shape from the slice length.
            assert out.shape[0] == len(range(i0, i1, item.step or 1))
            return out
        elif isinstance(item, tuple):
            # Multidimensional indexing.
            if len(item) == 1:
                return self[item[0]]
            elif len(item) == 2:
                return self[item[0]][:, item[1]]
        elif isinstance(item, int):
            if item < 0:
                # Deal with negative indices.
                k = -int(np.floor(item / self.n_samples))
                item = item + self.n_samples * k
                assert 0 <= item < self.n_samples
            if not 0 <= item < self.n_samples:  # pragma: no cover
                raise IndexError(
                    "index %d is out of bounds for axis 0 with size %d" % (item, self.n_samples))
            out = self[item:item + 1]
            return out[0]
        elif isinstance(item, (list, np.ndarray)):  # pragma: no cover
            raise NotImplementedError("Indexing with multiple values is currently unsupported.")
        return fallback  # pragma: no cover

    def __del__(self):
        self.close()


#------------------------------------------------------------------------------
# High-level API
#------------------------------------------------------------------------------

def check(data, out, outmeta):
    """Check that the compressed data matches the original array byte per byte."""
    unc = decompress(out, outmeta)
    # Read all chunks.
    for chunk_idx, chunk_start, chunk_length in unc.iter_chunks():
        chunk = unc.read_chunk(chunk_idx, chunk_start, chunk_length)
        # Find the corresponding chunk from the original data array.
        i0, i1 = unc.chunk_bounds[chunk_idx], unc.chunk_bounds[chunk_idx + 1]
        expected = data[i0:i1]
        # Check the dtype and shape match.
        assert chunk.dtype == expected.dtype
        assert chunk.shape == expected.shape
        if np.issubdtype(chunk.dtype, np.integer):
            # For integer dtypes, check that the array are exactly equal.
            assert np.array_equal(chunk, expected)
        else:
            # For floating point dtypes, check that the array are almost equal
            # (diff followed by cumsum does not yield exactly the same floating point numbers).
            assert np.allclose(chunk, expected, atol=1e-16)


def compress(
        path, out=None, outmeta=None,
        sample_rate=None, n_channels=None, dtype=None,
        chunk_duration=DEFAULT_CHUNK_DURATION, compression_algorithm=None,
        compression_level=-1, do_diff=True):
    """Compress a NumPy-like array (may be memmapped) into a compressed format
    (two files, out and outmeta).

    Parameters
    ----------

    path : str or Path
        Path to a raw data binary file.
    out : str or Path
        Path the to compressed data file.
    outmeta : str or Path
        Path to the compression header JSON file.
    sample_rate : float
        Sampling rate, in Hz.
    dtype : dtype
        The data type of the array in the raw data file.
    n_channels : int
        Number of channels in the file.
    chunk_duration : float
        Duration of the chunks, in seconds.
    compression_algorithm : str
        Name of the compression algorithm. Only `zlib` is supported at the moment.
    compression_level : int
        Compression level of the chosen algorithm.
    do_diff : bool
        Whether to compute the time-wise diff of the array before compressing.

    Returns
    -------

    length : int
        Number of bytes written.

    Metadata dictionary
    -------------------

    Saved in the cmeta file as JSON.

    version : str
        Version number of the compression format.
    compression_algorithm : str
        Name of the compression algorithm. Only `zlib` is supported at the moment.
    compression_level : str
        Compression level to be passed to the compression function.
    n_channels : int
        Number of channels.
    sample_rate : float
        Sampling rate, in Hz.
    chunk_bounds : list of ints
        Offsets of the chunks in time samples.
    chunk_offsets : list of ints
        Offsets of the chunks within the compressed raw buffer.

    """

    w = Writer(
        chunk_duration=chunk_duration, compression_algorithm=compression_algorithm,
        compression_level=compression_level, do_diff=do_diff)
    w.open(path, sample_rate=sample_rate, n_channels=n_channels, dtype=dtype)
    length = w.write(out, outmeta)
    w.close()
    return length


def decompress(cdata, cmeta, out=None):
    """Read an array from a compressed dataset (two files, cdata and cmeta), and
    return a NumPy-like array (memmapping the compressed data file, and decompressing on the fly).

    Note: the reader should be closed after use.

    Parameters
    ----------

    cdata : str or Path
        Path to the compressed data file.
    cmeta : str or Path
        Path to the compression header JSON file.
    out : str or Path
        Path to the decompressed file to be written.

    Returns
    -------

    reader : Reader instance
        This object implements the NumPy slicing syntax to access
        parts of the actual data as NumPy arrays.

    """

    r = Reader()
    r.open(cdata, cmeta)
    if out:
        r.tofile(out)
    return r


#------------------------------------------------------------------------------
# Command-line API: mtscomp
#------------------------------------------------------------------------------

def mtscomp_parse_args(args):
    """Command-line interface to compress a file."""
    parser = argparse.ArgumentParser(description='Compress a raw binary file.')

    parser.add_argument(
        'path', type=str, help='input path of a raw binary file')

    parser.add_argument(
        'out', type=str, nargs='?',
        help='output path of the compressed binary file')

    parser.add_argument(
        'outmeta', type=str, nargs='?',
        help='output path of the compression metadata file')

    parser.add_argument('-d', type=str, help='data type')
    parser.add_argument('-s', type=float, help='sample rate')
    parser.add_argument('-n', type=int, help='number of channels')

    return parser.parse_args(args)


def mtscomp(args=None):
    """Compress a file."""
    parser = mtscomp_parse_args(args or sys.argv[1:])
    compress(
        parser.path, parser.out, parser.outmeta,
        sample_rate=parser.s, n_channels=parser.n, dtype=np.dtype(parser.d))


#------------------------------------------------------------------------------
# Command-line API: mtsdecomp
#------------------------------------------------------------------------------

def mtsdecomp_parse_args(args):
    """Command-line interface to decompress a file."""
    parser = argparse.ArgumentParser(description='Decompress a raw binary file.')

    parser.add_argument(
        'cdata', type=str,
        help='path to the compressed binary file')

    parser.add_argument(
        'cmeta', type=str,
        help='path to the compression metadata file')

    parser.add_argument(
        'out', type=str, nargs='?',
        help='path to the decompressed file')

    parser.add_argument('-d', type=str, help='data type')
    parser.add_argument('-s', type=float, help='sample rate')
    parser.add_argument('-n', type=int, help='number of channels')

    return parser.parse_args(args)


def mtsdecomp(args=None):
    """Decompress a file."""
    parser = mtsdecomp_parse_args(args or sys.argv[1:])
    decompress(parser.cdata, parser.cmeta, parser.out)


#------------------------------------------------------------------------------
# Command-line API: mtsdesc
#------------------------------------------------------------------------------

def mtsdesc(args=None):
    """Describe a compressed file."""
    parser = mtsdecomp_parse_args(args or sys.argv[1:])
    r = Reader()
    r.open(parser.cdata, parser.cmeta)
    sr = float(r.cmeta.sample_rate)
    info = dict(
        dtype=r.dtype,
        sample_rate=sr,
        n_channels=r.n_channels,
        duration='%.1fs' % (r.n_samples / sr),
        n_samples=r.n_samples,
        chunk_duration='%.1fs' % (np.diff(r.chunk_bounds).mean() / sr),
        n_chunks=r.n_chunks,
    )
    for k, v in info.items():
        print('{:<15}'.format(k), str(v))
