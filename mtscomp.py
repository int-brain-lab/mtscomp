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
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import os.path as op
from pathlib import Path
import sys
import zlib

from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Global variables
#------------------------------------------------------------------------------

__version__ = '0.1.0a1'
FORMAT_VERSION = '0.0'  # development

__all__ = ('load_raw_data', 'Writer', 'Reader', 'compress', 'decompress')


DEFAULT_CHUNK_DURATION = 1.  # in seconds
DEFAULT_ALGORITHM = 'zlib'  # only algorithm supported currently
DEFAULT_COMPRESSION_LEVEL = -1
DEFAULT_DO_TIME_DIFF = True
DEFAULT_DO_SPATIAL_DIFF = False  # benchmarks seem to show no compression performance benefits
DEFAULT_CACHE_SIZE = 10  # number of chunks to keep in memory while reading the data
DEFAULT_N_THREADS = mp.cpu_count()

# Automatic checks when compressing/decompressing.
CHECK_AFTER_COMPRESS = True  # check the integrity of the compressed file
CHECK_AFTER_DECOMPRESS = True  # check the integrity of the decompressed file saved to disk
CHECK_ATOL = 1e-16  # tolerance for floating point array comparison check
CRITICAL_ERROR_URL = \
    "https://github.com/int-brain-lab/mtscomp/issues/new?title=Critical+error"


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


def diff_along_axis(chunk, axis=None):
    """Perform a diff along a given axis in a 2D array.
    Keep the first line/column identical.
    """
    if axis is None:
        return chunk
    assert 0 <= axis < chunk.ndim
    chunkd = np.diff(chunk, axis=axis)
    # The first row is the same (we need to keep the initial values in order to reconstruct
    # the original array from the diff).
    if axis == 0:
        logger.log(5, "Performing time diff.")
        chunkd = np.concatenate((chunk[0, :][np.newaxis, :], chunkd), axis=axis)
    elif axis == 1:
        logger.debug("Performing spatial diff.")
        chunkd = np.concatenate((chunk[:, 0][:, np.newaxis], chunkd), axis=axis)
    return chunkd


def cumsum_along_axis(chunk, axis=None):
    """Perform a cumsum (inverse of diff) along a given axis in a 2D array."""
    if axis is None:
        return chunk
    assert 0 <= axis < chunk.ndim
    chunki = np.empty_like(chunk)
    np.cumsum(chunk, axis=axis, out=chunki)
    return chunki


#------------------------------------------------------------------------------
# Low-level API
#------------------------------------------------------------------------------

class Writer:
    """Handle compression of a raw data file.

    Constructor
    -----------

    chunk_duration : float
        Duration of the chunks, in seconds.
    algorithm : str
        Name of the compression algorithm. Only `zlib` is supported at the moment.
    comp_level : int
        Compression level of the chosen algorithm.
    do_time_diff : bool
        Whether to compute the time-wise diff of the array before compressing.
    do_spatial_diff : bool
        Whether to compute the spatial diff of the array before compressing.
    n_threads : int
        Number of CPUs to use for compression. By default, use all of them.
    before_check : function
        A callback method that could be called just before the integrity check.
    check_after_compress : bool
        Whether to perform the automatic check after compression.

    """
    def __init__(
            self, chunk_duration=None, before_check=None,
            algorithm=None, comp_level=None,
            do_time_diff=None, do_spatial_diff=None,
            n_threads=None, check_after_compress=None,
    ):
        self.chunk_duration = chunk_duration or DEFAULT_CHUNK_DURATION
        self.algorithm = algorithm or DEFAULT_ALGORITHM
        assert self.algorithm == 'zlib', "Only zlib is currently supported."
        self.comp_level = comp_level if comp_level is not None else DEFAULT_COMPRESSION_LEVEL
        self.do_time_diff = do_time_diff if do_time_diff is not None else DEFAULT_DO_TIME_DIFF
        self.do_spatial_diff = (
            do_spatial_diff if do_spatial_diff is not None else DEFAULT_DO_SPATIAL_DIFF)
        self.n_threads = n_threads or DEFAULT_N_THREADS  # 1 means no multithreading
        self.before_check = before_check or (lambda x: None)
        self.check_after_compress = (
            check_after_compress if check_after_compress is not None else CHECK_AFTER_COMPRESS)

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
        duration = self.data.shape[0] / self.sample_rate
        logger.info("Open %s, duration %.1fs, %d channels.", data_path, duration, self.n_channels)
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
        logger.log(5, "Chunk bounds: %s", self.chunk_bounds)
        # Batches.
        self.batch_size = self.n_threads  # in each batch, there is 1 chunk per thread.
        self.n_batches = int(np.ceil(self.n_chunks / self.batch_size))

    def get_cmeta(self):
        """Return the metadata of the compressed file."""
        return {
            'version': FORMAT_VERSION,
            'algorithm': self.algorithm,
            'comp_level': self.comp_level,
            'do_time_diff': self.do_time_diff,
            'do_spatial_diff': self.do_spatial_diff,
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

    def _compress_chunk(self, chunk_idx):
        # Retrieve the chunk data as a 2D NumPy array.
        chunk = self.get_chunk(chunk_idx)
        assert chunk.ndim == 2
        assert chunk.shape[1] == self.n_channels
        # Compute the diff along the time and/or spatial axis.
        chunkd = diff_along_axis(chunk, axis=0 if self.do_time_diff else None)
        chunkd = diff_along_axis(chunkd, axis=1 if self.do_spatial_diff else None)
        assert chunkd.shape == chunk.shape
        assert chunkd.dtype == chunk.dtype
        # Check first line/column of the diffed chunk.
        assert chunkd[0, 0] == chunk[0, 0]
        if self.do_time_diff and not self.do_spatial_diff:
            assert np.array_equal(chunkd[0, :], chunk[0, :])
        elif not self.do_time_diff and self.do_spatial_diff:
            assert np.array_equal(chunkd[:, 0], chunk[:, 0])
        # Compress the diff.
        logger.log(5, "Compressing %d MB...", (chunkd.size * chunk.itemsize) / 1024. ** 2)
        chunkdc = zlib.compress(chunkd.tobytes())
        ratio = 100 - 100 * len(chunkdc) / (chunk.size * chunk.itemsize)
        logger.debug("Chunk %d/%d: -%.3f%%.", chunk_idx + 1, self.n_chunks, ratio)
        return chunk_idx, chunkdc

    def compress_batch(self, first_chunk, last_chunk):
        """Write a given chunk into the output file.

        Parameters
        ----------

        first_chunk : int
            Index of the first chunk in the batch (included).
        last_chunk : int
            Index of the last chunk in the batch (excluded).

        Returns
        -------

        chunks : dict
            A dictionary mapping chunk indices to compressed chunks.

        """
        assert 0 <= first_chunk < last_chunk <= self.n_chunks
        if self.n_threads == 1:
            chunks = [
                self._compress_chunk(chunk_idx) for chunk_idx in range(first_chunk, last_chunk)]
        elif self.n_threads >= 2:
            chunks = self.pool.map(self._compress_chunk, range(first_chunk, last_chunk))
        return dict(chunks)

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
        # Create the thread pool.
        self.pool = ThreadPool(self.batch_size)
        logger.info("Starting compression on %d thread(s).", self.n_threads)
        with open(out, 'wb') as fb:
            for batch in tqdm(range(self.n_batches), desc='Compressing'):
                first_chunk = self.batch_size * batch  # first included
                last_chunk = min(self.batch_size * (batch + 1), self.n_chunks)  # last excluded
                assert 0 <= first_chunk < last_chunk <= self.n_chunks
                logger.debug(
                    "Processing batch #%d/%d with chunks %s.",
                    batch + 1, self.n_batches, ', '.join(map(str, range(first_chunk, last_chunk))))
                # Compress all chunks in the batch.
                compressed_chunks = self.compress_batch(first_chunk, last_chunk)
                # Return a dictionary chunk_idx: compressed_buffer
                assert set(compressed_chunks.keys()) <= set(range(first_chunk, last_chunk))
                # Write the batch chunks to disk.
                # Warning: we need to process the chunks in order.
                for chunk_idx in sorted(compressed_chunks.keys()):
                    compressed_chunk = compressed_chunks[chunk_idx]
                    fb.write(compressed_chunk)
                    # Append the chunk offsets.
                    length = len(compressed_chunk)
                    offset += length
                    self.chunk_offsets.append(offset)
            # Final size of the file.
            csize = fb.tell()
        assert self.chunk_offsets[-1] == csize
        # Close the thread pool.
        self.pool.close()
        self.pool.join()
        # Compute the compression ratio.
        ratio = csize / self.file_size
        logger.info("Wrote %s (%.1f GB, -%.3f%%).", out, csize / 1024 ** 3, 100 - 100 * ratio)
        # Write the metadata file.
        with open(outmeta, 'w') as f:
            json.dump(self.get_cmeta(), f, indent=2, sort_keys=True)
        # Check that the written file matches the original file (once decompressed).
        if self.check_after_compress:
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
    check_after_decompress : bool
        Whether to perform the automatic check after decompression.

    """
    def __init__(self, cache_size=None, check_after_decompress=None):
        self.cache_size = cache_size if cache_size is not None else DEFAULT_CACHE_SIZE
        self.check_after_decompress = (
            check_after_decompress if check_after_decompress is not None
            else CHECK_AFTER_DECOMPRESS)

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
        chunki = cumsum_along_axis(chunk, axis=1 if self.cmeta.do_spatial_diff else None)
        chunki = cumsum_along_axis(chunki, axis=0 if self.cmeta.do_time_diff else None)
        assert chunki.dtype == chunk.dtype
        assert chunki.shape == chunk.shape == (n_samples_chunk, self.n_channels)
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
        # if out.exists():  # pragma: no cover
        #     raise ValueError("The output file %s already exists." % out)
        # Read all chunks and save them to disk.
        with open(out, 'wb') as fb:
            for chunk_idx, chunk_start, chunk_length in tqdm(
                    self.iter_chunks(), desc='Decompressing', total=self.n_chunks):
                self.read_chunk(chunk_idx, chunk_start, chunk_length).tofile(fb)
            dsize = fb.tell()
        logger.info("Wrote %s (%.1f GB).", out, dsize / 1024 ** 3)
        if self.check_after_decompress:
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
    for chunk_idx, chunk_start, chunk_length in tqdm(
            unc.iter_chunks(), total=unc.n_chunks, desc='Checking'):
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
            assert np.allclose(chunk, expected, atol=CHECK_ATOL)


def compress(
        path, out=None, outmeta=None,
        sample_rate=None, n_channels=None, dtype=None,
        chunk_duration=None, algorithm=None,
        comp_level=None, do_time_diff=None, do_spatial_diff=None,
        n_threads=None, check_after_compress=None):
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
    algorithm : str
        Name of the compression algorithm. Only `zlib` is supported at the moment.
    comp_level : int
        Compression level of the chosen algorithm.
    do_time_diff : bool
        Whether to compute the time-wise diff of the array before compressing.
    do_spatial_diff : bool
        Whether to compute the spatial diff of the array before compressing.
    n_threads : int
        Number of CPUs to use for compression. By default, use all of them.
    check_after_compress : bool
        Whether to perform the automatic check after compression.

    Returns
    -------

    length : int
        Number of bytes written.

    Metadata dictionary
    -------------------

    Saved in the cmeta file as JSON.

    version : str
        Version number of the compression format.
    algorithm : str
        Name of the compression algorithm. Only `zlib` is supported at the moment.
    comp_level : str
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
        chunk_duration=chunk_duration,
        algorithm=algorithm,
        comp_level=comp_level,
        do_time_diff=do_time_diff,
        do_spatial_diff=do_spatial_diff,
        n_threads=n_threads,
        check_after_compress=check_after_compress,
    )
    w.open(path, sample_rate=sample_rate, n_channels=n_channels, dtype=dtype)
    length = w.write(out, outmeta)
    w.close()
    return length


def decompress(cdata, cmeta, out=None, check_after_decompress=None):
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
    check_after_decompress : bool
        Whether to perform the automatic check after decompression.

    Returns
    -------

    reader : Reader instance
        This object implements the NumPy slicing syntax to access
        parts of the actual data as NumPy arrays.

    """

    r = Reader(check_after_decompress=check_after_decompress)
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
    parser.add_argument('-p', type=int, help='number of CPUs to use')
    parser.add_argument('-c', type=int, help='chunk duration')
    parser.add_argument('-nc', action='store_false', help='no check')
    parser.add_argument('-v', action='store_true', help='verbose')

    return parser.parse_args(args)


def mtscomp(args=None):
    """Compress a file."""
    parser = mtscomp_parse_args(args or sys.argv[1:])
    add_default_handler('DEBUG' if parser.v else 'INFO')
    compress(
        parser.path, parser.out, parser.outmeta,
        sample_rate=parser.s, n_channels=parser.n, dtype=np.dtype(parser.d),
        chunk_duration=parser.c, n_threads=parser.c, check_after_compress=parser.nc)


#------------------------------------------------------------------------------
# Command-line API: mtsdecomp
#------------------------------------------------------------------------------

def mtsdecomp_parse_args(args):
    """Command-line interface to decompress a file."""
    parser = argparse.ArgumentParser(description='Decompress a raw binary file.')

    parser.add_argument(
        'cdata', type=str,
        help='path to the input compressed binary file')

    parser.add_argument(
        'cmeta', type=str,
        help='path to the input compression metadata file')

    parser.add_argument(
        'out', type=str, nargs='?',
        help='path to the output decompressed file')

    parser.add_argument('-nc', action='store_false', help='no check')

    return parser.parse_args(args)


def mtsdecomp(args=None):
    """Decompress a file."""
    add_default_handler('INFO')
    parser = mtsdecomp_parse_args(args or sys.argv[1:])
    decompress(parser.cdata, parser.cmeta, parser.out, check_after_decompress=parser.nc)


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
