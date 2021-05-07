# -*- coding: utf-8 -*-

"""mtscomp: multichannel time series lossless compression in Python."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import argparse
import bisect
from functools import lru_cache
import hashlib
import json
import logging
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import os
import os.path as op
from pathlib import Path
import sys
from threading import Lock
import zlib

# import traceback

from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


lock = Lock()  # use for concurrent read on the same file with multithreaded decompression


#------------------------------------------------------------------------------
# Global variables
#------------------------------------------------------------------------------

__version__ = '1.0.2'
FORMAT_VERSION = '1.0'

__all__ = ('load_raw_data', 'Writer', 'Reader', 'compress', 'decompress')


DEFAULT_CONFIG = list(dict(
    algorithm='zlib',  # only algorithm supported currently
    cache_size=10,  # number of chunks to keep in memory while reading the data
    check_after_compress=True,  # check the integrity of the compressed file
    check_after_decompress=True,  # check the integrity of the decompressed file saved to disk
    chunk_duration=1.,  # in seconds
    chunk_order='F',  # leads to slightly better compression than C order
    comp_level=-1,  # zlib compression level
    do_spatial_diff=False,  # benchmarks seem to show no compression performance benefits
    do_time_diff=True,
    n_threads=mp.cpu_count(),
).items())  # convert to a list to ensure this dictionary is read-only

CHECK_ATOL = 1e-16  # tolerance for floating point array comparison check
CRITICAL_ERROR_URL = \
    "https://github.com/int-brain-lab/mtscomp/issues/new?title=Critical+error"


#------------------------------------------------------------------------------
# Misc utils
#------------------------------------------------------------------------------

# Set a null handler on the root logger
logger = logging.getLogger('mtscomp')
logger.setLevel(logging.INFO)
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


def _clip(x, a, b):
    return max(a, min(b, x))


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
    f_size = op.getsize(str(path))
    n_samples = (f_size - offset) // (item_size * n_channels)
    size = n_samples * n_channels
    if size * item_size != (f_size - offset):
        raise ValueError(
            ("The file size (%d bytes) is incompatible with the specified parameters " % f_size) +
            ("(n_channels=%d, dtype=%s, offset=%d)" % (n_channels, dtype, offset)))
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
# Config
#------------------------------------------------------------------------------

def config_path():
    """Path to the configuration file."""
    path = Path('~') / '.mtscomp'
    path = path.expanduser()
    return path


CONFIG_PATH = config_path()


def read_config(**kwargs):
    """Return the configuration dictionary, with default values and values set by the user
    in the configuration file."""
    params = dict(DEFAULT_CONFIG)

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open('r') as f:
            user_config = json.load(f)
    else:
        user_config = {}
    # Update the user defaults, then the values passed to the function.
    # We only update non-None values.
    for d in (user_config, kwargs):
        params.update({k: v for k, v in d.items() if v is not None})
    return Bunch(params)


def write_config(**kwargs):
    """Save some configuration key-values in the configuration file."""
    config = read_config(**kwargs)
    CONFIG_PATH.parent.mkdir(exist_ok=True, parents=True)
    with CONFIG_PATH.open('w') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return config


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

    def __init__(self, before_check=None, **kwargs):
        self.pool = None
        self.quiet = kwargs.pop('quiet', False)
        config = read_config(**kwargs)
        self.config = config
        self.chunk_duration = config.chunk_duration
        self.algorithm = config.algorithm
        assert self.algorithm == 'zlib', "Only zlib is currently supported."
        self.comp_level = config.comp_level
        self.do_time_diff = config.do_time_diff
        self.do_spatial_diff = config.do_spatial_diff
        self.n_threads = config.n_threads
        self.before_check = before_check or (lambda x: None)
        self.check_after_compress = config.check_after_compress
        self.chunk_order = config.chunk_order

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

        self.data_path = Path(data_path)

        # Get default parameters from the config file, if it exists.
        sample_rate = sample_rate or self.config.get('sample_rate', None)
        if not sample_rate:
            raise ValueError("Please provide a sample rate (-s option in the command-line).")

        if str(data_path).endswith('.npy'):
            # NPY files.
            self.data = np.load(data_path, mmap_mode='r')
            self.shape = self.data.shape
            if self.data.ndim >= 3:
                self.data = np.reshape(self.data, (-1, self.data.shape[-1]))
            self.dtype = dtype = self.data.dtype
            self.n_channels = n_channels = self.data.shape[1]
        else:
            # Raw binary files.
            n_channels = n_channels or self.config.get('n_channels', None)
            if not n_channels:
                raise ValueError("Please provide n_channels (-n option in the command-line).")
            dtype = dtype or self.config.get('dtype', None)
            if not dtype:
                raise ValueError("Please provide a dtype (-d option in the command-line).")
            self.dtype = np.dtype(dtype)
            self.data = load_raw_data(data_path, n_channels=n_channels, dtype=self.dtype)
            self.shape = self.data.shape

        self.sample_rate = float(sample_rate)
        assert sample_rate > 0
        assert n_channels > 0
        self.file_size = self.data.size * self.data.itemsize
        assert self.data.ndim == 2
        self.n_samples, self.n_channels = self.data.shape
        assert self.n_samples > 0
        assert self.n_channels > 0
        assert n_channels == self.n_channels
        duration = self.data.shape[0] / self.sample_rate
        logger.info(
            "Opening %s, duration %.1fs, %d channels.", data_path, duration, self.n_channels)
        self._compute_chunk_bounds()
        self.sha1_compressed = hashlib.sha1()
        self.sha1_uncompressed = hashlib.sha1()

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
            'chunk_order': self.chunk_order,
            'sha1_compressed': self.sha1_compressed.hexdigest(),
            'sha1_uncompressed': self.sha1_uncompressed.hexdigest(),
            'shape': self.shape
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
        # order=Fortran: Transposing (demultiplexing) the chunk may save a few %.
        chunkdc = zlib.compress(chunkd.tobytes(order=self.chunk_order))
        ratio = 100 - 100 * len(chunkdc) / (chunk.size * chunk.itemsize)
        logger.debug("Chunk %d/%d: -%.3f%%.", chunk_idx + 1, self.n_chunks, ratio)
        return chunk_idx, (chunk, chunkdc)

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
        logger.debug('\n'.join('%s = %s' % (k, v) for (k, v) in self.config.items()))
        ts = ' on %d CPUs.' % self.n_threads if self.n_threads > 1 else '.'
        logger.info("Starting compression" + ts)
        with open(out, 'wb') as fb:
            for batch in tqdm(range(self.n_batches), desc='Compressing', disable=self.quiet):
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
                    uncompressed_chunk, compressed_chunk = compressed_chunks[chunk_idx]
                    fb.write(compressed_chunk)
                    # Append the chunk offsets.
                    length = len(compressed_chunk)
                    offset += length
                    self.chunk_offsets.append(offset)
                    # Compute the SHA1 hashes.
                    self.sha1_uncompressed.update(uncompressed_chunk)
                    self.sha1_compressed.update(compressed_chunk)
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

    def __init__(self, **kwargs):
        self.pool = None
        self.cdata = None
        self.quiet = kwargs.pop('quiet', False)
        self.config = read_config(**kwargs)
        self.cache_size = self.config.cache_size
        self.check_after_decompress = self.config.check_after_decompress

    def open(self, cdata, cmeta=None):
        """Open a compressed data file.

        Parameters
        ----------

        cdata : str or Path
            Path to the compressed data file.
        cmeta : str or Path or dict
            Path to the compression header JSON file, or its contents as a Python dictionary.

        """
        # Read metadata file.
        if cmeta is None:
            cmeta = Path(cdata).with_suffix('.ch')
        if not isinstance(cmeta, dict):
            with open(cmeta, 'r') as f:
                cmeta = json.load(f)
        assert isinstance(cmeta, dict)
        self.cmeta = Bunch(cmeta)
        # Read some values from the metadata file.
        self.n_channels = self.cmeta.n_channels
        self.sample_rate = self.cmeta.sample_rate
        self.dtype = np.dtype(self.cmeta.dtype)
        self.chunk_offsets = self.cmeta.chunk_offsets
        self.chunk_bounds = self.cmeta.chunk_bounds
        self.chunk_order = self.cmeta.chunk_order
        self.n_samples = self.chunk_bounds[-1]
        self.n_chunks = len(self.chunk_bounds) - 1
        self.shape = (self.n_samples, self.n_channels)
        self.ndim = 2

        # Batches.
        self.batch_size = self.config.n_threads  # in each batch, there is 1 chunk per thread.
        self.n_batches = int(np.ceil(self.n_chunks / self.batch_size))

        # Open data.
        if isinstance(cdata, (str, Path)):
            if Path(cdata).suffix in ('.bin', '.dat'):  # pragma: no cover
                # This can arise if trying to decompress an already-decompressed file.
                logger.error("File to decompress has unexpected extension %s.", Path(cdata).suffix)
            cdata = open(cdata, 'rb')
        self.cdata = cdata

        self.set_cache_size()

    def set_cache_size(self, cache_size=None):
        """Set the LRU cache size for self.read_chunk()."""
        if cache_size != self.cache_size:
            cache_size = cache_size or self.cache_size
            assert cache_size > 0
            self.read_chunk = lru_cache(maxsize=cache_size)(self.read_chunk)
            self.cache_size = cache_size

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
        logger.debug(f"Reading compressed chunk {chunk_idx}, {chunk_start}, {chunk_length}")
        # Load the compressed chunk from the file.
        if hasattr(os, 'pread'):
            # On UNIX, we use an atomic system call to read N bytes of data from the file so that
            # this call is thread-safe.
            cbuffer = os.pread(self.cdata.fileno(), chunk_length, chunk_start)
        else:  # pragma: no cover
            # Otherwise, we have to use two system calls, a seek and a read, and we need to
            # put a lock so that we're sure that this pair of calls is atomic across threads.
            with lock:
                self.cdata.seek(chunk_start)
                cbuffer = self.cdata.read(chunk_length)
        assert len(cbuffer) == chunk_length
        # Decompress the chunk.
        try:
            buffer = zlib.decompress(cbuffer)
        except Exception:  # pragma: no cover
            raise IOError("Compressed chunk #%d is corrupted." % chunk_idx)
        chunk = np.frombuffer(buffer, self.dtype)
        assert chunk.dtype == self.dtype
        # Find the chunk shape.
        i0, i1 = self.chunk_bounds[chunk_idx:chunk_idx + 2]
        assert i0 <= i1
        n_samples_chunk = i1 - i0
        assert chunk.size == n_samples_chunk * self.n_channels
        # Reshape the chunk.
        chunk = chunk.reshape((n_samples_chunk, self.n_channels), order=self.chunk_order)
        chunki = cumsum_along_axis(chunk, axis=1 if self.cmeta.do_spatial_diff else None)
        chunki = cumsum_along_axis(chunki, axis=0 if self.cmeta.do_time_diff else None)
        assert chunki.dtype == chunk.dtype
        assert chunki.shape == chunk.shape == (n_samples_chunk, self.n_channels)
        return np.ascontiguousarray(chunki)  # needed when using F ordering in compression

    def _decompress_chunk(self, chunk_idx):
        """Decompress a chunk."""
        logger.debug("Starting decompression of chunk %d.", chunk_idx)
        assert 0 <= chunk_idx <= self.n_chunks - 1
        chunk_start = self.chunk_offsets[chunk_idx]
        chunk_length = self.chunk_offsets[chunk_idx + 1] - chunk_start
        return chunk_idx, self.read_chunk(chunk_idx, chunk_start, chunk_length)

    def decompress_chunks(self, chunk_ids, pool=None):
        # Return a dictionary chunk_idx: decompressed_chunk
        assert pool
        out = dict(pool.map(self._decompress_chunk, chunk_ids))
        assert set(out.keys()) == set(chunk_ids)
        return out

    def _validate_index(self, i, value_for_none=0):
        if i is None:
            i = value_for_none
        elif i < 0:
            i += self.n_samples
        i = _clip(i, 0, self.n_samples)
        assert 0 <= i <= self.n_samples
        return i

    def _chunks_for_interval(self, i0, i1):
        """Find the first and last chunks to be loaded in order to get the data between
        time samples `i0` and `i1`."""

        i0 = _clip(i0, 0, self.n_samples - 1)
        i1 = _clip(i1, i0, self.n_samples - 1)
        assert 0 <= i0 <= i1 <= self.n_samples

        first_chunk = _clip(
            bisect.bisect_right(self.chunk_bounds, i0) - 1, 0, self.n_chunks - 1)
        assert 0 <= first_chunk < self.n_chunks
        assert self.chunk_bounds[first_chunk] <= i0
        # Ensure we don't load unnecessary chunks.
        assert self.chunk_bounds[first_chunk + 1] > i0

        last_chunk = _clip(
            bisect.bisect_right(self.chunk_bounds, i1, lo=first_chunk) - 1, 0, self.n_chunks - 1)
        assert 0 <= last_chunk < self.n_chunks
        assert self.chunk_bounds[last_chunk + 1] >= i1
        # Ensure we don't load unnecessary chunks.
        assert self.chunk_bounds[last_chunk] <= i1

        assert 0 <= first_chunk <= last_chunk <= self.n_chunks - 1
        return first_chunk, last_chunk

    def start_thread_pool(self):
        """Start the thread pool for multithreaded decompression."""
        if self.pool:  #  pragma: no cover
            return self.pool
        logging.debug("Starting thread pool with %d CPUs.", self.batch_size)
        self.pool = ThreadPool(self.batch_size)
        return self.pool

    def stop_thread_pool(self):
        """Stop the thread pool."""
        logger.debug("Stopping thread pool.")
        self.pool.close()
        self.pool.join()
        self.pool = None

    def tofile(self, out, overwrite=False):
        """Write the decompressed array to disk."""
        if out is None:
            out = Path(self.cdata.name).with_suffix('.bin')
        out = Path(out)
        # Handle overwriting.
        if not overwrite and out.exists():  # pragma: no cover
            raise ValueError(
                "The output file %s already exists, use --overwrite or specify another "
                "output path." % out)
        elif overwrite and out.exists():
            # NOTE: for some reason, on my computer (Ubuntu 19.04 on fresh ext4 HDD), closing the
            # output file is very slow if it is being overwritten, rather than if it's a new file.
            # So deleting the file to be overwritten before overwriting it saves ~10 seconds.
            logger.debug("Deleting %s.", out)
            out.unlink()
        # Create the thread pool.
        self.start_thread_pool()
        with open(out, 'wb') as fb:
            for batch in tqdm(range(self.n_batches), desc='Decompressing', disable=self.quiet):
                first_chunk = self.batch_size * batch  # first included
                last_chunk = min(self.batch_size * (batch + 1), self.n_chunks)  # last excluded
                assert 0 <= first_chunk < last_chunk <= self.n_chunks
                logger.debug(
                    "Processing batch #%d/%d with chunks %s.",
                    batch + 1, self.n_batches, ', '.join(map(str, range(first_chunk, last_chunk))))
                # Decompress all chunks in the batch.
                decompressed_chunks = self.decompress_chunks(
                    range(first_chunk, last_chunk), self.pool)
                # Write the batch chunks to disk.
                # Warning: we need to process the chunks in order.
                for chunk_idx in sorted(decompressed_chunks.keys()):
                    decompressed_chunk = decompressed_chunks[chunk_idx]
                    fb.write(decompressed_chunk)
            dsize = fb.tell()
        assert dsize == self.chunk_bounds[-1] * self.n_channels * self.dtype.itemsize
        # Close the thread pool.
        self.stop_thread_pool()
        logger.info("Wrote %s (%.1f GB).", out, dsize / 1024 ** 3)
        if self.check_after_decompress:
            decompressed = load_raw_data(out, n_channels=self.n_channels, dtype=self.dtype)
            check(decompressed, self.cdata, self.cmeta)
            logger.debug("Automatic integrity check after decompression PASSED.")

    def close(self):
        """Close all file handles."""
        if self.cdata:
            self.cdata.close()

    def chop(self, n_chunks, out=None):
        assert n_chunks > 0
        if n_chunks >= self.n_chunks:  # pragma: no cover
            logger.warning("Cannot chop more chunks than there are in the original file.")
            return
        # self.cdata.seek(0)
        assert n_chunks < self.n_chunks

        # if out is None:
        #     out = self.cdata.with_suffix('.chopped.cbin')
        assert out is not None, "The output path must be specified."
        out = Path(out)
        assert out.suffix == '.cbin'
        if out.exists():  # pragma: no cover
            raise IOError("File %s already exists." % out)
        out.parent.mkdir(exist_ok=True, parents=True)

        # Write the chopped .cbin file
        with open(out, 'wb') as f:
            offset = 0
            for i in tqdm(range(n_chunks), desc='Chopping %d chunks' % n_chunks):
                chunk_length = self.chunk_offsets[i + 1] - self.chunk_offsets[i]
                with lock:
                    self.cdata.seek(offset)
                    cbuffer = self.cdata.read(chunk_length)
                assert len(cbuffer) == chunk_length
                f.write(cbuffer)
                offset += chunk_length
                assert self.cdata.tell() == offset
                assert f.tell() == offset
        # logger.info("Wrote %s.", out)

        # Write the .ch file.
        outmeta = out.with_suffix('.ch')
        if outmeta.exists():  # pragma: no cover
            raise IOError("File %s already exists." % out)

        cmeta = Bunch(self.cmeta.copy())
        cmeta['chunk_bounds'] = cmeta['chunk_bounds'][:n_chunks + 1]
        cmeta['chunk_offsets'] = cmeta['chunk_offsets'][:n_chunks + 1]
        assert cmeta['chunk_offsets'][-1] == offset
        cmeta['sha1_compressed'] = None
        cmeta['sha1_uncompressed'] = None
        cmeta['chopped'] = True
        with open(outmeta, 'w') as f:
            json.dump(cmeta, f, indent=2, sort_keys=True)
        # logger.info("Wrote %s.", outmeta)

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
            if first_chunk < last_chunk:
                # Concatenate all chunks.
                ns = sum(chunk.shape[0] for chunk in chunks)
                arr = np.empty((ns, self.n_channels), dtype=self.dtype)
                arr = np.concatenate(chunks, out=arr)
            else:
                assert len(chunks) == 1
                arr = chunks[0]
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
            elif len(item) == 2 and np.isscalar(item[0]):
                return self[item[0]][item[1]]
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
    try:
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
    finally:
        unc.close()


def compress(
        path, out=None, outmeta=None, sample_rate=None, n_channels=None, dtype=None, **kwargs):
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

    w = Writer(**kwargs)
    w.open(path, sample_rate=sample_rate, n_channels=n_channels, dtype=dtype)
    length = w.write(out, outmeta)
    w.close()
    return length


def decompress(cdata, cmeta=None, out=None, write_output=False, overwrite=False, **kwargs):
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
    write_output : bool
        Whether to write the output to a file.
    overwrite : bool
        Whether to overwrite the output file if it already exists.

    Returns
    -------

    reader : Reader instance
        This object implements the NumPy slicing syntax to access
        parts of the actual data as NumPy arrays.

    """
    if out:
        write_output = True
    r = Reader(**kwargs)
    r.open(cdata, cmeta)
    if write_output:
        r.tofile(out, overwrite=overwrite)
    return r


#------------------------------------------------------------------------------
# Command-line API utils
#------------------------------------------------------------------------------

def exception_handler(
        exception_type, exception, traceback, debug_hook=sys.excepthook):  # pragma: no cover
    if '--debug' in sys.argv or '-v' in sys.argv:
        debug_hook(exception_type, exception, traceback)
    else:
        print("%s: %s" % (exception_type.__name__, exception))


def _shared_options(parser):
    parser.add_argument('-nc', '--no-check', action='store_false', help='no check')
    parser.add_argument('-v', '--debug', action='store_true', help='verbose')
    parser.add_argument('-p', '--cpus', type=int, help='number of CPUs to use')


def _args_to_config(parser, args, compress=True):
    pargs = parser.parse_args(args)
    # parser.nc=True means that the flag was not given => switch to default (True or config)
    check_after = None if pargs.no_check is True else False
    kwargs = dict(
        n_threads=pargs.cpus,
    )
    if compress:
        kwargs.update(
            sample_rate=pargs.sample_rate,
            n_channels=pargs.n_channels,
            dtype=pargs.dtype.strip() if pargs.dtype else pargs.dtype,
            chunk_duration=pargs.chunk,
            check_after_compress=check_after,
        )
    else:
        kwargs.update(
            check_after_decompress=check_after,
        )
    config = read_config(**kwargs)
    return pargs, config


#------------------------------------------------------------------------------
# Command-line API: mtscomp
#------------------------------------------------------------------------------

def mtscomp_parser():
    """Command-line interface to compress a file."""
    parser = argparse.ArgumentParser(description='Compress a raw binary file.')

    parser.add_argument(
        'path', type=str, help='input path of a raw binary file')

    parser.add_argument(
        'out', type=str, nargs='?',
        help='output path of the compressed binary file (.cbin)')

    parser.add_argument(
        'outmeta', type=str, nargs='?',
        help='output path of the compression metadata JSON file (.ch)')

    parser.add_argument('-d', '--dtype', type=str, help='data type')
    parser.add_argument('-s', '--sample-rate', type=float, help='sample rate')
    parser.add_argument('-n', '--n-channels', type=int, help='number of channels')
    parser.add_argument('-c', '--chunk', type=int, help='chunk duration')

    _shared_options(parser)

    parser.add_argument(
        '--set-default', action='store_true', help='set the specified parameters as the default')

    return parser


def mtscomp(args=None):
    """Compress a file."""
    sys.excepthook = exception_handler
    parser = mtscomp_parser()
    pargs, config = _args_to_config(
        parser, args or sys.argv[1:], compress=True)
    add_default_handler('DEBUG' if pargs.debug else 'INFO')
    if pargs.set_default:
        write_config(**config)
    compress(pargs.path, pargs.out, pargs.outmeta, **config)


#------------------------------------------------------------------------------
# Command-line API: mtsdecomp
#------------------------------------------------------------------------------

def mtsdecomp_parser():
    """Command-line interface to decompress a file."""
    parser = argparse.ArgumentParser(description='Decompress a raw binary file.')

    parser.add_argument(
        'cdata', type=str,
        help='path to the input compressed binary file (.cbin)')

    parser.add_argument(
        'cmeta', type=str, nargs='?',
        help='path to the input compression metadata JSON file (.ch)')

    parser.add_argument(
        '-o', '--out', type=str, nargs='?',
        help='path to the output decompressed file (.bin)')

    parser.add_argument('--overwrite', '-f', action='store_true', help='overwrite existing output')

    _shared_options(parser)

    return parser


def mtsdecomp(args=None):
    """Decompress a file."""
    sys.excepthook = exception_handler
    parser = mtsdecomp_parser()
    pargs, config = _args_to_config(parser, args or sys.argv[1:], compress=False)
    add_default_handler('DEBUG' if pargs.debug else 'INFO')
    decompress(
        pargs.cdata, pargs.cmeta, out=pargs.out,
        # check_after_decompress=config.check_after_compress,
        write_output=True,
        overwrite=pargs.overwrite,
        **config
    )


#------------------------------------------------------------------------------
# Command-line API: mtsdesc
#------------------------------------------------------------------------------

def mtsdesc(args=None):
    """Describe a compressed file."""
    sys.excepthook = exception_handler
    parser = mtsdecomp_parser()
    parser.description = 'Describe a compressed file.'
    pargs = parser.parse_args(args or sys.argv[1:])
    r = Reader()
    r.open(pargs.cdata, pargs.cmeta)
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


#------------------------------------------------------------------------------
# Command-line API: mtschop
#------------------------------------------------------------------------------

def mtschop(args=None):
    """Chop a compressed file to N chunks without decompressing it."""
    sys.excepthook = exception_handler
    parser = argparse.ArgumentParser(
        description='Chop a compressed file to N chunks without decompressing it.')

    parser.add_argument(
        'cdata', type=str,
        help='path to the input compressed binary file (.cbin)')

    parser.add_argument('-n', '--n_chunks', type=int, help='number of chunks to chop')

    parser.add_argument(
        '-o', '--out', type=str,
        help='path to the output chopped compressed file (.cbin)')

    _shared_options(parser)

    pargs = parser.parse_args(args or sys.argv[1:])
    r = Reader()
    r.open(pargs.cdata)
    r.chop(pargs.n_chunks, pargs.out)
    r.close()
