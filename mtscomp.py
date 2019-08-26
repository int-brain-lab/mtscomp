# -*- coding: utf-8 -*-

"""mtscomp: multichannel time series lossless compression in Python."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import json
import logging
import os.path as op
from pathlib import Path
import zlib

import numpy as np

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Global variables
#------------------------------------------------------------------------------

__version__ = '0.1.0a1'
FORMAT_VERSION = '1.0'
DO_DIFF = True


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


#------------------------------------------------------------------------------
# I/O utils
#------------------------------------------------------------------------------

def load_raw_data(path=None, n_channels=None, dtype=None, offset=None):
    """Load raw data at a given path."""
    path = Path(path)
    assert path.exists(), "File %s does not exist." % path
    assert dtype, "The data type must be provided."
    # Compute the array shape.
    item_size = np.dtype(dtype).itemsize
    offset = offset or 0
    n_samples = (op.getsize(str(path)) - offset) // (item_size * n_channels)
    shape = (n_samples, n_channels)
    # Memmap the file into a NumPy-like array.
    return np.memmap(str(path), dtype=dtype, shape=shape, offset=offset)


#------------------------------------------------------------------------------
# Low-level API
#------------------------------------------------------------------------------

class Writer:
    """Compress a raw data file."""
    def __init__(self, chunk_duration=1., compression_algorithm=None, compression_level=-1):
        self.chunk_duration = chunk_duration
        self.compression_algorithm = compression_algorithm or 'zlib'
        self.compression_level = compression_level

    def open(self, data_path, sample_rate=None, n_channels=None, dtype=None):
        self.sample_rate = sample_rate
        assert sample_rate > 0
        self.dtype = dtype
        self.data = load_raw_data(data_path, n_channels=n_channels, dtype=dtype)
        self.file_size = self.data.size * self.data.itemsize
        assert self.data.ndim == 2
        self.n_samples, self.n_channels = self.data.shape
        assert n_channels == self.n_channels
        logger.debug("Open %s with size %s.", data_path, self.data.shape)
        self._compute_chunk_bounds()

    def _compute_chunk_bounds(self):
        # Compute the chunk bounds.
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

    def get_meta(self):
        return {
            'version': FORMAT_VERSION,
            'compression_algorithm': self.compression_algorithm,
            'compression_level': self.compression_level,
            'do_diff': DO_DIFF,

            'n_channels': self.n_channels,
            'sample_rate': self.sample_rate,
            'chunk_bounds': self.chunk_bounds,
            'chunk_offsets': self.chunk_offsets,
        }

    def get_chunk(self, chunk_idx):
        assert 0 <= chunk_idx <= self.n_chunks - 1
        i0 = self.chunk_bounds[chunk_idx]
        i1 = self.chunk_bounds[chunk_idx + 1]
        return self.data[i0:i1, :]

    def write_chunk(self, chunk_idx, fb):
        # Retrieve the chunk data as a 2D NumPy array.
        chunk = self.get_chunk(chunk_idx)
        assert chunk.ndim == 2
        assert chunk.shape[1] == self.n_channels
        # Compute the diff along the time axis.
        if DO_DIFF:
            chunkd = np.concatenate((chunk[0, :][np.newaxis, :], np.diff(chunk, axis=0)), axis=0)
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
        ratio = 100 - 100 * csize / self.file_size
        logger.info("Wrote %s (-%.3f%%).", out, ratio)
        # Write the metadata file.
        with open(outmeta, 'w') as f:
            json.dump(self.get_meta(), f, indent=2, sort_keys=True)

    def close(self):
        """Close all file handles."""
        self.data._mmap.close()


class Reader:
    def open(self, cdata, cmeta):
        pass

    def open_cdata(self, cdata_path):
        pass

    def open_cmeta(self, cmeta_path):
        pass

    def read_chunk(self, chunk_idx):
        pass

    def read(self):
        pass

    def close(self):
        pass

    def __getitem__(self):
        # Implement NumPy array slicing, return a regular in-memory NumPy array.
        pass


#------------------------------------------------------------------------------
# High-level API
#------------------------------------------------------------------------------

def write(
        data, out, outmeta, sample_rate=None,
        data_type=None, n_channels=None,
        chunk_duration=None, compression_level=-1):
    """Compress a NumPy-like array (may be memmapped) into a compressed format
    (two files, out and outmeta).

    Parameters
    ----------

    data : NumPy-like array
        An array with shape `(n_samples, n_channels)`.
    out : str or file handle
        Output file for the compressed data.
    outmeta : str or file handle
        JSON file with metadata about the compression (see doc of `compress()`).
    sample_rate : float
        Sampling rate, in Hz.
    data_type : dtype
        The data type of the array in the raw data file.
    n_channels : int
        Number of channels in the file.
    chunk_duration : float
        Length of the chunks, in seconds.
    compression_level : int
        zlib compression level.

    Returns
    -------

    cdata : NumPy-like array
        Compressed version of the data, wrapped in a NumPy-like interface.

    Metadata dictionary
    -------------------

    Saved in the cmeta file as JSON.

    version : str
        Version number of the compression format.
    compression_algorithm : str
        Name of the compression algorithm.
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


def read(cdata, cmeta):
    """Read an array from a compressed dataset (two files, cdata and cmeta), and
    return a NumPy-like array (memmapping the compressed data on the fly).

    Parameters
    ----------

    cdata : str or file handle
        File with the compressed data (if file handle, should be open in `a` mode).
    cmeta : dict or str (path to the metadata file) or file handle
        A dictionary with metadata about the compression (see doc of `compress()`).

    Returns
    -------

    data : NumPy-like array
        Compressed version of the data, wrapped in a NumPy-like interface.

    """
