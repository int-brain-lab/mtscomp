# -*- coding: utf-8 -*-

"""SVD-based raw ephys data lossy compression."""


#-------------------------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------------------------

import argparse
from itertools import islice
import logging
from pathlib import Path

from tqdm import tqdm
import numpy as np
from numpy.linalg import inv
from numpy.lib.format import open_memmap

from .mtscomp import Bunch, decompress, add_default_handler


logger = logging.getLogger('mtscomp')
logger.setLevel(logging.DEBUG)


#-------------------------------------------------------------------------------------------------
# Constants
#-------------------------------------------------------------------------------------------------

DOWNSAMPLE_FACTOR = 4
CHUNKS_EXCERPTS = 20

FILE_EXTENSION_LOSSY = '.lossy.npy'
FILE_EXTENSION_SVD = '.svd.npz'
DEFAULT_QUANTILE = .0025
# UINT8_MARGIN = .05
DTYPE = np.uint8
MAX_UINT8 = 255


#-------------------------------------------------------------------------------------------------
# Util classes
#-------------------------------------------------------------------------------------------------

class SVD(Bunch):
    """A special dictionary that holds information about lossy compression.

    It mostly holds the SVD matrices necessary to reconstruct the original signal, as well
    as scaling factors to resample to/from uint8.

    """

    def __init__(
            self, U, sigma, rank=None, ab=None, minmax=None, quantile=None,
            sample_rate=None, downsample_factor=DOWNSAMPLE_FACTOR):
        super(SVD, self).__init__()
        self.U = U  # the "U" in the "U @ sigma @ V" SVD decomposition
        self.n_channels = U.shape[0]  # number of channels
        assert sigma.shape == (self.n_channels,)

        self.Usigma_inv = inv(U @ np.diag(sigma))  # inverse of "U @ sigma"
        assert self.Usigma_inv.shape == self.U.shape

        self.sigma = sigma  # the diagonal of the SVD decomposition
        self.rank = rank  # the number of SVD components to keep
        self.ab = ab  # the uint8 scaling factors "y = ax+b"
        self.sample_rate = sample_rate  # the sampling rate
        self.downsample_factor = downsample_factor  # the downsample factor, an integer
        self.minmax = minmax  # the min and max of the signal across all channels
        self.quantile = quantile  # the q and 1-q quantiles of the signal

    def save(self, path):
        """Save this SVD object to a .npz file."""
        assert self.U is not None
        assert self.Usigma_inv is not None
        assert self.sigma is not None
        assert self.ab is not None
        assert self.n_channels >= 1
        assert self.rank >= 1
        assert self.sample_rate > 0
        assert self.downsample_factor >= 1
        assert self.minmax is not None
        assert self.quantile is not None

        np.savez(
            path,
            U=self.U,
            sigma=self.sigma,
            ab=self.ab,
            minmax=np.array(self.minmax),
            quantile=np.array(self.quantile),

            # NOTE: need to convert to regular arrays for np.savez
            rank=np.array([self.rank]),
            sample_rate=np.array([self.sample_rate]),
            downsample_factor=np.array([self.downsample_factor]),
        )
        logger.info(f"SVD file saved to {path}")

    def __repr__(self):
        return f"<SVD n_channels={self.n_channels}, rank={self.rank}>"


def load_svd(path):
    """Load a .npz file containing the SVD information, and return a SVD object."""
    d = np.load(path)
    svd = SVD(
        U=d['U'],
        sigma=d['sigma'],
        ab=d['ab'],
        rank=int(d['rank'][0]),
        sample_rate=d['sample_rate'][0],
        downsample_factor=int(d['downsample_factor'][0]),
        minmax=d['minmax'],
        quantile=d['quantile'],
    )
    assert svd.n_channels >= 1
    return svd


#-------------------------------------------------------------------------------------------------
# Util functions
#-------------------------------------------------------------------------------------------------

def _car(x):
    """Common average referencing (remove the mean along time).

    Parameters
    ----------

    x : ndarray (n_samples, n_channels)
        Signal.

    """
    assert x.ndim == 2
    # ns, nc
    assert x.shape[0] > x.shape[1]
    y = x.astype(np.float32)
    y -= y.mean(axis=0)
    return y


def _downsample(x, factor=DOWNSAMPLE_FACTOR):
    """Hard downsampling."""
    assert x.ndim == 2
    ns, nc = x.shape
    # ns, nc
    assert x.shape[0] > x.shape[1]
    y = x.T[:, :ns - ns % factor].reshape((nc, ns // factor, factor)).mean(axis=2)
    # nc, ns
    return y


def _svd(x):
    """Compute the SVD of a signal. Return a SVD object.

    Parameters
    ----------

    x : ndarray (n_channels, n_samples)
        Signal.

    """
    assert x.ndim == 2
    # nc, ns
    assert x.shape[0] < x.shape[1]

    U, sigma, _ = np.linalg.svd(x, full_matrices=False)
    return SVD(U, sigma)


def _uint8_coefs(x, q=DEFAULT_QUANTILE):
    """Compute the (a, b) rescaling coefficients to downsample a signal to uint8."""
    m, M = np.quantile(x, q), np.quantile(x, 1 - q)
    # m, M = x.min(), x.max()
    d = M - m
    assert d > 0

    # m -= d * margin
    # M += d * margin

    a = MAX_UINT8 / d
    b = m
    return a, b


def to_uint8(x, ab=None):
    """Downsample a signal to uint8. The rescaling coefficients can be passed or recomputed."""
    a, b = ab if ab is not None else _uint8_coefs(x)

    y = (x - b) * a
    # inverse: x = y / a + b

    overshoot = np.mean((y < 0) | (y > MAX_UINT8))
    if overshoot > 0:
        logger.debug(
            f"casting to {str(DTYPE)}: clipping {overshoot * 100:.3f}% of values")
    y = np.clip(y, 0, MAX_UINT8)

    return y.astype(DTYPE), (a, b)


def from_uint8(y, ab):
    """Resample a uint8 signal to a float32 signal, using the rescaling coefficients."""
    a, b = ab
    return y.astype(np.float32) / a + b


#-------------------------------------------------------------------------------------------------
# Processing functions
#-------------------------------------------------------------------------------------------------

def _preprocess_default(raw):
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

    pp = _car(raw)
    # pp is (ns, nc)
    assert pp.shape[0] > pp.shape[1]

    pp = _downsample(pp, factor=DOWNSAMPLE_FACTOR)
    # pp is (nc', ns)
    assert pp.shape[0] < pp.shape[1]

    return pp


def _get_excerpts(reader, kept_chunks=CHUNKS_EXCERPTS, preprocess=None):
    """Get evenly-spaced excerpts of a mtscomp-compressed file.

    Parameters
    ----------

    reader : mtscomp.Reader
        The input array.
    kept_chunks : int (default: 20)
        The number of 1-second excerpts to keep.
    preprocess : function `(n_samples, n_channels) int16 => (n_channels, n_samples) float32`
        The preprocessing function to run on each 1-second excerpt.

    Returns
    -------

    excerpts : ndarray (n_channels, n_samples_excerpts)
        The excerpts concatenated along the time axis.

    """

    assert reader
    assert kept_chunks >= 2
    preprocess = preprocess or _preprocess_default

    arrs = []
    n_chunks = reader.n_chunks
    assert reader.shape[0] > reader.shape[1]
    skip = max(1, n_chunks // kept_chunks)
    for chunk_idx, chunk_start, chunk_length in tqdm(
            islice(reader.iter_chunks(), 0, n_chunks + 1, skip),
            total=n_chunks // skip,
            desc="Extracting excerpts..."):

        chunk = reader.read_chunk(chunk_idx, chunk_start, chunk_length)
        # chunk is (ns, nc)
        assert chunk.shape[0] > chunk.shape[1]

        pp = preprocess(chunk)
        # pp is (nc, ns)
        assert chunk.shape[0] > chunk.shape[1]

        arrs.append(pp)

    excerpts = np.hstack(arrs)
    # excerpts is (nc, ns)
    assert excerpts.shape[0] < excerpts.shape[1]
    assert excerpts.shape[0] == reader.n_channels

    return excerpts


def excerpt_svd(reader, rank, kept_chunks=CHUNKS_EXCERPTS, preprocess=None):
    """Compute the SVD on evenly-spaced excerpts of a mtscomp-compressed file.

    Parameters
    ----------

    reader : mtscomp.Reader
        The input array.
    rank : int
        The number of SVD components to keep.
    kept_chunks : int (default: 20)
        The number of 1-second excerpts to keep.
    preprocess : function `(n_samples, n_channels) int16 => (n_channels, n_samples) float32`
        The preprocessing function to run on each 1-second excerpt.

    Returns
    -------

    svd : SVD instance
        An object containg the SVD information.

    """

    assert rank
    excerpts = _get_excerpts(reader, kept_chunks=kept_chunks, preprocess=preprocess)
    # excerpts is (nc, ns)
    assert excerpts.shape[0] < excerpts.shape[1]

    # Compute the SVD of the excerpts.
    svd = _svd(excerpts)
    assert svd.U.shape[0] == reader.n_channels

    svd.minmax = (excerpts.min(), excerpts.max())
    svd.quantile = (
        np.quantile(excerpts.ravel(), DEFAULT_QUANTILE),
        np.quantile(excerpts.ravel(), 1 - DEFAULT_QUANTILE))

    svd.sample_rate = reader.sample_rate
    svd.downsample_factor = DOWNSAMPLE_FACTOR
    svd.rank = min(rank, svd.n_channels)

    assert svd
    return svd


def _compress_chunk(raw, svd, preprocess=None):
    """Compress a chunk of data.

    Parameters
    ----------

    raw : ndarray (n_samples, n_channels)
        The input array.
    svd : SVD instance
        The SVD object returned by `excerpt_svd()`
    preprocess : function `(n_samples, n_channels) int16 => (n_channels, n_samples) float32`
        The preprocessing function to run on each 1-second excerpt.

    Returns
    -------

    lossy : ndarray (rank, n_samples)
        The compressed signal.

    """

    # raw is (ns, nc)
    assert raw.shape[0] > raw.shape[1]
    preprocess = preprocess or _preprocess_default

    pp = preprocess(raw)
    # pp is (nc, ns)
    assert pp.shape[0] < pp.shape[1]

    rank = svd.rank
    assert rank > 0

    lossy = (svd.Usigma_inv @ pp)[:rank, :]
    # lossy is (rank, ns)
    assert lossy.shape[0] < lossy.shape[1]

    return lossy


def _decompress_chunk(lossy, svd, rank=None):
    """Decompress a chunk of data.

    Parameters
    ----------

    lossy : ndarray (rank, n_samples)
        The lossy-compressed array.
    svd : SVD instance
        The SVD object returned by `excerpt_svd()`
    rank : int (default: None)
        If set, override the SVD rank (must be lower than the SVD rank).
        Used to simulate reconstruction with a smaller rank.

    Returns
    -------

    arr : ndarray (n_channels, n_samples)
        The reconstructed signal.

    """

    # lossy is (nc, ns)
    assert lossy.shape[0] < lossy.shape[1]

    assert svd
    assert isinstance(svd, SVD)
    U, sigma = svd.U, svd.sigma
    rank = rank or svd.rank
    rank = min(rank, svd.rank)
    rank = min(rank, svd.n_channels)

    assert rank > 0

    # arr is (nc, ns)
    arr = (U[:, :rank] @ np.diag(sigma[:rank]) @ lossy[:rank, :])

    return arr


#-------------------------------------------------------------------------------------------------
# Mock Reader classes
#-------------------------------------------------------------------------------------------------

class ArrayReader:
    """Wrapper to an array-like object, that provides the same interface as a mtscomp.Reader."""

    def __init__(self, arr, sample_rate=None):
        assert sample_rate > 0
        self.sample_rate = sample_rate
        self.chunk_length = int(np.ceil(sample_rate))

        self._arr = arr
        assert arr.ndim == 2
        # arr shape is (n_samples, n_channels)
        assert arr.shape[0] > arr.shape[1]
        self.shape = arr.shape
        self.n_samples, self.n_channels = self.shape
        self.n_chunks = int(np.ceil(self.n_samples / float(sample_rate)))

    def iter_chunks(self, last_chunk=None):
        offset = 0
        n = (last_chunk + 1) if last_chunk else self.n_chunks
        for i in range(n):
            # (chunk_idx, chunk_start, chunk_length)
            yield (i, offset, min(self.chunk_length, self.n_samples - offset))
            offset += self.chunk_length

    def read_chunk(self, chunk_idx, chunk_start, chunk_length):
        return self._arr[chunk_start:chunk_start + chunk_length, :]

    def __getitem__(self, idx):
        return self._arr[idx]


#-------------------------------------------------------------------------------------------------
# Compressor
#-------------------------------------------------------------------------------------------------

def compress_lossy(
        path_cbin=None, cmeta=None, reader=None, rank=None, max_chunks=0,
        chunks_excerpts=None, downsampling_factor=None,
        preprocess=None, overwrite=False, dry_run=False,
        out_lossy=None, out_svd=None):
    """Compress a .cbin file or an arbitrary signal array.

    Parameters
    ----------

    path_cbin : str or Path
        Path to the compressed data binary file (typically ̀.cbin` file extension).
    cmeta : str or Path (default: None)
        Path to the compression header JSON file (typically `.ch` file extension).
    reader : Reader instance
        A mtscomp.Reader or ArrayReader object.
    rank : int (mandatory)
        Number of SVD components to keep in the compressed file.
    max_chunks : int (default: None)
        Maximum number of chunks to compress (use None to compress the entire file).
    chunks_excerpts : int (default: 20)
        Number of evenly-spaced 1-second chunks to extract to compute the SVD.
    downsampling_factor : int
        Number of times the original will be downsampled.
    preprocess : function `(n_samples, n_channels) int16 => (n_channels, n_samples) float32`

    svd : SVD instance
        The SVD object returned by `excerpt_svd()`
    preprocess : function `(n_samples, n_channels) int16 => (n_channels, n_samples) float32`
        The preprocessing function to run on each chunk.
    overwrite : bool (default: False)
        Whether the lossy compressed files may be overwritten.
    dry_run : bool (default: False)
        If true, the lossy compressed files will not be written.
    out_lossy : str or Path
        Path to the output `.lossy.npy` file.
    out_svd : str or Path
        Path to the output `.svd.npz` SVD file.

    Returns
    -------

    out_lossy : str or Path
        Path to the output `.lossy.npy` file.

    """

    # Check arguments.
    assert rank, "The rank must be set"
    assert path_cbin or reader, "The raw ephys data file must be specified"
    preprocess = preprocess or _preprocess_default

    chunks_excerpts = chunks_excerpts or CHUNKS_EXCERPTS
    downsampling_factor = downsampling_factor or DOWNSAMPLE_FACTOR

    assert downsampling_factor >= 1
    assert chunks_excerpts >= 2

    # Create a mtscomp Reader.
    if path_cbin:
        reader = decompress(path_cbin, cmeta=cmeta)
    assert reader
    sr = int(reader.sample_rate)
    ns = reader.n_samples
    nc = reader.n_channels
    n_chunks = reader.n_chunks if not max_chunks else max_chunks
    if max_chunks:
        ns = max_chunks * sr  # NOTE: assume 1-second chunks

    assert n_chunks > 0
    assert sr > 0
    assert ns > 0
    assert nc > 0
    assert rank <= nc, "The rank cannot exceed the number of channels"

    # Filenames.
    if out_lossy is None and path_cbin:
        out_lossy = Path(path_cbin).with_suffix(FILE_EXTENSION_LOSSY)
    out_lossy = Path(out_lossy)
    assert out_lossy, "An output file path for the .lossy.npy file must be provided"

    if out_svd is None:
        out_svd = Path(out_lossy).with_suffix('').with_suffix(FILE_EXTENSION_SVD)
    out_svd = Path(out_svd)
    assert out_svd, "An output file path for the .svd.npz file must be provided"

    if dry_run:
        return out_lossy

    # Create a new memmapped npy file
    if not overwrite and out_lossy.exists():
        raise IOError(f"File {out_lossy} already exists.")
    shape = (ns // downsampling_factor, rank)
    lossy = open_memmap(out_lossy, 'w+', dtype=DTYPE, shape=shape)
    logger.info(f"Writing file {out_lossy}...")

    # Compute the SVD on an excerpt of the data.
    svd = excerpt_svd(reader, rank, kept_chunks=chunks_excerpts, preprocess=preprocess)

    # Compress the data.
    offset = 0
    for chunk_idx, chunk_start, chunk_length in tqdm(
            reader.iter_chunks(last_chunk=n_chunks - 1),
            desc='Compressing (lossy)...',
            total=n_chunks):

        # Decompress the chunk.
        raw = reader.read_chunk(chunk_idx, chunk_start, chunk_length)

        # raw is (ns, nc)
        assert raw.shape[0] > raw.shape[1]
        nsc, _ = raw.shape
        assert _ == nc

        # Compress the chunk.
        chunk_lossy = _compress_chunk(raw, svd, preprocess=preprocess)
        # chunk_lossy is (nc, ns)
        assert chunk_lossy.shape[0] < chunk_lossy.shape[1]

        # Write the compressed chunk to disk.
        l = chunk_lossy.shape[1]
        k = min(l, shape[0] - offset)
        assert k <= l
        chunk_lossy = chunk_lossy[:, :k]
        lossy[offset:offset + l, :], ab = to_uint8(chunk_lossy.T, svd.ab)
        # NOTE: keep the ab scaling factors for uint8 conversion only for the first chunk
        if svd.ab is None:
            svd.ab = ab

            # Save the SVD info to a npz file.
            svd.save(out_svd)

        offset += l

    extra = shape[0] - offset
    if extra > 0:
        lossy[-extra:, :] = lossy[-extra - 1, :]

    # # Save the SVD info to a npz file.
    # svd.save(out_svd)

    return out_lossy


#-------------------------------------------------------------------------------------------------
# Decompressor
#-------------------------------------------------------------------------------------------------

class LossyReader:
    """Array-like interface to the reconstruction of a lossy compressed file."""

    def __init__(self):
        self.path_lossy = None
        self.path_svd = None

    def open(self, path_lossy=None, path_svd=None):
        """Open a .lossy.npy/.svd.npz pair of files.

        Parameters
        ----------

        path_lossy : str or Path
            Path to the lossy compressed file (typically ̀`.lossy.npy` file extension).
        path_svd : str or Path (default: None)
            Path to the lossy compressed SVD file (typically ̀`.svd.npz` file extension).

        """
        assert path_lossy

        if path_svd is None:
            path_svd = Path(path_lossy).with_suffix('').with_suffix('.svd.npz')

        self.path_lossy = Path(path_lossy)
        self.path_svd = Path(path_svd)

        assert self.path_lossy
        assert self.path_svd
        assert self.path_lossy.exists()
        assert self.path_svd.exists()

        self._lossy = open_memmap(path_lossy, 'r')
        # ns, nc
        assert self._lossy.shape[0] > self._lossy.shape[1]
        assert self._lossy.dtype == DTYPE

        self._svd = load_svd(self.path_svd)
        self.rank = self._svd.rank
        self.downsample_factor = ds = self._svd.downsample_factor
        self.sample_rate = self._svd.sample_rate

        assert self.rank >= 1
        assert ds >= 1
        assert self._svd.ab is not None

        self.n_channels = self._svd.U.shape[0]
        self.n_samples = self._lossy.shape[0] * ds
        self.duration = self.n_samples / float(self.sample_rate)
        self.ndim = 2
        self.shape = (self.n_samples, self.n_channels)
        self.size = self.n_samples * self.n_channels
        self.size_bytes = self._lossy.size * self._lossy.itemsize
        self.itemsize = 1
        self.dtype = DTYPE

        size_original = 2 * self.n_channels * self.n_samples
        self.compression = size_original / float(self.size_bytes)

    def _decompress(self, lossy, rank=None):
        """Decompress a chunk.

        Parameters
        ----------

        lossy : ndarray (rank, n_samples)
            Compressed array.
        rank : int (default: None)
            If set, overrides the number of components to reuse for the reconstruction.

        Returns
        -------

        arr : ndarray (n_channels, n_samples)
            The reconstructed signal.

        """

        lossy_float = from_uint8(lossy, self._svd.ab).T
        return _decompress_chunk(lossy_float, self._svd, rank=rank).T

    def get(self, t0, t1, rank=None, cast_to_uint8=False, filter=None):
        """Return the reconstructed signal between two times (in seconds).

        Parameters
        ----------

        t0 : float
            Start time.
        t1 : float
            End time.
        rank : int (default: None)
            If set, overrides the number of components to reuse for the reconstruction.
        cast_to_uint8 : bool (default: False)
            Whether the reconstructed signal should be downsampled to uint8 (for viz purposes).
        filter : function
            Filter to apply to the signal (before casting to uint8).

        Returns
        -------

        arr : ndarray (n_channels, n_samples)
            The reconstructed signal.

        """
        ds = self._svd.downsample_factor
        i0 = int(round(t0 * float(self.sample_rate) / ds))
        i1 = int(round(t1 * float(self.sample_rate) / ds))
        lossy = self._lossy[i0:i1]
        arr = self._decompress(lossy, rank=rank)
        if filter:
            arr = filter(arr)
        if cast_to_uint8:
            m, M = self._svd.quantile
            d = M - m
            assert d > 0
            a = MAX_UINT8 / d
            b = m
            arr, _ = to_uint8(arr, ab=(a, b))
        return arr

    def t2s(self, t):
        return np.round(t * self.sample_rate).astype(np.uint64)

    def s2t(self, s):
        return s / float(self.sample_rate)

    def __getitem__(self, idx):
        """Array-like interface."""
        lossy = self._lossy[idx]
        return self._decompress(lossy)


def decompress_lossy(path_lossy=None, path_svd=None):
    """Decompress a .lossy.npy/.svd.npz pair of lossy compressed files.

    Parameters
    ----------

    path_lossy : str or Path
        Path to the `.lossy.npy` file.
    path_svd : str or Path (default: None)
        Path to the `.svd.npz` SVD file.

    Returns
    -------

    reader : LossyReader instance
        An array-like interface to the reconstructed signal.

    """

    reader = LossyReader()
    reader.open(path_lossy, path_svd=path_svd)
    return reader


#-------------------------------------------------------------------------------------------------
# Command-line API: mtscomp
#-------------------------------------------------------------------------------------------------

def mtsloss_parser():
    """Command-line interface to lossy-compress a .cbin file."""
    parser = argparse.ArgumentParser(description='Lossy compression of .cbin files.')

    parser.add_argument(
        'path', type=str, help='input path of a .cbin file')

    parser.add_argument(
        'out', type=str, nargs='?',
        help='output path of the lossy-compressed file (.lossy.npy)')

    parser.add_argument(
        'outsvd', type=str, nargs='?',
        help='output path of the compression metadata SVD file (.svd.npz)')

    parser.add_argument(
        '--rank', type=int, help='number of SVD components to keep during compression')

    parser.add_argument(
        '--excerpts', type=int, help='number of chunks to use when computing the SVD')

    parser.add_argument('--max-chunks', type=int, help='maximum number of chunks to compress')

    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')

    parser.add_argument('--dry', action='store_true', help='dry run')

    parser.add_argument('-v', '--debug', action='store_true', help='verbose')

    return parser


def mtsloss(args=None):
    """Compress a file."""
    parser = mtsloss_parser()
    pargs = parser.parse_args(args)
    add_default_handler('DEBUG' if pargs.debug else 'INFO')

    compress_lossy(
        path_cbin=pargs.path,
        out_lossy=pargs.out,
        out_svd=pargs.outsvd,
        chunks_excerpts=pargs.excerpts,
        rank=pargs.rank,
        max_chunks=pargs.max_chunks,
        overwrite=pargs.overwrite,
        dry_run=pargs.dry,
    )
