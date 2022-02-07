# -*- coding: utf-8 -*-

"""SVD-based raw ephys data lossy compression."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import argparse
from functools import lru_cache
from itertools import islice
import logging
import os
import os.path as op
from pathlib import Path
import sys

from tqdm import tqdm
import numpy as np
from numpy.lib.format import open_memmap

from .mtscomp import Bunch, decompress, Reader


logger = logging.getLogger(('mtscomp'))


#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

DOWNSAMPLE_FACTOR = 6

FILE_EXTENSION_LOSSY = '.lossy.npy'
FILE_EXTENSION_SVD = '.svd.npz'


#------------------------------------------------------------------------------
# Util classes
#------------------------------------------------------------------------------

class SVD(Bunch):
    def __init__(
            self, U, sigma, rank=None, a=1, b=0,
            sample_rate=None, downsample_factor=DOWNSAMPLE_FACTOR):
        super(SVD, self).__init__()
        self.U = U
        self.n_channels = U.shape[0]
        assert sigma.shape == (self.n_channels,)
        self.Usigma_inv = np.linalg.inv(U @ np.diag(sigma))
        assert self.Usigma_inv.shape == self.U.shape
        self.sigma = sigma
        self.rank = rank
        self.a = a
        self.b = b
        self.sample_rate = sample_rate
        self.downsample_factor = downsample_factor

    def __repr__(self):
        return f"<SVD n_channels={self.n_channels}, rank={self.rank}>"


#------------------------------------------------------------------------------
# Util functions
#------------------------------------------------------------------------------

def _car(x):
    assert x.ndim == 2
    # ns, nc
    assert x.shape[0] > x.shape[1]
    y = x.astype(np.float32)
    y -= y.mean(axis=0)
    return y


def _downsample(x, factor=1):
    assert x.ndim == 2
    ns, nc = x.shape
    # ns, nc
    assert x.shape[0] > x.shape[1]
    y = x.T.reshape((nc, (ns - ns % factor) // factor, factor)).mean(axis=2)
    # nc, ns
    return y


def _svd(x):
    U, sigma, _ = np.linalg.svd(x, full_matrices=False)
    return SVD(U, sigma)


def _uint8_coefs(x):
    # k = .05
    # m, M = np.quantile(x, k), np.quantile(x, 1 - k)
    m, M = x.min(), x.max()
    a = 255 / (M - m)
    b = m
    return a, b


def to_uint8(x, ab=None):
    a, b = ab if ab is not None else _uint8_coefs(x)
    y = (x - b) * a
    # x = y / a + b
    # assert np.all((0 <= y) & (y <= 255))
    return y.astype(np.uint8), (a, b)
    # return y, (a, b)


def from_uint8(y, ab):
    a, b = ab
    return y.astype(np.float32) / a + b


#------------------------------------------------------------------------------
# Processing functions
#------------------------------------------------------------------------------

def _preprocess(raw):
    assert raw.shape[0] > raw.shape[1]
    # raw is (ns, nc)

    pp = _car(raw)
    # pp is (ns, nc)
    assert pp.shape[0] > pp.shape[1]

    pp = _downsample(pp, factor=DOWNSAMPLE_FACTOR)
    # pp is (nc, ns)
    assert pp.shape[0] < pp.shape[1]

    return pp


def _get_excerpts(reader, kept_chunks=20):
    assert reader
    assert isinstance(reader, Reader)
    assert kept_chunks >= 2

    arrs = []
    n = 0
    n_chunks = reader.n_chunks
    assert reader.shape[0] > reader.shape[1]
    skip = n_chunks // kept_chunks
    for chunk_idx, chunk_start, chunk_length in tqdm(
            islice(reader.iter_chunks(), 0, n_chunks + 1, skip),
            total=n_chunks // skip, desc="extracting excerpts from the raw data"):

        chunk = reader.read_chunk(chunk_idx, chunk_start, chunk_length)
        # chunk is (ns, nc)
        assert chunk.shape[0] > chunk.shape[1]

        pp = _preprocess(chunk)
        # pp is (nc, ns)
        assert chunk.shape[0] > chunk.shape[1]

        arrs.append(pp)

    excerpts = np.hstack(arrs)
    # excerpts is (nc, ns)
    assert excerpts.shape[0] < excerpts.shape[1]
    assert excerpts.shape[0] == reader.n_channels

    return excerpts


def excerpt_svd(reader, rank, kept_chunks=20):
    assert rank
    excerpts = _get_excerpts(reader, kept_chunks=kept_chunks)
    # excerpts is (nc, ns)
    assert excerpts.shape[0] < excerpts.shape[1]

    # Compute the SVD of the excerpts.
    svd = _svd(excerpts)
    assert svd.U.shape[0] == reader.n_channels

    svd.sample_rate = reader.sample_rate
    svd.downsample_factor = DOWNSAMPLE_FACTOR
    svd.rank = min(rank, svd.n_channels)

    # NOTE: compute the uint8 scaling on the first second of data
    svd.ab = _uint8_coefs(excerpts[:, :int(svd.sample_rate)])

    assert svd
    return svd


def _compress_chunk(raw, svd):
    # raw is (ns, nc)
    assert raw.shape[0] > raw.shape[1]

    pp = _preprocess(raw)
    # pp is (nc, ns)
    assert pp.shape[0] < pp.shape[1]

    rank = svd.rank
    assert rank > 0
    assert svd.a != 0

    lossy = (svd.Usigma_inv @ pp)[:rank, :]
    # lossy is (nc, ns)
    assert lossy.shape[0] < lossy.shape[1]

    # lossy8, _ = to_uint8(lossy, (svd.a, svd.b))
    # return lossy8

    return lossy


def _decompress_chunk(lossy, svd, rank=None):
    # lossy is (nc, ns)
    assert lossy.shape[0] < lossy.shape[1]

    assert svd
    assert isinstance(svd, SVD)
    U, sigma = svd.U, svd.sigma
    rank = rank or svd.rank
    rank = min(rank, svd.rank)
    rank = min(rank, svd.n_channels)

    # lossy = from_uint8(lossy, (svd.a, svd.b))
    assert rank > 0

    return (U[:, :rank] @ np.diag(sigma[:rank]) @ lossy[:rank, :])


def compress_lossy(
        path=None, cmeta=None, rank=None, max_chunks=0, downsampling_factor=None,
        out_lossy=None, out_svd=None):

    # Check arguments.
    assert rank, "The rank must be set"
    assert path, "The raw ephys data file must be specified"

    if downsampling_factor is None:
        downsampling_factor = DOWNSAMPLE_FACTOR
    assert downsampling_factor >= 1

    # Create a mtscomp Reader.
    reader = decompress(path, cmeta=cmeta)
    ns = reader.n_samples
    nc = reader.n_channels
    n_chunks = reader.n_chunks if max_chunks == 0 else max_chunks
    assert n_chunks > 0
    assert rank <= nc, "The rank cannot exceed the number of channels"

    # Compute the SVD on an excerpt of the data.
    svd = excerpt_svd(reader, rank)

    # Filenames.
    if out_lossy is None:
        out_lossy = Path(path).with_suffix(FILE_EXTENSION_LOSSY)
    assert out_lossy

    if out_svd is None:
        out_svd = Path(path).with_suffix(FILE_EXTENSION_SVD)
    assert out_svd

    # Create a new memmapped npy file
    if out_lossy.exists():
        raise IOError(f"File {out_lossy} already exists.")
    shape = (ns // downsampling_factor, rank)
    lossy = open_memmap(out_lossy, 'w+', dtype=np.uint8, shape=shape)

    # Compress the data.
    offset = 0
    for chunk_idx, chunk_start, chunk_length in \
            tqdm(reader.iter_chunks(last_chunk=n_chunks - 1), total=n_chunks):

        # Decompress the chunk.
        raw = reader.read_chunk(chunk_idx, chunk_start, chunk_length)

        # raw is (ns, nc)
        assert raw.shape[0] > raw.shape[1]
        nsc, _ = raw.shape
        assert _ == nc

        # Process the chunk.
        pp = _preprocess(raw)
        # pp is (nc, ns)
        assert pp.shape[0] < pp.shape[1]

        # Compress the chunk.
        chunk_lossy = _compress_chunk(pp)
        # chunk_lossy is (nc, ns)
        assert chunk_lossy.shape[0] < chunk_lossy.shape[1]

        # Write the compressed chunk to disk.
        l = chunk_lossy.shape[1]
        lossy[offset:offset + l, :] = to_uint8(chunk_lossy.T, svd.ab)
        offset += l

    # Save the SVD info to a npz file.
    np.savez(out_svd, **SVD)


#------------------------------------------------------------------------------
# Decompressor
#------------------------------------------------------------------------------

class LossyReader:
    def __init__(self):
        self.path_lossy = None
        self.path_svd = None

    def open(self, path_lossy=None, path_svd=None):
        self.path_lossy = path_lossy
        self.path_svd = path_svd

        assert self.path_lossy
        assert self.path_svd

        self._lossy = open_memmap(path_lossy, 'r')
        # ns, nc
        assert self._lossy.shape[1] > self._lossy.shape[0]
        assert self._lossy.dtype == np.uint8

        self._svd = np.load(self.path_svd)
        ds = self._svd.downsample_factor
        assert ds >= 1

        self.n_channels = self._svd.U.shape[0]
        self.n_samples = self._lossy.shape[0] * ds
        self.ndim = 2
        self.shape = (self.n_samples, self.n_channels)
        self.size = self.n_samples * self.n_channels
        self.size_bytes = self._lossy.size * self._lossy.itemsize
        self.itemsize = 1
        self.dtype = np.uint8

    def _decompress(self, lossy, rank=None):
        lossy_float = from_uint8(lossy, self._svd.ab)
        return _decompress_chunk(lossy_float, self._svd, rank=rank)

    def get(self, i0, i1, rank=None):
        lossy = self._lossy[i0:i1]
        return self._decompress(lossy, rank=rank)

    def __get_item__(self, idx):
        lossy = self._lossy[idx]
        return self._decompress(lossy)


def decompress_lossy(path_lossy="file.lossy.npy", path_svd="file.svd.npz"):
    reader = LossyReader()
    reader.open(path_lossy, path_svd=path_svd)
    return reader


def test():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.grid"] = False
    sns.set_theme(style="white")

    EPHYS_DIR = Path("/home/cyrille/ephys/globus/KS023/")
    path_cbin = EPHYS_DIR / "raw.cbin"
    path_ch = EPHYS_DIR / "raw.ch"

    reader = decompress(path_cbin)
    rank = 40
    chunks_excerpts = 3

    svd = excerpt_svd(reader, rank, chunks_excerpts)

    raw = reader[:30000, :]
    nc = raw.shape[1]
    compression = DOWNSAMPLE_FACTOR * 2 * nc / float(rank)

    lossy = _compress_chunk(raw, svd)
    reconst = _decompress_chunk(lossy, svd, rank=rank)

    # plt.figure()
    # plt.hist(lossy.ravel(), bins=64, log=True)

    lossy8, ab = to_uint8(lossy)
    lossy_ = from_uint8(lossy8, ab)
    reconst8 = _decompress_chunk(lossy_, svd, rank=rank)

    nrows = 2
    fix, axs = plt.subplots(nrows, 1, sharex=True)

    axs[0].imshow(_preprocess(raw), cmap="gray", aspect="auto")
    axs[0].set_title(f"original")

    axs[1].imshow(reconst8, cmap="gray", aspect="auto")
    axs[1].set_title(f"rank={rank}, compression={compression:.1f}x")

    # for i in range(1, nrows):
    #     # rank = 50 * i
    #     compression = DOWNSAMPLE_FACTOR * 2 * nc / float(rank)
    #     axs[i].imshow(_decompress_chunk(lossy, svd, rank=rank), cmap="gray", aspect="auto")
    #     axs[i].set_title(f"rank={rank}, compression={compression:.1f}x")
    plt.show()
