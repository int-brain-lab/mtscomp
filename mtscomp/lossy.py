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
from numpy.linalg import inv
from numpy.lib.format import open_memmap

from .mtscomp import Bunch, decompress, Reader


logger = logging.getLogger('mtscomp')
logger.setLevel(logging.DEBUG)


#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

DOWNSAMPLE_FACTOR = 6
CHUNKS_EXCERPTS = 20

FILE_EXTENSION_LOSSY = '.lossy.npy'
FILE_EXTENSION_SVD = '.svd.npz'
UINT8_MARGIN = .05


#------------------------------------------------------------------------------
# Util classes
#------------------------------------------------------------------------------

class SVD(Bunch):
    def __init__(
            self, U, sigma, rank=None, ab=None,
            sample_rate=None, downsample_factor=DOWNSAMPLE_FACTOR):
        super(SVD, self).__init__()
        self.U = U
        self.n_channels = U.shape[0]
        assert sigma.shape == (self.n_channels,)
        self.Usigma_inv = inv(U @ np.diag(sigma))
        assert self.Usigma_inv.shape == self.U.shape
        self.sigma = sigma
        self.rank = rank
        self.ab = ab
        self.sample_rate = sample_rate
        self.downsample_factor = downsample_factor

    def save(self, path):
        assert self.U is not None
        assert self.Usigma_inv is not None
        assert self.sigma is not None
        assert self.ab is not None
        assert self.n_channels >= 1
        assert self.rank >= 1
        assert self.sample_rate > 0
        assert self.downsample_factor >= 1

        np.savez(
            path,
            U=self.U,
            # Usigma_inv=self.Usigma_inv,
            sigma=self.sigma,
            ab=self.ab,

            # NOTE: need to convert to regular arrays for np.savez
            rank=np.array([self.rank]),
            sample_rate=np.array([self.sample_rate]),
            downsample_factor=np.array([self.downsample_factor]),
        )

    def __repr__(self):
        # return f"<SVD n_channels={self.n_channels}, rank={self.rank}>"
        return super(SVD, self).__repr__()


def load_svd(path):
    d = np.load(path)
    svd = SVD(
        U=d['U'],
        # Usigma_inv=d['Usigma_inv'],
        sigma=d['sigma'],
        ab=d['ab'],
        rank=int(d['rank'][0]),
        sample_rate=d['sample_rate'][0],
        downsample_factor=int(d['downsample_factor'][0]),
    )
    assert svd.n_channels >= 1
    return svd


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


def _downsample(x, factor=DOWNSAMPLE_FACTOR):
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


def _uint8_coefs(x, margin=UINT8_MARGIN):
    # m, M = np.quantile(x, UINT8_MARGIN), np.quantile(x, 1 - UINT8_MARGIN)

    m, M = x.min(), x.max()
    d = M - m
    assert d > 0

    m -= d * margin
    M += d * margin

    a = 255 / d
    b = m
    return a, b


def to_uint8(x, ab=None):
    a, b = ab if ab is not None else _uint8_coefs(x)

    y = (x - b) * a
    # inverse: x = y / a + b

    # assert np.all((0 <= y) & (y < 256))
    overshoot = np.mean((y < 0) | (y >= 256))
    if overshoot > 0:
        logger.debug(f"uint8 casting: clipping {overshoot * 100:.3f}% of overshooting values")
    y = np.clip(y, 0, 255)

    return y.astype(np.uint8), (a, b)


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


def _get_excerpts(reader, kept_chunks=CHUNKS_EXCERPTS):
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
            total=n_chunks // skip,
            desc="extracting excerpts..."):

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


def excerpt_svd(reader, rank, kept_chunks=CHUNKS_EXCERPTS):
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

    lossy = (svd.Usigma_inv @ pp)[:rank, :]
    # lossy is (nc, ns)
    assert lossy.shape[0] < lossy.shape[1]

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

    assert rank > 0

    return (U[:, :rank] @ np.diag(sigma[:rank]) @ lossy[:rank, :])


#------------------------------------------------------------------------------
# Compressor
#------------------------------------------------------------------------------

def compress_lossy(
        path_cbin=None, cmeta=None, rank=None, max_chunks=0,
        chunks_excerpts=CHUNKS_EXCERPTS, downsampling_factor=DOWNSAMPLE_FACTOR,
        overwrite=False, dry_run=False,
        out_lossy=None, out_svd=None):

    # Check arguments.
    assert rank, "The rank must be set"
    assert path_cbin, "The raw ephys data file must be specified"

    assert downsampling_factor >= 1
    assert chunks_excerpts >= 2

    # Create a mtscomp Reader.
    reader = decompress(path_cbin, cmeta=cmeta)
    sr = int(reader.sample_rate)
    ns = reader.n_samples
    nc = reader.n_channels
    n_chunks = reader.n_chunks if max_chunks == 0 else max_chunks

    assert n_chunks > 0
    assert sr > 0
    assert ns > 0
    assert nc > 0
    assert rank <= nc, "The rank cannot exceed the number of channels"

    # Filenames.
    if out_lossy is None:
        out_lossy = Path(path_cbin).with_suffix(FILE_EXTENSION_LOSSY)
    assert out_lossy

    if out_svd is None:
        out_svd = Path(path_cbin).with_suffix(FILE_EXTENSION_SVD)
    assert out_svd

    if dry_run:
        return out_lossy

    # Compute the SVD on an excerpt of the data.
    svd = excerpt_svd(reader, rank, kept_chunks=chunks_excerpts)

    # Create a new memmapped npy file
    if not overwrite and out_lossy.exists():
        raise IOError(f"File {out_lossy} already exists.")
    shape = (n_chunks * int(reader.sample_rate) // downsampling_factor, rank)
    lossy = open_memmap(out_lossy, 'w+', dtype=np.uint8, shape=shape)

    # Compress the data.
    offset = 0
    for chunk_idx, chunk_start, chunk_length in tqdm(
            reader.iter_chunks(last_chunk=n_chunks - 1),
            desc='compressing...',
            total=n_chunks):

        # Decompress the chunk.
        raw = reader.read_chunk(chunk_idx, chunk_start, chunk_length)

        # raw is (ns, nc)
        assert raw.shape[0] > raw.shape[1]
        nsc, _ = raw.shape
        assert _ == nc

        # Compress the chunk.
        chunk_lossy = _compress_chunk(raw, svd)
        # chunk_lossy is (nc, ns)
        assert chunk_lossy.shape[0] < chunk_lossy.shape[1]

        # Write the compressed chunk to disk.
        l = chunk_lossy.shape[1]
        lossy[offset:offset + l, :], ab = to_uint8(chunk_lossy.T, svd.ab)
        # NOTE: keep the ab scaling factors for uint8 conversion only for the first chunk
        if svd.ab is None:
            svd.ab = ab
        offset += l

    # Save the SVD info to a npz file.
    svd.save(out_svd)

    return out_lossy


#------------------------------------------------------------------------------
# Decompressor
#------------------------------------------------------------------------------

class LossyReader:
    def __init__(self):
        self.path_lossy = None
        self.path_svd = None

    def open(self, path_lossy=None, path_svd=None):
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
        assert self._lossy.dtype == np.uint8

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
        self.dtype = np.uint8

        size_original = 2 * self.n_channels * self.n_samples
        self.compression = size_original / float(self.size_bytes)

    def _decompress(self, lossy, rank=None):
        lossy_float = from_uint8(lossy, self._svd.ab).T
        return _decompress_chunk(lossy_float, self._svd, rank=rank).T

    def get(self, t0, t1, rank=None):
        ds = self._svd.downsample_factor
        i0 = int(round(t0 * float(self.sample_rate) / ds))
        i1 = int(round(t1 * float(self.sample_rate) / ds))
        lossy = self._lossy[i0:i1]
        return self._decompress(lossy, rank=rank)

    def __getitem__(self, idx):
        lossy = self._lossy[idx]
        return self._decompress(lossy)


def decompress_lossy(path_lossy=None, path_svd=None):
    reader = LossyReader()
    reader.open(path_lossy, path_svd=path_svd)
    return reader
