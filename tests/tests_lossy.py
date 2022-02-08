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

from mtscomp import Reader, decompress, add_default_handler, lossy as ml

logger = logging.getLogger(__name__)
add_default_handler('DEBUG')


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

n_channels = 19
sample_rate = 1234.
duration = 5.67
normal_std = .25
time = np.arange(0, duration, 1. / sample_rate)
n_samples = len(time)


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


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def show_compare(lossless, lossy, t0, t1):
    assert isinstance(lossless, Reader)
    assert isinstance(lossy, ml.LossyReader)

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.grid"] = False
    sns.set_theme(style="white")

    nrows = 2
    fix, axs = plt.subplots(nrows, 1, sharex=True)

    sr = lossless.sample_rate
    i0 = int(round(t0 * sr))
    i1 = int(round(t1 * sr))

    axs[0].imshow(ml._preprocess(lossless[i0:i1]), cmap="gray", aspect="auto")
    axs[0].set_title(f"original")

    axs[1].imshow(lossy.get(t0, t1).T, cmap="gray", aspect="auto")
    axs[1].set_title(f"rank={lossy.rank}, compression={lossy.compression:.1f}x")

    plt.show()


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_lossy():

    EPHYS_DIR = Path("/home/cyrille/ephys/globus/KS023/")
    path_cbin = EPHYS_DIR / "raw.cbin"

    rank = 40
    max_chunks = 10

    out_lossy = ml.compress_lossy(
        path_cbin=path_cbin,
        chunks_excerpts=3,
        rank=rank,
        max_chunks=max_chunks,
        overwrite=True,
        dry_run=True,
    )

    lossless = decompress(path_cbin)
    lossy = ml.decompress_lossy(out_lossy)

    show_compare(lossless, lossy, 0, .2)

    # chunks_excerpts = 3
    # svd = excerpt_svd(reader, rank, chunks_excerpts)

    # nc = raw.shape[1]
    # compression = DOWNSAMPLE_FACTOR * 2 * nc / float(rank)

    # lossy = _compress_chunk(raw, svd)
    # reconst = _decompress_chunk(lossy, svd, rank=rank)

    # lossy8, ab = to_uint8(lossy)
    # lossy_ = from_uint8(lossy8, ab)
    # reconst8 = _decompress_chunk(lossy_, svd, rank=rank)
