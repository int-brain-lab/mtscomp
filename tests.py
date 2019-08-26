# -*- coding: utf-8 -*-

"""mtscomp tests."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
from pytest import fixture

from mtscomp import add_default_handler, Writer


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

n_channels = 385
sample_rate = 30000.
duration = .1
dtype = np.int16
normal_std = .25


add_default_handler('DEBUG')


@fixture
def path(tmp_path):
    time = np.arange(0, duration, 1 / sample_rate)
    n_samples = len(time)
    arr = np.sin(10 * time)[:, np.newaxis]
    arr = arr + np.random.normal(loc=0, scale=normal_std, size=(n_samples, n_channels))
    path = Path(tmp_path) / 'data.bin'
    m, M = arr.min(), arr.max()
    arrn = (arr - m) / (M - m)  # normalize to [0, 1]
    arr16 = (arrn * 32767).astype(np.int16)
    with open(path, 'wb') as f:
        arr16.tofile(f)
    return path


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_1(path):
    # path = Path('data/imec_385_10s.bin')
    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'

    w = Writer()
    w.open(path, sample_rate=sample_rate, n_channels=n_channels, dtype=dtype)
    w.write(out, outmeta)
    w.close()
