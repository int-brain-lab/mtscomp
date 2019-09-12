# Multichannel time series lossless compression in Python

[![Build Status](https://travis-ci.org/int-brain-lab/mtscomp.svg?branch=master)](https://travis-ci.org/int-brain-lab/mtscomp)
[![Coverage Status](https://codecov.io/gh/int-brain-lab/mtscomp/branch/master/graph/badge.svg)](https://codecov.io/gh/int-brain-lab/mtscomp)

**Warning: This library is experimental. Do not use it in production yet as the file format is still evolving.**

This library implements a simple lossless compression scheme adapted to time-dependent high-frequency, high-dimensional signals. It is being developed within the [International Brain Laboratory](https://www.internationalbrainlab.com/) with the aim of being the compression library used for all large-scale electrophysiological recordings based on Neuropixels. The signals are typically recorded at 30 kHz and 10 bit depth, and contain several hundreds of channels.


## Compression scheme

The requested features for the compression scheme were as follows:

* Lossless compression only (one should retrieve byte-to-byte exact decompressed data).
* Written in pure Python (no C extensions) with minimal dependencies so as to simplify distribution.
* Scalable to large sample rates, large number of channels, long recording time.
* Faster than real time (i.e. it should take less time to compress than to record).
* Multithreaded so as to leverage multiple CPU cores.
* On-the-fly decompression and random read accesses.
* As simple as possible.

The compression scheme is the following:

* The data is split into chunks along the time axis.
* The time differences are computed for all channels.
* These time differences are compressed with zlib.
* The compressed chunks are appended in a binary file.
* Metadata about the compression, including the chunk offsets within the compressed binary file, are saved in a secondary JSON file.

Saving the offsets allows for on-the-fly decompression and random data access: one simply has to determine which chunks should be loaded, and load them directly from the compressed binary file. The compressed chunks are decompressed with zlib, and the original data is recovered with a cumulative sum (the inverse of the time difference operation).

With large-scale neurophysiological recordings, a compression ration of 3x could be obtained.


## Dependencies

* Python 3.7+
* NumPy
* tqdm [for the progress bar]

For development only:

* flake8
* pytest
* pytest-cov
* coverage


## Installation

```
pip install git+https://github.com/int-brain-lab/mtscomp.git
```


## Command-line

```bash
# Compression: specify the number of channels, sample rate, dtype
mtscomp data.bin data.cbin cdata.ch -n 385 -s 30000 -d uint16
# Decompression
mtsdecomp data.cbin data.ch data_dec.bin
```


## High-level API

```python
import numpy as np
from mtscomp.mtscomp import compress, decompress

# Compress a .bin file into a pair .cbin (compressed binary file) and .ch (JSON file).
compress('data.bin', 'data.cbin', 'data.ch', sample_rate=20000., n_channels=256, dtype=np.uint16)
# Decompress a pair (.cbin, .ch) and return an object that can be sliced like a NumPy array.
arr = decompress('data.cbin', 'data.ch')
X = arr[start:end, :]  # decompress the data on the fly directly from the file on disk
```


## Low-level API

```python
import numpy as np
from mtscomp.mtscomp import Writer, Reader

# Define a writer to compress a flat raw binary file.
w = Writer(chunk_duration=1.)
# Open the file to compress.
w.open('data.bin', sample_rate=20000., n_channels=256, dtype=np.uint16)
# Compress it into a compressed binary file, and a JSON header file.
w.write('data.cbin', 'data.ch')
w.close()

# Define a reader to decompress a compressed array.
r = Reader()
# Open the compressed dataset.
r.open('data.cbin', 'data.ch')
# The reader can be sliced as a NumPy array: decompression happens on the fly. Only chunks
# that need to be loaded are loaded and decompressed.
# Here, we load everything in memory.
array = r[:]
# Or we can decompress into a new raw binary file on disk.
r.tofile('data_dec.bin')
r.close()
```


## Implementation details

* **Multithreading**: since Python's zlib releases the GIL, the library uses multiple threads when compressing a file. The chunks are grouped in batches containing as many chunks as threads. After each batch, the chunks are written in the binary file in the right order (since the threads of the batch have no reason to finish in order).


## Performance

Preliminary benchmarks on an Neuropixels dataset (30 kHz, 385 channels) and Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz:

* Compression ratio: -63% (compressed files are nearly 3x smaller)
* Compression write time (single-threaded): 10 MB/s, 2x slower than real time
* Compression write time (8 threads): 3-5x faster than single-threaded
* Compression read time (single-threaded): 24 MB/s, 30x slower than uncompressed, 3x faster than real time
