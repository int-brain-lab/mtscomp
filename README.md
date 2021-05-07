# Multichannel time series lossless compression in Python

[![Build Status](https://travis-ci.org/int-brain-lab/mtscomp.svg?branch=master)](https://travis-ci.org/int-brain-lab/mtscomp)
[![Coverage Status](https://codecov.io/gh/int-brain-lab/mtscomp/branch/master/graph/badge.svg)](https://codecov.io/gh/int-brain-lab/mtscomp)

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
* The compressed chunks (and initial values of each chunk) are appended in a binary file.
* Metadata about the compression, including the chunk offsets within the compressed binary file, are saved in a secondary JSON file.

Saving the offsets allows for on-the-fly decompression and random data access: one simply has to determine which chunks should be loaded, and load them directly from the compressed binary file. The compressed chunks are decompressed with zlib, and the original data is recovered with a cumulative sum (the inverse of the time difference operation).

With large-scale neurophysiological recordings, we achieved a compression ratio of 3x.

As a consistency check, the compressed file is by default automatically and transparently decompressed and compared to the original file on a byte-per-byte basis.


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
pip install mtscomp
```


## Command-line interface

Example:

```bash
# Compression: specify the number of channels, sample rate, dtype, optionally save the parameters
# as default in ~/.mtscomp with --set-default
mtscomp data.bin -n 385 -s 30000 -d int16 [--set-default]
# Decompression
mtsdecomp data.cbin -o data.decomp.bin
```

Usage:

```
usage: mtscomp [-h] [-d DTYPE] [-s SAMPLE_RATE] [-n N_CHANNELS] [-p CPUS]
               [-c CHUNK] [-nc] [-v] [--set-default]
               path [out] [outmeta]

Compress a raw binary file.

positional arguments:
  path                  input path of a raw binary file
  out                   output path of the compressed binary file (.cbin)
  outmeta               output path of the compression metadata JSON file
                        (.ch)

optional arguments:
  -h, --help            show this help message and exit
  -d DTYPE, --dtype DTYPE
                        data type
  -s SAMPLE_RATE, --sample-rate SAMPLE_RATE
                        sample rate
  -n N_CHANNELS, --n-channels N_CHANNELS
                        number of channels
  -p CPUS, --cpus CPUS  number of CPUs to use
  -c CHUNK, --chunk CHUNK
                        chunk duration
  -nc, --no-check       no check
  -v, --debug           verbose
  --set-default         set the specified parameters as the default



usage: mtsdecomp [-h] [-o [OUT]] [--overwrite] [-nc] [-v] cdata [cmeta]

Decompress a raw binary file.

positional arguments:
  cdata                 path to the input compressed binary file (.cbin)
  cmeta                 path to the input compression metadata JSON file (.ch)

optional arguments:
  -h, --help            show this help message and exit
  -o [OUT], --out [OUT]
                        path to the output decompressed file (.bin)
  --overwrite, -f       overwrite existing output
  -nc, --no-check       no check
  -v, --debug           verbose
```


## High-level API

Example:

```python
import numpy as np
from mtscomp.mtscomp import compress, decompress

# Compress a .bin file into a pair .cbin (compressed binary file) and .ch (JSON file).
compress('data.bin', 'data.cbin', 'data.ch', sample_rate=20000., n_channels=256, dtype=np.int16)
# Decompress a pair (.cbin, .ch) and return an object that can be sliced like a NumPy array.
arr = decompress('data.cbin', 'data.ch')
X = arr[start:end, :]  # decompress the data on the fly directly from the file on disk
arr.close()  # Close the file when done
```


## Low-level API

Example:

```python
import numpy as np
from mtscomp import Writer, Reader

# Define a writer to compress a flat raw binary file.
w = Writer(chunk_duration=1.)
# Open the file to compress.
w.open('data.bin', sample_rate=20000., n_channels=256, dtype=np.int16)
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

Performance on an Neuropixels dataset (30 kHz, 385 channels) and Intel 10-core i9-9820X CPU @ 3.3GHz:

* Compression ratio: -63% (compressed files are nearly 3x smaller)
* Compression time (20 threads): 88 MB/s, **4x faster than real time**
* Decompression time (single-threaded at the moment): 22 MB/s, **3x faster than real time**
