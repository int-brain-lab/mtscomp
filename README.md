# Multichannel time series lossless compression in Python

[![Build Status](https://travis-ci.org/int-brain-lab/mtscomp.svg?branch=master)](https://travis-ci.org/int-brain-lab/mtscomp)

Lossless compression for time-dependent signals with high sampling rate (tens of thousands of Hz) and high dimensionality (hundreds or thousands of channels). Developed for large-scale ephys neuro recordings (e.g. Neuropixels).


## Requested features

* Lossless
* Pure Python
* As simple as possible
* Scale well to large sampling rate and dimension
* Can be uncompressed on the fly quickly (random access)
* Amenable to multithreading


## Process

* Input data is a `(n_samples, n_channels)` array.
* Split it in the time domain into e.g. 1-second chunks.
* For every chunk, keep the first value on every channel.
* Compute the channel-wise time difference, i.e. `x[i + 1, :] - x[i, :]`.
* Compress that with a lossless compression algorithm (e.g. gzip).
* Save the offsets of the chunks within the compressed file in a separate metadata file.


## File specification

* `data.cbin`: compressed data file. Binary concatenation of compressed chunk raw binary data. Every chunk is binary zlib-compressed data of the corresponding data chunk.
* `data.ch`: JSON file with metadata about the compression, including the chunk offsets used for on-the-fly reading.


## Dependencies

* Python 3.7+
* NumPy

For development only:

* flake8
* pytest
* pytest-cov
* coverage


## Command-line

```bash
mtscomp data.bin data.cbin cdata.ch --sample-rate|-s 30000 --dtype|-d uint8 --chunk-duration|-d 1 --compression-level|-l -1
mtsuncomp data.cbin data.ch data.bin
```


## High-level API

```python

```


## Low-level API

```python
w = Writer(chunk_duration=1., compression_algorithm=None, compression_level=-1)
w.open(path, sample_rate=sample_rate, n_channels=n_channels, dtype=dtype)
w.write(out, outmeta)
w.close()
```
