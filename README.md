# mtscomp: Multichannel time series lossless compression in Python

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


## Command-line

```bash
mtscomp data.bin data.cbin cdata.ch --sample-rate|-s 30000 --chunk-duration|-d 1 --compression-level|-l -1
mtsuncomp data.cbin data.ch data.bin
```


## High-level API

```python
def write(data, out, outmeta, sample_rate=None, chunk_duration=None, compression_level=-1):
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
    chunk_duration : float
        Length of the chunks, in seconds.
    compression_level : int
        zlib compression level.

    Returns
    -------

    cdata : NumPy-like array
        Compressed version of the data, wrapped in a NumPy-like interface.
    cmeta : dict
        A dictionary with metadata about the compression.

        version : str
            Version number of the compression format.
        sample_rate : float
            Sampling rate, in Hz.
        chunk_bounds : list of ints
            Offsets of the chunks in time samples.
        chunk_offsets : list of ints
            Offsets of the chunks within the compressed raw buffer.

    """
    return cdata, {
        'version': '1.0',
        'compression_algorithm': 'zlib',
        'compression_level': '-1',
        'sample_rate': 30000.,
        'chunk_bounds': [0, 30000, 60000, ...],
        'chunk_offsets': [0, 1234, 5678, ...],
    }


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
    return data

```


## Low-level API

```python
class Writer:
    def __init__(self, chunk_duration=1., compression_level=-1):
        pass

    def open(self, data_file, sample_rate=None):
        pass

    def write_chunk(self, chunk_idx, data):
        pass

    def write(self, out, outmeta):
        pass

    def close(self):
        pass


class Reader:
    def open(self, cdata, cmeta):
        pass

    def open_cdata(self, cdata_file):
        pass

    def open_cmeta(self, cmeta_file):
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

```

