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


## High-level API

```python
def compress(data, cdata_file=None, cmeta_file=None, sample_rate=None, chunk_duration=None):
    """Compress a NumPy-like array (may be memmapped).

    Parameters
    ----------

    data : NumPy-like array
        An array with shape `(n_samples, n_channels)`.
    cdata_file : str or file handle
        Output file for the compressed data (if file handle, should be open in `a` mode).
    cmeta_file : str (path to the metadata file) or file handle (open in `w` mode)
        JSON file with a dictionary with metadata about the compression (see doc of `compress()`).
    sample_rate : float
        Sampling rate, in Hz.
    chunk_duration : float
        Length of the chunks, in seconds.

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
        'sample_rate': 30000.,
        'chunk_bounds': [0, 30000, 60000, ...],
        'chunk_offsets': [0, 1234, 5678, ...],
    }


def uncompress(cdata_file, cmeta):
    """Uncompress a compressed file and return a NumPy-like array (memmapping the compressed data).

    Parameters
    ----------

    cfile : str or file handle
        File with the compressed data (if file handle, should be open in `a` mode).
    cmeta : dict or str (path to the metadata file) or file handle
        A dictionary with metadata about the compression (see doc of `compress()`).

    Returns
    -------

    cdata : NumPy-like array
        Compressed version of the data, wrapped in a NumPy-like interface.

    """
    return cdata

```

