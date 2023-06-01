"""Estimates the power spectrum density (PSD) by the periodogram."""

import numpy as np
from numpy.fft import fft, fftshift


def periodogram(signal, axis=-1, shift=True, ncoh=1, nicoh=1, extra_ok=False):
    """Estimates the power spectrum density (PSD) by the periodogram."""
    signal = coherent_average(signal, ncoh, axis, extra_ok=extra_ok)
    signal = _discard_extra(signal, axis, nicoh, extra_ok)
    shape, axis = _split_axis(signal, axis, nicoh, 0)
    spectrum = np.abs(fft(np.reshape(signal, shape),
                          axis=axis + 1)) ** 2 / shape[axis + 1]
    if shift:
        spectrum = fftshift(spectrum, axes=axis + 1)
    return spectrum.sum(axis=axis) / nicoh


def coherent_average(signal, ncoh, axis=-1, extra_ok=False):
    """Performs the coherent integration of the signal.
    'ncoh' adjacent bins along 'axis' are averaged."""
    signal = _discard_extra(signal, axis, ncoh, extra_ok=extra_ok)
    shape, axis = _split_axis(signal, axis, ncoh, 1)
    return np.reshape(signal, shape).sum(axis=axis) / ncoh


def incoherent_average(signal, nicoh, axis=-1, extra_ok=False):
    """Performs the incoherent integration of the signal.
    'nicoh' blocks along 'axis' are averaged."""
    signal = _discard_extra(signal, axis, nicoh, extra_ok=extra_ok)
    shape, axis = _split_axis(signal, axis, nicoh, 0)
    return np.reshape(signal, shape).sum(axis=axis) / nicoh


def _discard_extra(signal, axis, n, extra_ok=True):
    """Discard extra samples to divide signal by n."""
    if np.shape(signal)[axis] % n != 0:
        if not extra_ok:
            raise ValueError("signal {} cannot be divided into {}".format(np.shape(signal), n))
        indices = [slice(None) for _ in range(np.ndim(signal))]
        indices[axis] = slice(0, np.shape(signal)[axis] // n * n)
        signal = signal[tuple(indices)]
    return signal


def _split_axis(array, axis, n, which=0):
    """Returns new 'shape' that splits 'axis' of 'array' by 'n'.
    
    Returns:
        shape, axis

    shape:
        New shape.
    axis:
        Axis index."""
    shape = np.shape(array)
    if axis < 0:
        axis = len(shape) + axis
    shape1 = shape[:axis]
    shape2 = shape[axis + 1:]
    rest = shape[axis] // n
    assert rest * n == shape[axis]
    if which == 0:
        axes = n, rest
    elif which == 1:
        axes = rest, n
    else:
        raise ValueError
    new_axis = axis + which
    new_shape = shape1 + axes + shape2
    return new_shape, new_axis


def velocity_axis(n, frequency, sampling_interval, ncoh=1, nicoh=1, extend=False):
    """
    Velocity axis for Doppler radar.

    Args:
        n: The number of FFT points.
        frequency: The frequency in Hz.
        sampling_interval: The sampling interval in second.
        ncoh: The number of coherent integration. Adjacent data bins are averaged by this.
        nicoh: The number of incoherent integration. The data is decimated by this.
        extend: If true, the result also have the negative maximum frequency and its length becomes n+1.

    Returns:

    """
    faxis = frequency_axis(n, sampling_interval, ncoh=ncoh, nicoh=nicoh, extend=extend)
    vpf = 0.5 * 299792458 / frequency
    return faxis * vpf


def frequency_axis(n, sampling_interval, ncoh=1, nicoh=1, extend=False):
    n //= ncoh
    # vpf = 0.5 * 299792458 / frequency
    nf = 0.5 / sampling_interval
    c = n // 2
    ans = (np.arange(n) / c - 1) * nf
    ans = np.array(ans[::nicoh])
    if extend:
        ans.resize((n // nicoh + 1,))
        ans[-1] = -ans[0]  # -vmin is largest velocity.
    return ans
