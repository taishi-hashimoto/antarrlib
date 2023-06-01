"Noise generator/estimator."
import numpy as np
from typing import List
from numpy.random import standard_normal
from scipy.special import erfcinv
from .periodogram import _discard_extra, _split_axis


def noise(size: List[int], power: float = 1.):
    """Generates random numbers that follow normal distribution.
    The result is normalized by 1/sqrt(2).
    Used for the thermal and galactic noise in atmospheric radars.

    size:
        The size of output array.
    power:
        The mean output power.
    """
    return np.sqrt(power / 2) * (
        standard_normal(size) + 1j * standard_normal(size))


def mean_chisq(
    data: np.ndarray,
    df: int,
    nseg: int = None,
    lseg: int = None,
    axis: int = None,
):
    """Approximate the standard deviation of the Chi square distribution
    by the mean of the normal distribution when the degree of freedom is large.
    """
    if axis is None:
        data = np.ravel(data)
        return mean_chisq(data, axis=-1, df=df, nseg=nseg, lseg=lseg)
    
    nt = np.shape(data)[axis]
    if nseg is not None:
        km = nseg
        ns = nt // nseg
    elif lseg is not None:
        km = nt // lseg
        ns = lseg
    print(nt, km, ns)
    ka = ns * df
    a = np.sqrt(2) * erfcinv(2 / km)
    e = km / np.sqrt(2 * np.pi) * np.exp(-1/2 * a**2)
    cm = 1 / (1 - e / np.sqrt(ka))
    # Rearrange data
    data = _discard_extra(data, axis=axis, n=ns*km)
    new_shape, new_axis = _split_axis(data, axis, ns, 1)
    sm = np.nanmin(np.nanmean(data.reshape(new_shape), axis=new_axis), axis=axis)
    return sm * cm
