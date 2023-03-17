"Spherical surface integral."
import numpy as np
from typing import List


def patch_area(ze: List[float], az: List[float], ra: float = 1.):
    """Approximate area of each small patch on the spherical polar coordinates.
    
    `np.sum(patch_area(...))` will give a surface area of given part of sphere.

    Parameters
    ==========
    ze: 1-d array of float
        The zenith angles.
    az: 1-d array of float
        The azimuth angles.
    ra: float
        The radius. Default is 1.

    Returns
    =======
    area:
        Area of each small patch.
    """
    ze = np.ravel(ze)
    az = np.ravel(az)
    # Zenith steps for each zenith bin.
    dz = np.abs(ze[:-1] - ze[1:])
    dz = np.r_[dz / 2, [0]] + np.r_[[0], dz / 2]
    # Azimuth steps for each azimuth bin.
    da = np.abs(az[:-1] - az[1:])
    da = np.r_[da / 2, [0]] + np.r_[[0], da / 2]
    # Special treatment on singular points.
    sin_ze = np.sin(ze)
    si = np.logical_or(sin_ze == 0, sin_ze == np.pi)
    # Here, patches are not rectangular, but triangular.
    sin_ze[si] = np.sin(dz[si] / 2) / 2
    # Area of each patch.
    return np.reshape(dz * sin_ze, (-1, 1)) * da * ra ** 2
