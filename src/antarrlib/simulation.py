"Helper methods for running simulation of antenna array processing."

import numpy as np
from .noise import noise as _noise

__all__ = ["point_source", "noise"]


def point_source(k, r, xyz, power: float = 1.):
    """Simulate echoes from a point source.
    
    Parameters
    ==========
    k: ndarray
        Wave numbers [M]
    r: ndarray
        Antenna positions [N, 3]
    xyz: ndarray
        Target's location in 3-d space [T, 3]
    power: float
        Combined received power of the first sample.

    Returns
    =======
    signal: ndarray
        Received complex signal [M, N, T].
        Power is normalized such that the combined signal from all antennas for
        each frequency becomes the specified value.
    """
    k = np.ravel(k)
    r = np.reshape(r, (-1, 3))
    xyz = np.reshape(xyz, (-1, 3))
    tx_distance = np.linalg.norm(xyz - np.linalg.norm(r, axis=0), axis=-1)
    rx_distance = np.linalg.norm(r[:, None, :] - xyz[None, :, :], axis=-1)
    distance = tx_distance + rx_distance  # [M, N, T]
    phase = np.exp(1j * k[:, None, None] * distance)
    power = power * (distance[0:1] / distance)**2 / len(r)
    return np.sqrt(power) * phase


def noise(size: tuple[int, int, int], power: float = 1.):
    """Generates random numbers that follow normal distribution.
    The result is normalized by 1/sqrt(2).
    Used for the thermal and galactic noise in atmospheric radars.

    Parameters
    ==========
    size:
        The size of output array.
        The last dimension is the number of antennas.
    power:
        The mean output power.
    
    Returns
    =======
    noise: ndarray
        The generated noise.
        The result is normalized such that the combined signal from all antennas
        for each frequency becomes the specified value.
    """
    return _noise(size, power) / np.sqrt(size[-1])
