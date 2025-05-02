"Helper methods for running simulation of antenna array processing."

import numpy as np
from .noise import noise

__all__ = ["point_source", "noise"]


def point_source(k, r, xyz, rx_power: float = 1.):
    """Simulate echoes from a point source.
    
    Parameters
    ==========
    k: ndarray
        Wave numbers [M]
    r: ndarray
        Antenna positions [N, 3]
    xyz: ndarray
        Target's location in 3-d space [T, 3]
    rx_power: float
        Combined received power of the first sample.

    Returns
    =======
    signal: ndarray
        Received power [M, N]
    """
    k = np.ravel(k)
    r = np.reshape(r, (-1, 3))
    xyz = np.reshape(xyz, (-1, 3))
    tx_distance = np.linalg.norm(xyz - np.linalg.norm(r, axis=0), axis=-1)
    rx_distance = np.linalg.norm(r[:, None, :] - xyz[None, :, :], axis=-1)
    distance = tx_distance + rx_distance  # [M, N, T]
    phase = np.exp(-1j * k[:, None, None] * distance)
    power = rx_power * (distance[0:1] / distance)**2 / (len(k) * len(r) * len(xyz))
    return np.sqrt(power) * phase
