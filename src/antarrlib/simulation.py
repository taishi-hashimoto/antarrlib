"Helper methods for running simulation of antenna array processing."

import numpy as np
from .noise import noise as _noise

__all__ = ["noise"]


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
