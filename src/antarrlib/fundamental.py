"Compact library for antenna array."

import numpy as np
from typing import List, Tuple


SPEED_OF_LIGHT = 299792458.
"Speed of light in m/s ~ 3e8 m/s."


def freq2wlen(frequency: float) -> float:
    "Compute wave length from frequency."
    return SPEED_OF_LIGHT / frequency


def wlen2wnum(wavelength: float) -> float:
    "Compute wave number from wave length."
    return 2. * np.pi / wavelength


def freq2wnum(frequency: float) -> float:
    "Compute wave number from frequency."
    return wlen2wnum(freq2wlen(frequency))


def steering_vector(
    k: float,
    r: List[Tuple[float, float, float]],
    v: List[Tuple[float, float, float]],
    s: int = 1
) -> np.ndarray:
    """Steering vector of an antenna array.

    The result is not normalized.

    Parameters
    ==========
    k: float
        The wave number.
    r: M by 3 float ndarray
        The antenna location [M, 3], where M is the number of antennas.
    v: N by 3 float ndarray
        Radial vectors [N, 3], where N is the number of directions.
    s: int
        +1 for Tx, -1 for Rx.

    Returns
    =======
    a: N by M complex ndarray
        Steering vector.
    """
    return np.exp(s * 1j * k * np.dot(v, np.transpose(r)))


def radial(ze: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Radial unit vector to the specified direction.

    Angles are in radian.
    Azimuth angle is measured counterclockwise from East.

    Parameters
    ==========
    ze: N by 3 float ndarray
        Zenith angles.
    az: N by 3 float ndarray
        Azimuth angles.
    
    Returns
    =======
    v: N by 3 float ndarray
        Radial (look-direction) vector to the specified direction.
    """
    ze = np.reshape(ze, (-1, 1))
    az = np.reshape(az, (-1, 1))
    return np.column_stack((
        np.sin(ze) * np.cos(az),
        np.sin(ze) * np.sin(az),
        np.cos(ze)))
