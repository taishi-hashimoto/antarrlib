"Helper methods for running simulation of antenna array processing."

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from .noise import noise as _noise

__all__ = ["noise"]


def noise(size: list[int], power: float = 1.):
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


def _spread_radial(v: NDArray[np.float64], sigma: float, n_points: int, rng: np.random.Generator):
    """Spread radial vector `v` by Gaussian random offset for zenith angle.
    
    Parameters
    ==========
    v: ndarray
        The radial vector.
    sigma: float
        The standard deviation of the Gaussian random offset for zenith angle in radians.
    n_points: int
        The number of points to generate.
    rng: np.random.Generator
        The random number generator to use.

    Notes
    =====
    The actual angular spread is not following the exact Gaussian distribution with `sigma` due to the nature of spherical coordinates.
    Use `spread_radial` for automatic calculation of `sigma` from the desired angular spread.
    """
    v = v / np.linalg.norm(v)
    if np.allclose(v, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(v, [0, 0, 1])
        u /= np.linalg.norm(u)
    w = np.cross(v, u)

    theta = rng.normal(loc=0, scale=sigma, size=n_points)
    phi = rng.uniform(0, 2 * np.pi, size=n_points)

    v_part = np.outer(np.cos(theta), v)
    u_part = np.outer(np.sin(theta) * np.cos(phi), u)
    w_part = np.outer(np.sin(theta) * np.sin(phi), w)

    directions = v_part + u_part + w_part
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    return directions


def spread_power(
    r0: float,
    v0: NDArray[np.float64],
    r: NDArray[np.float64],
    v: NDArray[np.float64],
    angular_spread: float,
    range_spread: float,
) -> NDArray[np.float64]:
    """Calculate the power spread for given angular and range spread."""
    angles = angle_between(v0, v)
    dr = r - r0
    power = np.exp(-0.5 * ((angles / angular_spread)**2 + (dr / range_spread)**2))
    return power


def spread_radial(
    v: NDArray[np.float64],
    angular_spread: float,
    n_points: int | None = None,
    rng: np.random.Generator | None = None,
    angular_resolution: float | None = None,
):
    """Spread radial vector `v` by Gaussian random offset for zenith angle.
    Parameters
    ==========
    v: ndarray
        The radial vector.
    angular_spread: float
        The desired angular spread in degrees or radians.
    n_points: int, optional
        The number of points to generate. If not specified, it is calculated from `angular_spread` and `angular_resolution`.
    rng: np.random.Generator, optional
        The random number generator to use. If not specified, a default random generator is used.
    angular_resolution: float, optional
        The angular resolution in degrees or radians. If not specified, it is calculated as `angular_spread / 100`.
    """
    if angular_resolution is None:
        angular_resolution = angular_spread / 100
    if n_points is None:
        n_points = int(6 * angular_spread / angular_resolution)
    if rng is None:
        rng = np.random.default_rng()

    sigma = find_sigma_from_angular_spread(v, angular_spread, n_points=n_points, rng=rng)

    return _spread_radial(v, sigma, n_points, rng)


def angle_between(v0, directions):
    v0 = v0 / np.linalg.norm(v0)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    dot_products = np.clip(np.dot(directions, v0), -1.0, 1.0)
    angles = np.arccos(dot_products)
    return angles


def find_sigma_from_angular_spread(v0, angular_spread, n_points=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    def objective(sigma):
        dirs = _spread_radial(v0, sigma, n_points, rng=rng)
        angles = angle_between(v0, dirs)
        return (np.std(angles) - angular_spread) ** 2

    res = minimize_scalar(objective, bounds=(0, np.pi / 2), method='bounded')
    return res.x


def _spread_range(
    r: float,
    sigma: float,
    n_points: int,
    rng: np.random.Generator,
):
    """Spread range vector `r` by Gaussian random offset for range.
    
    Parameters
    ==========
    r: ndarray
        The range vector.
    sigma: float
        The standard deviation of the Gaussian random offset for range in meters.
    n_points: int
        The number of points to generate.
    rng: np.random.Generator
        The random number generator to use.
    
    Returns
    =======
    directions: ndarray
        The spread range vectors.
    """
    return rng.normal(loc=r, scale=sigma, size=n_points)


def spread_range(
    r: float,
    range_spread: float,
    n_points: int | None = None,
    range_resolution: float | None = None,
    rng: np.random.Generator | None = None,
):
    """Spread range vector `r` by Gaussian random offset for range.
    
    Parameters
    ==========
    r: float
        The range vector.
    range_spread: float
        The desired range spread in meters.
    n_points: int, optional
        The number of points to generate. If not specified, it is calculated from `range_spread`.
    rng: np.random.Generator, optional
        The random number generator to use. If not specified, a default random generator is used.
    
    Returns
    =======
    directions: ndarray
        The spread range vectors.
    """
    if range_resolution is None:
        range_resolution = range_spread / 100
    if n_points is None:
        n_points = int(6 * range_spread / range_resolution)
    if rng is None:
        rng = np.random.default_rng()

    return _spread_range(r, range_spread, n_points, rng)
