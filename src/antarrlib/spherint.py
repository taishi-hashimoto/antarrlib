"Spherical surface integral."

from typing import List
from dataclasses import dataclass
from math import isclose
import numpy as np
from scipy.spatial import ConvexHull


def patch_area(ze: List[float], az: List[float], ra: float = 1.0) -> np.ndarray:
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
    return np.reshape(dz * sin_ze, (-1, 1)) * da * ra**2


# New implementations below.


def _ensure_1d_sorted(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if np.any(np.diff(x) <= 0):
        raise ValueError("Input coordinates must be strictly increasing.")
    return x


def trapezoid_weights_nonuniform(x: np.ndarray) -> np.ndarray:
    """Trapezoid weights for nonuniform x."""
    x = _ensure_1d_sorted(x)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 points")
    w = np.empty_like(x)
    dx = np.diff(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if n > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def periodic_trapezoid_weights_nonuniform(x: np.ndarray, period: float) -> np.ndarray:
    """Trapezoid weights for periodic x."""
    x = _ensure_1d_sorted(x)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 points")
    diffs = np.diff(x)
    wrap_gap = period - (x[-1] - x[0])
    total_span = np.sum(diffs) + wrap_gap
    if not np.isclose(total_span, period, rtol=1e-10, atol=1e-12):
        raise ValueError("x does not span a full period for periodic weighting.")
    gap_fwd = np.concatenate([diffs, [wrap_gap]])
    gap_bwd = np.concatenate([[wrap_gap], diffs])
    w = 0.5 * (gap_fwd + gap_bwd)
    return w


@dataclass
class GridSphericalIntegrator:
    """Numerical integration on the unit sphere for irregular grids in (theta, phi)."""

    theta: np.ndarray
    phi: np.ndarray
    periodic_phi: bool = True
    period_phi: float = 2 * np.pi

    def __post_init__(self):
        self.theta = _ensure_1d_sorted(np.asarray(self.theta, float))
        self.phi = _ensure_1d_sorted(np.asarray(self.phi, float))
        if self.theta[0] < 0 - 1e-12 or self.theta[-1] > np.pi + 1e-12:
            raise ValueError("theta should lie within [0, pi].")
        if self.periodic_phi:
            self.w_phi = periodic_trapezoid_weights_nonuniform(
                self.phi, self.period_phi
            )
        else:
            self.w_phi = trapezoid_weights_nonuniform(self.phi)
        self.w_theta = trapezoid_weights_nonuniform(self.theta)
        self.sin_theta = np.sin(self.theta)

    def integrate(self, values: np.ndarray) -> float:
        """Integrate values on the spherical grid."""
        f = np.asarray(values, dtype=float)
        if f.shape != (self.theta.size, self.phi.size):
            raise ValueError(
                f"values must have shape {(self.theta.size, self.phi.size)}"
            )
        g = f @ self.w_phi
        integral = np.dot(self.w_theta, g * self.sin_theta)
        return float(integral)

    def average(self, values: np.ndarray) -> float:
        """Average values on the spherical grid."""
        return self.integrate(values) / (4 * np.pi)


def _to_unit_xyz(theta, phi):
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = st * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def spherical_triangle_area(u, v, w):
    """Area of a spherical triangle on the unit sphere given by vertices u, v, w."""

    def ang(a, b):
        dot = np.clip((a * b).sum(-1), -1.0, 1.0)
        return np.arccos(dot)

    a = ang(v, w)
    b = ang(w, u)
    c = ang(u, v)
    s = 0.5 * (a + b + c)
    tan_e4 = (
        np.tan(0.5 * s)
        * np.tan(0.5 * (s - a))
        * np.tan(0.5 * (s - b))
        * np.tan(0.5 * (s - c))
    )
    tan_e4 = np.maximum(tan_e4, 0.0)
    e = 4.0 * np.arctan(np.sqrt(tan_e4))
    return e


class TriGridSphericalIntegrator:
    """Numerical integration on the unit sphere for scattered points via spherical triangulation."""

    def __init__(self, theta, phi):
        theta = np.asarray(theta, float).ravel()
        phi = np.asarray(phi, float).ravel()
        if theta.size != phi.size:
            raise ValueError("theta and phi must have the same size")
        if np.any(theta < -1e-12) or np.any(theta > np.pi + 1e-12):
            raise ValueError("theta must be within [0, pi].")

        self.theta = theta
        self.phi = phi
        self.xyz = _to_unit_xyz(theta, phi)

        hull = ConvexHull(self.xyz, qhull_options="QJ")
        self.triangles = hull.simplices

        self._areas = self._compute_triangle_areas(self.triangles)
        self._vertex_areas = self._accumulate_vertex_areas(
            self.triangles, self._areas, theta.size
        )

    def _compute_triangle_areas(self, triangles):
        u = self.xyz[triangles[:, 0]]
        v = self.xyz[triangles[:, 1]]
        w = self.xyz[triangles[:, 2]]
        a = spherical_triangle_area(u, v, w)
        a = np.where(np.isfinite(a) & (a > 0), a, 0.0)
        return a

    def _accumulate_vertex_areas(self, triangles, tri_areas, npoints):
        va = np.zeros(npoints, dtype=float)
        share = (tri_areas / 3.0)[:, None]
        np.add.at(va, triangles.reshape(-1), np.repeat(share, 3, axis=1).ravel())
        return va

    @property
    def num_triangles(self):
        """Number of triangles."""
        return self.triangles.shape[0]

    @property
    def vertex_area_weights(self):
        """Area weights associated with each vertex."""
        return self._vertex_areas.copy()

    def integrate(self, values):
        """Integrate values weighted by vertex areas."""
        f = np.asarray(values, float).ravel()
        if f.size != self.theta.size:
            raise ValueError("values must have the same length as theta/phi")
        return float(np.dot(f, self._vertex_areas))

    def average(self, values):
        """Average values at the scattered points."""
        return self.integrate(values) / (4 * np.pi)


def random_scattered(n, rng=None):
    """Generate n random scattered points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    mu = rng.uniform(-1, 1, size=n)
    theta = np.arccos(mu)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    return theta, phi


def test_grid(n_theta=361, n_phi=721):
    """Test GridSphericalIntegrator with random irregular grids."""
    theta = np.sort(np.random.default_rng(0).random(n_theta)) * np.pi
    phi = np.sort(np.random.default_rng(1).random(n_phi) * 2 * np.pi)

    integrator = GridSphericalIntegrator(theta, phi)

    f1 = np.ones((n_theta, 1)) * np.ones((1, n_phi))
    f2 = np.cos(theta)[:, None] * np.ones((1, n_phi))
    f3 = (np.cos(theta) ** 2)[:, None] * np.ones((1, n_phi))

    i1 = integrator.integrate(f1)
    i2 = integrator.integrate(f2)
    i3 = integrator.integrate(f3)

    print("GridSphericalIntegrator results:")
    print(f"  I[1]     = {i1:.10f} (expected {4*np.pi:.10f})")
    print(f"  I[cosθ]  = {i2:.10f} (expected 0.0)")
    print(f"  I[cos²θ] = {i3:.10f} (expected {4*np.pi/3:.10f})")

    assert isclose(i1, 4 * np.pi, rel_tol=1e-3)
    assert isclose(i2, 0.0, abs_tol=1e-3)
    assert isclose(i3, 4 * np.pi / 3, rel_tol=1e-3)


def test_triang(n=2000, rng=None):
    """Test TriSphericalIntegrator with random scattered points."""
    th, ph = random_scattered(n, rng)
    tri = TriGridSphericalIntegrator(th, ph)
    f1 = np.ones(n)
    f2 = np.cos(th)
    f3 = np.cos(th) ** 2

    i1 = tri.integrate(f1)
    i2 = tri.integrate(f2)
    i3 = tri.integrate(f3)

    assert isclose(i1, 4 * np.pi, rel_tol=1e-3)
    assert isclose(i2, 0.0, abs_tol=1e-3)
    assert isclose(i3, 4 * np.pi / 3, rel_tol=1e-3)

    return {
        "ntri": tri.num_triangles,
        "sum_vertex_areas": float(tri.vertex_area_weights.sum()),
        "I[1]": float(i1),
        "I[cosθ]": float(i2),
        "I[cos²θ]": float(i3),
        "4π": float(4 * np.pi),
        "4π/3": float(4 * np.pi / 3),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(2025)
    results = test_triang(1500, rng)
    print(results)
    test_grid()
