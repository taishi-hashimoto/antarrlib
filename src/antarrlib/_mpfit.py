
# %% jax version
from typing import Any
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxopt import GaussNewton


def silverman_bandwidth(x):
    """Silverman's rule of thumb for bandwidth selection."""
    std = jnp.std(x)
    n = x.shape[0]
    return 1.06 * std * n**(-1/5)


def gaussian_kernel(u):
    """Standard Gaussian kernel."""
    return jnp.exp(-0.5 * u**2) / jnp.sqrt(2 * jnp.pi)


def kde_jax(x, points=None, bandwidth=None):
    """Evaluate KDE at points based on sample x."""
    if points is None:
        points = x
    if bandwidth is None:
        bandwidth = silverman_bandwidth(x)
    
    def single_eval(xi):
        u = (xi - x) / bandwidth
        return jnp.mean(gaussian_kernel(u)) / bandwidth

    return jax.vmap(single_eval)(points)


def mp_pdf(lambdas: ArrayLike, sigma2, c):
    lambda_min = sigma2 * (1 - jnp.sqrt(c)) ** 2
    lambda_max = sigma2 * (1 + jnp.sqrt(c)) ** 2

    # avoid division by zero
    lambdas = jnp.clip(lambdas, 1e-12, None)

    diff1 = lambda_max - lambdas
    diff2 = lambdas - lambda_min
    inside_sqrt = jnp.clip(diff1 * diff2, a_min=0.0)  # clip negative to zero

    density = jnp.sqrt(inside_sqrt) / (2 * jnp.pi * sigma2 * c * lambdas)
    # outside support: 0
    density = jnp.where((lambdas < lambda_min) | (lambdas > lambda_max), 0.0, density)
    return density


def fit_noise_mp(eigvals: ArrayLike, T: int, npoints: int = 100, maxiter: int = 50) -> tuple[float, Any]:
    """Fit eigenvalues on the Marchenko-Pastur distribution to obtain the noise variance.
    
    Parameters
    ==========
    eigvals: array-like
        Eigenvalues of the correlation matrix.
    T: int
        Number of snapshots
    maxiter: int
        Maximum number of iterations for the optimization.
        
    Returns
    =======
    sigma2: float
        Estimated noise variance.
    state: Any
        State of the Gauss-Newton solver.
    """
    eigvals = jnp.atleast_2d(eigvals)
    sigma2, state = jax.vmap(_fit_mp, in_axes=(0, None, None, None))(eigvals, T, npoints, maxiter)
    return sigma2.ravel(), state


def _fit_mp(eigvals: jnp.ndarray, T: int, npoints: int, maxiter: int):
    eigvals = jnp.clip(eigvals, 1e-12, None)
    c = eigvals.shape[-1] / T

    xmin, xmax = jnp.min(eigvals), jnp.max(eigvals)
    centers = jnp.linspace(xmin, xmax, npoints)
    kde_ref = kde_jax(eigvals, centers)
    
    def residual_fun(sigma2):
        pdf_mp = mp_pdf(centers, sigma2, c)
        return kde_ref - jnp.where(c > 1, pdf_mp / (1.0 - 1.0 / c), pdf_mp)

    solver = GaussNewton(
        residual_fun,
        maxiter=maxiter,
    )
    result = solver.run(jnp.array([jnp.median(eigvals)]))
    return result.params, result.state
