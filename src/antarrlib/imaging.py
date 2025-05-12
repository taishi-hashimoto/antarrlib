"Imaging library."
import numpy as np
import jax
import jax.numpy as jnp


def subrange_centers(r0: np.ndarray, rr: np.ndarray, nsubr: int) -> np.ndarray:
    """Compute the center of the subrange gates for each range gate bin.

    Assuming that the radar is located at the origin.

    Parameters
    ==========
    r0: ndarray
        Lower bounds of the original range gate grid [nhigh]
    rr: ndarray
        The range resolution
    nsubr: int
        Number of subranges.
    
    Returns
    =======
    c: float
        Subrange centers [nhigh, nsubr]
    """
    csubr = (np.arange(nsubr) - nsubr/2 + 0.5) / nsubr + 0.5
    return np.reshape(r0, (-1, 1)) + rr * np.reshape(csubr, (1, -1))


def steering_vector(
    k: np.ndarray,
    p: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Generalized steering vector for frequency and spatial domain interferometry.

    The result is not normalized.

    Parameters
    ==========
    k: ndarray
        The wave number of the size `[nfreq]`, where `nfreq` is the number of frequencies.
    p: ndarray
        The antenna location of the size `[nant, 3]`, where `nant` is the number of antennas.
    r: ndarray
        The range and subrange gate bounds `[nr, nsubr]`, where `nr` is the number of range gate and
        `nsubr` is the number of subdivision. Can be computed by `subrange_centers()`.
    v: ndarray
        Radial vectors of the size `[ndir, 3]`, where `ndir` is the number of directions.

    Returns
    =======
    a: ndarray
        Steering vector of the size `[nr, nsubr, ndir, nfreq, nant]`.
    """
    k = np.ravel(k)
    p = np.reshape(p, (-1, 3))
    r = np.atleast_2d(r)[:, :, None, None]
    v = np.reshape(v, (-1, 3))[None, None, :, :]

    # Tx distance is the subrange gate bounds.
    # Here, we assume that r is measured from the center of the antenna array.
    tx_distance = r

    # Rx positions are measured from the center of the antenna array.
    rx_positions = v * r + np.mean(p, axis=0)[None, None, None, :]

    # Rx distance is measured for each antenna [nr, nsubr, ndir, nant].
    rx_distance = np.linalg.norm(rx_positions[..., None, :] - p[..., :], axis=-1)

    distance = tx_distance + rx_distance
    a = np.exp(1j * k[None, None, None, :, None] * distance[..., None, :])

    return a


def capon(
    rxx_i: jnp.ndarray,
    a: jnp.ndarray,
):
    """Capon beamforming.
    
    Parameters
    ==========
    rxx_i: ndarray
        The covariance matrix of the size `[nchan, nchan]`, where `nchan` is
        the number of channels.
        `nchan = nfreq * nant` for FDI + SDI cases.
    a: ndarray
        The steering vector of the size `[nconst, nchan]`, where nconst is the
        number of constraints.
        `nconst = ndir * nsubr` for FDI + SDI cases.
        
    Returns
    =======
    spc: array of float of the size [nconst]
        Capon spectrum for `nconst` set of constraints.
    """
    rxx_i = jnp.array(rxx_i)
    a = jnp.array(a)

    def each(a1: jnp.ndarray) -> jnp.ndarray:
        """Compute the Capon spectrum for each steering vector."""
        return 1 / jnp.abs(a1.conj().T @ rxx_i @ a1.T)

    return jax.vmap(each)(a)
