import numpy as np
import jax
import jax.numpy as jnp


def subrange(r0: np.ndarray, rr: np.ndarray, nsubr: int) -> np.ndarray:
    """Compute subrange gates for each range gate.

    Assuming that the radar is located at the origin.

    Parameters
    ==========
    r0: ndarray
        Lower bounds the range gate [nhigh]
    rr: ndarray
        The range resolution
    nsubr: int
        Number of subranges.
    
    Returns
    =======
    c: float
        Subranges [nhigh, nsubr]
    """
    csubr = (np.arange(nsubr) - nsubr/2 + 0.5) / nsubr + 0.5
    return np.reshape(r0, (-1, 1)) + rr * np.reshape(csubr, (1, -1))


def steering_vector(
    k: jnp.ndarray,
    r: jnp.ndarray,
    v: jnp.ndarray,
    c: jnp.ndarray,
    s: int = 1
) -> jnp.ndarray:
    """Generalized steering vector for frequency and spatial domain interferometry.

    The result is not normalized.

    Parameters
    ==========
    k: ndarray
        The wave number [nfreq].
    r: ndarray
        The antenna location [nant, 3].
    v: ndarray
        Radial vectors [ndir, 3], where ndir is the number of directions.
    c: ndarray
        The range and subrange gate bounds [nr, nsubr], wherenr is the number of range gate and
        nsubr is the number of subdivision. Can be computed by subrange().
    s: int
        +1 for Tx, -1 for Rx.

    Returns
    =======
    a: ndarray
        Steering vector [nr, ndir, nsubr, nfreq, nant]
    """
    v = jnp.reshape(v, (-1, 3))
    c = jnp.atleast_2d(c)

    ndir = len(v)
    nr, nsubr = c.shape

    # Tx distance is the subrange gate bounds.
    # Here, we assume that c is measured from the center of the antenna array.
    tx_distance = c[:, None, :, None]

    # Rx positions are measured from the center of the antenna array.
    rx_positions = v[None, :, None, :] * c[:, None, :, None] + jnp.mean(r, axis=0)[None, None, None, :]

    # Rx distance is measured for each antenna.
    rx_distance = jnp.linalg.norm(rx_positions[..., None, :] - r[None, ..., :], axis=-1)
    
    distance = tx_distance + rx_distance
    a = jnp.exp(1j * k[None, :, None] * distance[..., None, :])

    return a