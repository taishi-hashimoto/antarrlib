"Imaging library."
import numpy as np


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
    v: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Generalized steering vector for frequency and spatial domain interferometry.

    The result is not normalized.

    Parameters
    ==========
    k: ndarray
        The wave number of the size `[nfreq]`, where `nfreq` is the number of frequencies.
    p: ndarray
        The antenna location of the size `[nant, 3]`, where `nant` is the number of antennas.
    v: ndarray
        Radial vectors of the size `[ndir, 3]`, where `ndir` is the number of directions.
    r: ndarray
        The range and subrange gate bounds `[nr, nsubr]`, where `nr` is the number of range gate and
        `nsubr` is the number of subdivision. Can be computed by `subrange()`.

    Returns
    =======
    a: ndarray
        Steering vector of the size `[nr, ndir, nsubr, nfreq, nant]`.
    """
    k = np.ravel(k)
    p = np.reshape(p, (-1, 3))
    v = np.reshape(v, (-1, 3))
    r = np.atleast_2d(r)[:, None, :, None]

    # Tx distance is the subrange gate bounds.
    # Here, we assume that c is measured from the center of the antenna array.
    tx_distance = r

    # Rx positions are measured from the center of the antenna array.
    rx_positions = v[None, :, None, :] * r + np.mean(p, axis=0)[None, None, None, :]

    # Rx distance is measured for each antenna.
    rx_distance = np.linalg.norm(rx_positions[..., None, :] - p[None, ..., :], axis=-1)

    distance = tx_distance + rx_distance
    a = np.exp(1j * k[None, :, None] * distance[..., None, :])

    return a
