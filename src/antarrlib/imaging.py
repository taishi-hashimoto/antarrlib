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
    normalize: bool = False,
) -> np.ndarray:
    """Generalized steering vector for frequency and spatial domain interferometry.

    The result is not normalized by default.

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
        Radial vectors of the size `[ndir, 3]`, `[nsubr, ndir, 3]`, or `[nr, nsubr, ndir, 3]`.
        Here, `ndir` is the number of directions.
        For the latter two cases, radial vectors can be different for each subrange/range gate.
    normalize: bool, optional
        If True, the steering vector is normalized to have unit norm.
        Note this regards the last two axes, i.e., the antenna and frequency axes, to be flattened.

    Returns
    =======
    a: ndarray
        Steering vector of the size `[nr, nsubr, ndir, nfreq, nant]`.
    """
    k = np.ravel(k)
    p = np.reshape(p, (-1, 3))
    r = np.atleast_2d(r)[:, :, None, None]
    v = np.atleast_2d(v)
    if v.ndim == 2:  # [ndir, 3]
        v = v[None, None, :, :]
    elif v.ndim == 3:  # [nsubr, ndir, 3]
        v = v[None, :, :, :]
    elif v.ndim == 4:  # [nr, nsubr, ndir, 3]
        pass
    else:
        raise ValueError(f"Invalid shape of v: {v.shape}. Expected [ndir, 3], [nsubr, ndir, 3], or [nr, nsubr, ndir, 3].")

    # Tx distance is the subrange gate bounds.
    # Here, we assume that r is measured from the center of the antenna array.
    tx_distance = r

    # Rx positions are measured from the center of the antenna array.
    rx_positions = v * r + np.mean(p, axis=0)[None, None, None, :]

    # Rx distance is measured for each antenna [nr, nsubr, ndir, nant].
    rx_distance = np.linalg.norm(rx_positions[..., None, :] - p[..., :], axis=-1)

    # Two way distance.
    distance = tx_distance + rx_distance

    # Unnormalized steering vector.
    a = np.exp(1j * k[None, None, None, :, None] * distance[..., None, :])

    if normalize:
        # Keep the original shape.
        shape = a.shape
        # Flatten the last two axes (frequency and antenna) to a single channel axis.
        a.shape = a.shape[:3] + (-1,)
        # Normalize the steering vector along the channel axis.
        a /= np.linalg.norm(a, axis=-1, keepdims=True)
        # Restore the original shape.
        a.shape = shape
    return a


def capon(
    rxx: np.ndarray,
    a: np.ndarray,
    regularization: float = None,
    batch_size: int = None,
    devices=None,
) -> np.ndarray:
    """Capon beamforming.
    
    Parameters
    ==========
    rxx: ndarray
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
    rxx = np.array(rxx)

    if regularization is not None:
        rxx += regularization * np.eye(rxx.shape[0])

    if devices is None:
        devices = jax.devices()
    ndevices = len(devices)

    def each(a1: jnp.ndarray) -> jnp.ndarray:
        """Compute the Capon spectrum for each steering vector."""
        return jnp.abs(1. / jnp.vdot(a1, jnp.linalg.solve(rxx, a1)))

    if batch_size is None:
        batch_size = len(a)
    nbatches = len(a) // batch_size

    if ndevices == 1 or nbatches == 1:  # Use vmap
        batched_each = jax.vmap(each, in_axes=(0,))
        result = np.empty((a.shape[0],), dtype=np.float64)
        for i in range(0, a.shape[0], batch_size):
            result[i:i + batch_size] = batched_each(a[i:i + batch_size])
        return result
    else:  # Use pmap
        batches_per_device = nbatches // ndevices
        rem = nbatches - batches_per_device * ndevices
        nadd = 0
        if rem != 0:
            # Add small number of dummy tasks to make it divisible by n_devices
            nadd = ndevices - rem
            a = _extend(a, nadd)

        # Split the steering vector into batches for each device.
        a = _split(a, ndevices)

        result = _merge(jax.pmap(jax.vmap(each, in_axes=(0,)))(a))
        return result[:-nadd] if nadd > 0 else result


def _extend(arr: np.ndarray, nadd: int):
    """Extend rows by nadd"""
    nsamples, *rest = arr.shape
    return np.resize(arr, (nsamples + nadd, *rest))


def _split(arr: np.ndarray, ndevices: int):
    """Reshape arr so that its leading axis becomes [ndevices, batches_per_device, …]."""
    b, *rest = arr.shape
    if b % ndevices != 0:
        raise ValueError(f"Batch size {b} not divisible by #devices {ndevices}")
    batches_per_device = b // ndevices
    return arr.reshape(ndevices, batches_per_device, *rest)


def _merge(arr: np.ndarray):
    """Inverse of _split()."""
    ndevices, batches_per_device, *rest = arr.shape
    return arr.reshape(ndevices * batches_per_device, *rest)
