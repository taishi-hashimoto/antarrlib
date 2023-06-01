"""Decibel and its inverse functions."""

import numpy as np


def dB(x, b=1):
    """Decibel function, i.e., `y := 10*log10(x/b)`.

    Parameters
    ==========
    x: array-like
        Input value.
    b: array-like
        Base for decibel. Default is 1.

    Returns
    =======
    array-like with the same shape with `x`.
        Decibel value.
    """
    if isinstance(b, (int, float, str)):
        if b == "max":
            b = np.nanmax(x)
        elif b == "min":
            b = np.nanmin(x)
        elif b == 0:
            b = np.full_like(x, np.nan)
    else:
        b = np.array(b, dtype=float)
        b[b == 0] = np.nan
    x = np.array(x, dtype=float)
    x[x <= 0] = np.nan
    return 10 * np.log10(x / b)


def idB(y):
    """Inverse function of `dB`, i.e., `x := 10^(y/10)`."""
    return np.power(10, 0.1*np.array(y))
