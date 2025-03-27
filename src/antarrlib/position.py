"Antenna position helper."
import numpy as np
from numpy.typing import NDArray


def incirc_trigrid(dx: float, radius: float) -> NDArray[np.float64]:
    """Generate a triangular grid within a circle located at `(0, 0, 0)`.

    A lattice point always comes to the center of the circle.

    Parameters
    ==========
    dx: float
        The distance between each antenna.
    radius: float
        The radius of the outer circle.

    Returns
    =======
    positions: ndarray of float
        [N, 3] array of float, the lattice points of the triangular grid in the given circle.
    """
    dy = dx * np.sqrt(3) / 2
    positions = []
    for iy in range(int(-radius // dy), int(radius // dy + 2)):
        y = iy * dy
        for ix in range(int(-radius // dx), int(radius // dx + 2)):
            x = ix * dx
            if abs(iy) % 2 == 1:
                x += dx/2
            if x**2 + y**2 < radius**2:
                positions.append((x, y, 0.))
    return np.array(positions)
