"Simple antenna array module."
from typing import Callable
import numpy as np
from .fundamental import steering_vector, radial, freq2wlen, wlen2wnum


class AntennaArray:
    """Antenna array class."""

    def __init__(
        self,
        frequency_hz: float,
        positions_m: np.ndarray,
        element_pattern: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> None:
        """Antenna array class.

        Parameters
        ==========
        frequency_hz : float
            Frequency in Hz.
        positions_m : np.ndarray, shape (N, 2)
            Antenna positions in meters in the local ENU coordinate system.
            N is the number of antennas.
        element_pattern : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Element pattern function that takes zenith and azimuth angles
            (in radians) and returns the power pattern.
        """
        self.freq = frequency_hz
        self.wavelength = freq2wlen(frequency_hz)
        self.k = wlen2wnum(self.wavelength)
        self.positions = positions_m
        self.element = element_pattern

    @property
    def num_antennas(self) -> int:
        "Number of antennas."
        return len(self.positions)

    def steering_vector(
        self,
        ze: np.ndarray,
        az: np.ndarray,
        normalize: bool = False
    ) -> np.ndarray:
        """Compute steering vector for given directions.

        Parameters
        ==========
        ze : np.ndarray
            Zenith angle in radians.
        az : np.ndarray
            Azimuth angle in radians.
        normalize : bool
            Whether to normalize the steering vector.

        Returns
        =======
        sv : np.ndarray, shape (N, M)
            Steering vector for each direction.
            N is the number of antennas.
        """
        s = np.shape(ze)
        ze = ze.ravel()
        az = az.ravel()
        v = steering_vector(
            self.k,
            self.positions,
            radial(ze, az))
        if normalize:
            v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return v.reshape(s + (len(self.positions),))

    def far_field(self, ze: np.ndarray, az: np.ndarray) -> np.ndarray:
        """Compute far field response in complex amplitude for given directions.

        Parameters
        ==========
        ze : np.ndarray
            Zenith angle in radians.
        az : np.ndarray
            Azimuth angle in radians.

        Returns
        =======
        sv : np.ndarray, shape (N, M)
            Steering vector for each direction.
            N is the number of antennas.
        """
        s = np.shape(ze)
        ze = ze.ravel()
        az = az.ravel()
        v = self.steering_vector(ze, az)
        e = np.sqrt(self.element((ze, az)))[:, None]
        return (v * e).reshape(s + (len(self.positions),))

    @staticmethod
    def _ensure_3d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr[None, None, :]
        elif arr.ndim == 2:
            return arr[None, :, :]
        return arr

    def beam_pattern(
        self,
        beam_ze: np.ndarray,
        beam_az: np.ndarray,
        ze: np.ndarray,
        az: np.ndarray
    ) -> np.ndarray:
        """Compute beam pattern (in power) for given beam directions.

        Parameters
        ==========
        beam_ze : np.ndarray
            Beam zenith angle in radians.
        beam_az : np.ndarray
            Beam azimuth angle in radians.
        ze : np.ndarray
            Zenith angle for evaluation in radians.
        az : np.ndarray
            Azimuth angle for evaluation in radians.
        Returns
        =======
        p : np.ndarray
            Beam pattern.
        """
        s = np.shape(beam_ze) + np.shape(ze)
        v = self.far_field(ze, az)
        w = self.steering_vector(beam_ze, beam_az, normalize=True)
        v = self._ensure_3d(v)
        w = self._ensure_3d(w)
        nze_beam, naz_beam, nant = w.shape
        nze_eval, naz_eval, _ = v.shape
        p = np.abs(
            w.reshape(nze_beam*naz_beam, nant).dot(
                v.reshape(nze_eval*naz_eval, nant).conj().T
            ).reshape(s))**2
        return p
