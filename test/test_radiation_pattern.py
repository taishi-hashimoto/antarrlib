import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from os.path import join, dirname
from math import isclose
from antarrlib import freq2wnum, steering_vector, dB, idB, radial


def test_radiation_pattern():
    # Frequency and wavenumber.
    frequency = 47e6
    k = freq2wnum(frequency)

    # Antenna position.
    df = pd.read_csv(join(dirname(__file__), "antpos.csv"))
    r = df.loc[df.Ready == 1, ["X(m)", "Y(m)", "Z(m) "]]

    # Element pattern.
    df = pd.read_csv(join(dirname(__file__), "antptn.csv"), header=0, index_col=0).rename(columns=float)
    lut = RegularGridInterpolator(
        (np.deg2rad(df.index.values), np.deg2rad(df.columns.values)),
        idB(df.values))

    # Element pattern function (NOTE: this returns power.)
    def element(ze, az):
        return lut(np.c_[np.ravel(ze), np.ravel(az)]).reshape(np.shape(ze))

    # Evaluation grid.
    ze = np.linspace(0, 90, 91)
    az = np.linspace(-180, 180, 361)
    ze_g, az_g = np.deg2rad(np.meshgrid(ze, az, indexing="ij"))

    # Set beam direction to 10 deg north.
    b = radial(np.deg2rad(10), np.deg2rad(90))

    # Weight vector for the beam.
    w = steering_vector(k, r, b) / np.sqrt(len(r))

    # Radial vectors to all evaluation direction.
    v = radial(ze_g, az_g)

    # Steering vector on the evaluation grid.
    a = steering_vector(k, r, v)

    # Element pattern on the evaluation grid.
    e = element(ze_g, az_g)

    # Radiation pattern in power.
    rp = np.reshape(
        np.abs(
            w.conjugate().dot(a.transpose() * np.sqrt(e).ravel())
        )**2,
        np.shape(ze_g)
    )

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="polar"))
    rp_dB = dB(rp)
    peak = np.max(rp_dB)
    fig.suptitle(f"Radiation Pattern of the PANSY radar (max={peak:.2f}dB)")
    m = ax.pcolormesh(np.deg2rad(az), ze, rp_dB, vmin=peak - 40)
    fig.colorbar(ax=ax, mappable=m).set_label("Radiation Power [dBi]")
    fig.tight_layout()

    # ~ 7.6 dBi (element gain) + 30 dBi (= dB(1000)) = 37.6 dBi
    # Attenuated a bit by beam steering.
    assert isclose(peak, 37.49609445360281)
    
    plt.show()