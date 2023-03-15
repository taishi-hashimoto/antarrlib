import numpy as np
from math import isclose
from antarrlib.spherint import patch_area
from antarrlib.fundamental import freq2wlen


def test_sphere():
    
    ze = np.linspace(0, 180, 181)
    az = np.linspace(-180, 180, 361)
    ra = 1.
    # Spherical integral.
    p1 = patch_area(np.deg2rad(ze), np.deg2rad(az))
    a1 = np.sum(p1)
    # Theoretical, a sphere with its radius = ra
    a2 = 4 * np.pi * ra**2
    # Difference of these should be small.
    assert isclose(a2 - a1, 0.00019937363414079812)


def short_dipole(frequency, length, theta, i=1, r=1):
    """
    The radiation field of a short dipole.

    :param frequency:
    :param length: element length.
    :param theta:
    :param i: Current.
    :param r: Distance.
    :return:
    """
    wl = 299792458. / frequency
    ph = 2 * np.pi / wl * r
    return 120j*np.pi * i * length * np.exp(-1j * ph) / (2 * wl * r) * np.sin(theta)


def test_radiated_power():
    
    z = np.linspace(0, 180, 1801)
    a = np.linspace(-180, 180, 3601)
    r = 1
    
    f = 47e6
    l = freq2wlen(f)
    d = 0.1 * l
    I = 1
    Z0 = 120 * np.pi

    # Electric field [V/m] at the distance r [m].
    e = np.tile(short_dipole(f, d, np.deg2rad(z), I, r), (len(a), 1)).T
    
    # The radiation power density [W/m^2] at the distance r.
    p = np.abs(e) ** 2

    # Spherical integral of the radiation pattern.
    spa = patch_area(np.deg2rad(z), np.deg2rad(a), r)

    # Numerical calculation of total radiation power.
    num = np.sum(p * spa) / 2 / Z0

    print("--")

    print("spherint.patch_area() vs. theory:")

    print("  Numerical   Total Radiation Power: {:.13f} W".format(num))

    # Theoretical value and its resistance.
    P = np.pi * Z0 / 3 * (I * d / l) ** 2
    R = 2 * P / I ** 2
    print("  Theoretical Total Radiation Power: {:.13f} W".format(P))
    print()

    print("Other important values:")
    print("  Theoretical Radiation Resistance : {:.5f} Ohm".format(R))

    # Difference should be small.
    assert isclose(P - num, -4.560796185160143e-13)