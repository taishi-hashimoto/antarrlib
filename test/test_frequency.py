from antarrlib import freq2wlen, freq2wnum
from math import isclose

def test_frequency():
    frequency = 47e6

    wavelength = freq2wlen(frequency)

    assert isclose(wavelength, 6.378562936170213)

    wavenumber = freq2wnum(frequency)

    assert isclose(wavenumber, 0.9850471603172903)