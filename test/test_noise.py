import numpy as np
from antarrlib import periodogram, mean_chisq, noise


def test_noise():
    true_power = 1.5
    nicoh = 100
    ndata = 128
    ndiv = 8
    raw = noise(ndata*nicoh, true_power)
    noi = periodogram(raw).reshape(nicoh, -1).mean(axis=0)

    estimated = mean_chisq(noi, df=2*nicoh, nseg=ndiv)
    assert np.allclose(true_power, estimated, 1e-1)
        
    print(true_power)
    print(estimated)
