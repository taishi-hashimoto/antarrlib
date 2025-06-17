import numpy as np
import matplotlib.pyplot as plt
from antarrlib.volume import compute_inner_volume_from_min_ze_max, draw_volume, compute_inner_volume
from pathlib import Path


def test_volume():
    ze_max_req = np.deg2rad(10)
    # r_min = 39600
    # r_max = 105300
    r_min = 54300
    r_max = 99300
    volume = compute_inner_volume_from_min_ze_max(ze_max_req, r_min, r_max)
    print(volume)

    assert np.allclose(np.arctan(volume['x_in'] / volume['r_max']), ze_max_req)
    print(f"ze_max required: {np.rad2deg(volume['ze_max']):.4f} deg")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    draw_volume(ax, volume, cylinder=False, box=True)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent.joinpath("test_volume.png"))
