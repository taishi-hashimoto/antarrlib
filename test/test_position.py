from antarrlib import incirc_trigrid, freq2wlen
import matplotlib.pyplot as plt
from os.path import join, dirname


def test_incirc_trigrid():
    f = 47e6
    lambda_ = freq2wlen(f)
    radius = 10
    # 0.7 lambda spacing.
    antpos = incirc_trigrid(lambda_ * 0.7, radius)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(antpos[:, 0], antpos[:, 1], c='red', marker='o')
    ax.add_patch(plt.Circle((0, 0), radius, color='blue', fill=False, linestyle='dashed'))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect('equal')
    ax.grid()
    fig.tight_layout()
    fig.savefig(join(dirname(__file__), "fig_incirc_trigrid.png"))
