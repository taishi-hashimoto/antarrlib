"In-beam imaging with the spatial- and frequency-domain interferometry (SDI + FDI)."

from antarrlib import freq2wlen, freq2wnum, radial, incirc_trigrid, dB, SPEED_OF_LIGHT as c0
import numpy as np
import matplotlib.pyplot as plt


import jax.numpy as jnp
from jax import jit, vmap

@jit
def capon(rxx_i, a):
    rxx_i = jnp.array(rxx_i)
    a = jnp.array(a)
    def compute_power(a1):
        denom = a1.conj().T @ rxx_i @ a1.T
        return jnp.square(jnp.abs(1 / denom))

    return vmap(compute_power)(a)


def test_imaging():
    # %% Settings

    # Target settings.
    target_x = -3000  # m
    target_y = -4500  # m
    target_altitude = 80.15e3  # [m]
    target_position = np.array([target_x, target_y, target_altitude])

    # True target direction and distance.
    ze0 = np.arccos(target_position[2] / np.linalg.norm(target_position, axis=-1))
    az0 = np.arctan2(target_position[1], target_position[0])

    # [[ze0, az0]] = direction(target_position)
    target_distance = np.linalg.norm(target_position)
    print("Target:")
    print(f"  True distance: {target_distance/1e3:.03f} km")
    print(f"  True direction (ze, az): {np.rad2deg(ze0):.02f}°, {np.rad2deg(az0):.02f}°")

    # Range gate settings.
    range_gate_width = 2e-6
    # Default range gate's lower bound.
    # Target will be at the center of the range gate.
    range_gate_lo = target_altitude - range_gate_width / 2 * c0
    # Target position offset ratio within range gate [-1, 1]
    offset_ratio = 0.0
    range_gate_lo += range_gate_width / 2 * c0 * -offset_ratio/2

    # %% Radar settings.

    f0 = 47e6  # Center frequency [Hz]
    freq_offsets = np.array([-500e3, -250e3, 0, 250e3, 500e3])   # Frequency offsets [Hz]
    num_freqs = len(freq_offsets)

    # Wavenumber for each frequency.
    k = freq2wnum(f0 + freq_offsets)

    # Antenna settings.
    lambda_ = freq2wlen(f0)  # wavelength [m]
    radius = 10  # [m]
    antenna_positions = incirc_trigrid(lambda_ * 0.7, radius)
    num_antennas = len(antenna_positions)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(antenna_positions[:, 0], antenna_positions[:, 1], c='red', marker='o')
    ax.add_patch(plt.Circle((0, 0), radius, color='blue', fill=False, linestyle='dashed'))
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.set_aspect('equal')
    ax.grid()
    fig.tight_layout()

    # Generate received signals for each frequency and antenna [num_freqs, num_antennas]
    tx_distance = np.linalg.norm(target_position)
    rx_distance = np.linalg.norm(antenna_positions - target_position, axis=-1)
    distance = tx_distance + rx_distance
    received_signals = np.exp(1j * k[:, None] * distance[None, :])

    # %% Spatial domain interferometry (SDI) + Frequency domain interferometry (FDI)

    # The number of subranges.
    # Each range gate is divided into this number of subranges.
    nsubr = 32

    # Index offset of subranges [nsubr].
    csubr = (np.arange(nsubr) - nsubr/2 + 0.5) / nsubr + 0.5

    # Range evaluation grid.
    r_m = range_gate_lo + csubr * range_gate_width * c0

    # Angular evaluation grid.
    ze_deg = np.linspace(0, 30, 31)
    az = np.linspace(-np.pi, np.pi, 361)

    # Final evaluation grid.
    ze_g, az_g, r_g = np.meshgrid(np.deg2rad(ze_deg), az, r_m, indexing="ij")

    # Steering vector.
    tx_distance = r_g.ravel()[:, None]
    rx_positions = radial(ze_g, az_g) * r_g.ravel()[:, None]
    rx_distance = np.linalg.norm(rx_positions[:, None, :] - antenna_positions[None, ...], axis=-1)
    distance = tx_distance + rx_distance
    e = np.exp(1j * k[None, :, None] * distance[..., None, :])

    # %% Imaging.

    # Beamforming.
    y = np.abs(np.sum(e.conj() * received_signals[None, ...], axis=(1, 2)))**2
    y.shape = ze_g.shape

    # Capon method.
    nchan = num_antennas*num_freqs
    x = received_signals.ravel()
    rxx_i = np.linalg.inv(x[:, None].dot(x.conj()[None, :]) + 0.001 * np.eye(nchan))
    ee = e.reshape(-1, nchan)
    # y2 = []
    # for ee1 in tqdm(ee):
    #     y2.append(np.abs(1 / ee1.conj().dot(rxx_i).T.dot(ee1.T))**2)
    y2 = capon(rxx_i, ee)
    y2 = np.reshape(y2, ze_g.shape)

    # %% Results for beamforming.

    # Determin nrows and ncols in the figure.
    nrow = int(np.sqrt(nsubr))
    ncol = nsubr // nrow
    if nrow * ncol < nsubr:
        ncol += 1

    # Color range.
    noi = np.median(y)
    y_dB = dB(y, noi)
    ymax = np.max(y_dB)
    ymaxs = np.max(y, axis=(0, 1))

    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 8), subplot_kw=dict(projection="polar"))
    for ax in axes.flat:
        ax.set_thetagrids([], [])
        ax.set_rgrids([], [])
    for i, raxis1 in enumerate(r_m):
        p = y_dB[..., i]
        ax = axes.flat[i]
        ax.pcolormesh(az, ze_deg, p, vmin=0, vmax=ymax)
        ax.set_title(f"{raxis1/1e3:.02f} km")
        ax.set_thetagrids(range(0, 360, 45), [])
        ax.set_rgrids(range(0, 90, 10), [])
        ax.plot(az0, np.rad2deg(ze0), "ro", mfc="none", ms=10)
    fig.tight_layout()

    # %% Results for Capon

    # Color range.
    noi = np.median(y2)
    y_dB = dB(y2, noi)
    ymax = np.max(y_dB)
    ymaxs2 = np.max(y2, axis=(0, 1))

    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 8), subplot_kw=dict(projection="polar"))
    for ax in axes.flat:
        ax.set_thetagrids([], [])
        ax.set_rgrids([], [])
    for i, raxis1 in enumerate(r_m):
        p = y_dB[..., i]
        ax = axes.flat[i]
        ax.pcolormesh(az, ze_deg, p, vmin=0, vmax=ymax)
        ax.set_title(f"{raxis1/1e3:.02f} km")
        ax.set_thetagrids(range(0, 360, 45), [])
        ax.set_rgrids(range(0, 90, 10), [])
        ax.plot(az0, np.rad2deg(ze0), "ro", mfc="none", ms=10)
    fig.tight_layout()

    # %% Peak at each range.

    ymaxs_dB = dB(ymaxs, "max")
    ymaxs2_dB = dB(ymaxs2, "max")
    plt.figure(figsize=(10, 3))
    plt.plot(r_m/1e3, ymaxs_dB, marker=".")
    plt.plot(r_m/1e3, ymaxs2_dB, marker=".")
    plt.axvline(target_distance/1e3, c="k", ls=":")
    plt.gca().xaxis.set_major_formatter("{x:.2f} km")
    plt.gca().yaxis.set_major_formatter("{x:.0f} dB")
    plt.ylim(-40, 5)
    plt.tight_layout()

    # %%
    plt.show()
