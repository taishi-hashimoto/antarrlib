# %%
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm.auto import tqdm
from antarrlib import freq2wlen, freq2wnum, incirc_trigrid, dB, SPEED_OF_LIGHT as c0
from antarrlib.simulation import point_source, noise
from antarrlib.imaging import steering_vector, subrange_centers, capon
from icogrid import Icogrid
from icogrid.skymap import direction, plot_skymap, triang_skymap

from bpadmm import basis_pursuit_admm
from bpadmm.jax import cosine_decay_schedule


c0 = 299792458.0  # Speed of light [m/s]
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
ax.set_title("Antenna positions")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_aspect('equal')
ax.grid()
fig.tight_layout()

# %%

# Simulate point source.
target_x = 2000  # m
target_y = 3000  # m
target_altitude = 80e3  # [m]
target_position = np.array([target_x, target_y, target_altitude])

# True target direction and distance.
ze0, az0 = direction(target_position, degrees=False)
ze0 = np.arccos(target_position[2] / np.linalg.norm(target_position, axis=-1))
az0 = np.arctan2(target_position[1], target_position[0])

target_distance = np.linalg.norm(target_position)
print("Target:")
print(f"  True distance: {target_distance/1e3:.03f} km")
print(f"  True direction (ze, az): {np.rad2deg(ze0):.02f}°, {np.rad2deg(az0):.02f}°")


received_signals = point_source(k, antenna_positions, target_position, 1).reshape((num_freqs, num_antennas))
received_signals += noise(received_signals.shape, 0.01)

# %%
range_gate_width = 2e-6
# Default range gate's lower bound.
# Target will be at the center of the range gate.
range_gate_lo = target_altitude - range_gate_width / 2 * c0
# Target position offset ratio within range gate [0, 1]
# Center will be 0.5.
offset_ratio = 0.5
range_gate_lo += range_gate_width * c0 * (0.5 - offset_ratio)

# %%
nsubr = 32

r_m = subrange_centers(range_gate_lo, range_gate_width*c0, nsubr).ravel()

angular_separation = 0.1  # [deg]
ico = Icogrid.from_angular_separation(angular_separation, degrees=True)
ico.set_extent(zemax=30, zemin=0, degrees=True)

# Decimate grid to some value.
ndirs_req = 40000
rng = np.random.default_rng(1)
indices = rng.choice(len(ico), size=ndirs_req, replace=False)

a = steering_vector(k, antenna_positions, r_m, ico.vertices[indices]).reshape((nsubr, -1, num_freqs, num_antennas))
# print(a)
# %%
# Fourier method.
y = np.abs(np.sum(a.conj() * received_signals, axis=(-1, -2)))**2

# Capon method.
nchan = num_antennas * num_freqs
x = received_signals.ravel()
rxx = x[:, None].dot(x.conj()[None, :])
rxx_i = np.linalg.inv(rxx + 1e-4 * np.eye(nchan))
aa = a.reshape(-1, nchan)
# %%
y2 = capon(rxx_i, aa).reshape((nsubr, -1))
# %%
# Determin nrows and ncols in the figure.
# nrow = int(np.sqrt(nsubr))
nrow = 4
ncol = nsubr // nrow
if nrow * ncol < nsubr:
    ncol += 1


# Plotting method.
def plot_images(y, zemax=10, scatter=False):
    # Color range.
    y_dB = dB(y, "max")
    ymaxs = np.nanmax(y, axis=1)
    is_bad = np.isnan(y_dB.ravel())
    y_dB.flat[is_bad] = np.nan if scatter else -40
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("k", 1.0)  # Set bad values to black.
    norm = Normalize(vmin=-20, vmax=0)

    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 8), subplot_kw=dict(projection="polar"))
    for ax in axes.flat:
        ax.set_thetagrids([], [])
        ax.set_rgrids([], [])
        
    ze, az = ico.to_direction()[indices, :].T
    tri = triang_skymap(ze, az, degrees=False)
    # mask = np.any(np.where(is_bad[tri.triangles], True, False), axis=-1)
    # tri.set_mask(mask)
    with tqdm(total=nsubr) as pbar:
        for i, raxis1 in enumerate(r_m):
            p = y_dB[i, :]
            ax = axes.flat[i]
            ax.set_facecolor("k")
            pbar.set_description(f"{p.shape}")
            if scatter:
                ax.scatter(az, np.rad2deg(ze), c=p, cmap=cmap, norm=norm, s=1)
            else:
                plot_skymap(ax, p, tri=tri, cmap=cmap, norm=norm, levels=20)
            ax.set_title(f"{raxis1/1e3:.02f} km")
            ax.set_thetagrids(range(0, 360, 45), [])
            ax.set_rgrids(range(0, 11, 10), [])
            ax.set_rlim(0, zemax)
            ax.plot(az0, np.rad2deg(ze0), "ro", mfc="none", ms=10)
            pbar.update(1)
    fig.tight_layout()
    return ymaxs
# %%
ymaxs = plot_images(y)
# %%
ymaxs2 = plot_images(y2)
# %%

a0 = steering_vector(k, antenna_positions, r_m, ico.vertices).reshape(-1, nchan) / np.sqrt(nchan)

# Decimate grid to some value.
ndirs_req = 40000
rng = np.random.default_rng(1)
indices = rng.choice(len(ico), size=ndirs_req, replace=False)

a2 = steering_vector(k, antenna_positions, r_m, ico.vertices[indices]).reshape(-1, nchan) / np.sqrt(nchan)

ndirs_eval = 100
indices2 = rng.choice(len(ico)*nsubr, size=ndirs_eval, replace=False)

v = a0[indices2]

C = np.square(np.abs(v.conj() @ a2.T))

b = np.real(np.einsum('lm,mn,ln->l', v.conj(), rxx, v))  # shape (L,)

Ci = np.linalg.pinv(C)

# %%

MAXITER = 1000
STEPITER = 50

threshold = cosine_decay_schedule(MAXITER * STEPITER, 0.1, 1e-3) * np.max(b)

result = basis_pursuit_admm(
        C, b, threshold=threshold,
        maxiter=MAXITER, stepiter=STEPITER, patience=50,
        Ai=Ci, init_x=y2)

x_ = result.x
state = result.state

# Plot convergence.
fig, axes = plt.subplots(2, 2, figsize=(10, 3))
ax = axes[0, 0]
ax.set_yscale("log")
ax.grid()
ax.plot(state.diff_x.T)
ax.axvline(result.nit, c="k", ls=":")
ax.set_title("Convergence of x")
ax = axes[0, 1]
ax.set_yscale("log")
ax.grid()
ax.plot(state.l1_norm.T)
ax.axvline(result.nit, c="k", ls=":")
ax.set_title("Convergence of l1")
ax = axes[1, 0]
ax.set_yscale("log")
ax.grid()
ax.plot(state.res_prim.T)
ax.axvline(result.nit, c="k", ls=":")
ax.set_title("Primal Residual")
ax = axes[1, 1]
ax.set_yscale("log")
ax.grid()
ax.plot(state.res_dual.T)
ax.axvline(result.nit, c="k", ls=":")
ax.set_title("Dual Residual")
fig.tight_layout()
result

# %%
ymaxs4 = plot_images(np.where(x_.real > 0, x_.real, np.nan).reshape(nsubr, -1), scatter=False)

ymaxs_dB = dB(ymaxs, "max")
ymaxs2_dB = dB(ymaxs2, "max")
ymaxs4_dB = dB(ymaxs4, "max")
plt.figure(figsize=(10, 3))
plt.plot(r_m/1e3, ymaxs_dB, marker=".")
plt.plot(r_m / 1e3, ymaxs2_dB, marker=".")
plt.plot(r_m/1e3, ymaxs4_dB, marker=".")
plt.axvline(target_distance/1e3, c="k", ls=":")
plt.gca().xaxis.set_major_formatter("{x:.2f} km")
plt.gca().yaxis.set_major_formatter("{x:.0f} dB")
plt.ylim(-40, 5)
plt.tight_layout()
# %%
# %%

# %%
import jax
import jax.numpy as jnp

# v: shape (L, M)  各ビーム方向のステアリングベクトル（L個）
# rxx: shape (M, M)  相関行列

def each(vl: jnp.ndarray) -> jnp.ndarray:
    """Compute beam output power at one direction."""
    return jnp.real(vl.conj().T @ rxx @ vl)  # スカラー

# batched version: returns shape (L,)
b_ = jax.vmap(each)(v)  # v: (L, M) → b: (L,)
