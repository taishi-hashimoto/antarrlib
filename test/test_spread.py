# %%
import numpy as np
from antarrlib.simulation import spread_radial, angle_between, spread_range, spread_power



# %%
n_points = 10000
# np.random.seed(1126)
v0 = np.array([0, 0, 1])
angular_spread = np.deg2rad(1.0)
v = spread_radial(v0, angular_spread)
angles = angle_between(v0, v)
print(f"std: {np.rad2deg(np.std(angles)):.3f}°")

# %%

r0 = 80e3
range_spread = 100
r = spread_range(r0, range_spread)
print(f"std: {np.std(r):.3f} m")
# %% Positions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

positions = r[:, None] * v

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2],
           c=90-np.rad2deg(angles), cmap='viridis', s=np.abs(r-r0)/10, alpha=0.5)

origin = np.zeros(3)
center_point = r0 * v0
ax.plot([0, center_point[0]], [0, center_point[1]], [0, center_point[2]], 'r-')

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Spread radial vectors \n(color: angle offset, size: range offset)\n(brighter: closer to center)")
plt.show()
# %% 
from antarrlib.volume import set_axes_equal

power = spread_power(r0, v0, r, v, angular_spread, range_spread)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:,0], positions[:,1], positions[:,2] - r0,
           c=power, cmap='viridis', s=power*20, alpha=0.5)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Power spread")
set_axes_equal(ax)
plt.show()
# %%