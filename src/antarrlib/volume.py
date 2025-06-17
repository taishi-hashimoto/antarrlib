"Design and visualize the radar scan volume."

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from scipy.optimize import least_squares


def compute_inner_volume(ze_max, r_min, r_max_req, degrees=False, error="raise"):
    """Compute the inner volume of a radar beam defined by zenith angle and range.
    
    Parameters
    ----------
    ze_max : float
        Required zenith angle in degrees or radians.
    r_min : float
        Minimum range in meters.
    r_max : float
        Maximum range in meters.
    degrees : bool, optional
        If True, `ze_max` is in degrees. If False, it is in radians.
    Returns
    -------
    dict
        A dictionary containing:
        - 'ze_max': Maximum zenith angle required to include the requested zenith angle.
        - 'r_in': Radius of the cylinder that fits inside the radar volume.
        - 'h_in': Height of the cylinder that fits inside the radar volume.
        - 'x_in': Half-width of the box that fits inside the radar volume in the x-direction.
        - 'y_in': Half-width of the box that fits inside the radar volume in the y-direction.
    """
    # Here, radar volume refers to the volume that is defined by
    # (ze, az, r) coordinates.
    
    if degrees:
        ze_max = np.deg2rad(ze_max)

    # Compute the radius of the cylinder that fits inside the radar volume.
    r_in = r_min * np.tan(ze_max)  # m
    # Compute the height of the cylinder that fits inside the radar volume.
    # h_in = np.sqrt(r_max**2 - r_in**2) - r_min  # m
    r_max = np.hypot(r_max_req, r_in)
    h_in = r_max_req - r_min  # m

    # Maximum zenith angle required to include the requested zenith angle
    # ze_max = np.arctan(r_in / r_max)
    
    if error == "raise" and (np.isnan(h_in) or h_in < 0):
        raise ValueError(f"The specified volume does not allow for a valid inner cylinder: h_in = {h_in} ")

    x_in = r_in / np.sqrt(2)
    y_in = r_in / np.sqrt(2)

    return {
        "r_in": r_in,
        "h_in": h_in,
        "x_in": x_in,
        "y_in": y_in,
        "r_min": r_min,
        "r_max": r_max,  # Computed r_max
        "ze_max": ze_max
    }


def compute_inner_volume_from_min_ze_max(ze_max_req, r_min, r_max, degrees=False):
    """Compute the inner volume of a radar beam defined by the minimum zenith angle required to include the requested zenith angle."""
    if degrees:
        ze_max_req = np.deg2rad(ze_max_req)
        

    def objective(ze_max):
        volume = compute_inner_volume(ze_max, r_min, r_max, error="ignore")
        return np.arctan(volume['x_in'] / volume["r_max"]) - ze_max_req

    result = least_squares(objective, ze_max_req)
    if not result.success:
        raise ValueError("Optimization failed")

    if degrees:
        ze_max = np.rad2deg(result.x[0])
    else:
        ze_max = result.x[0]

    volume = compute_inner_volume(ze_max, r_min, r_max, degrees=degrees)
    volume["ze_max"] = ze_max
    volume["opt"] = result
    return volume


def draw_conical_shell(ax, ze_max, r_min, r_max, resolution=64, color='green', alpha=0.1):
    """Draw a conical shell in 3D space defined by zenith angle and range.
    Parameters
    ----------
    ax : Axes3D
        3D axis object (e.g., `ax = fig.add_subplot(111, projection='3d')`)
    ze_max : float
        Maximum zenith angle in radians.
    r_min : float
        Minimum range (inner radius) in meters.
    r_max : float
        Maximum range (outer radius) in meters.
    resolution : int
        Number of points to sample in the azimuthal and zenith directions.
    color : str
        Color of the surfaces.
    alpha : float
        Transparency of the surfaces (0.0 to 1.0).
    """

    # Evaulation grid.
    theta = np.linspace(0, ze_max, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    tt, pp = np.meshgrid(theta, phi)

    # Lower surface: r = r_min
    x1 = r_min * np.sin(tt) * np.cos(pp)
    y1 = r_min * np.sin(tt) * np.sin(pp)
    z1 = r_min * np.cos(tt)
    ax.plot_surface(x1, y1, z1, color=color, alpha=alpha, linewidth=0)

    # Upper surface: r = r_max
    x2 = r_max * np.sin(tt) * np.cos(pp)
    y2 = r_max * np.sin(tt) * np.sin(pp)
    z2 = r_max * np.cos(tt)
    ax.plot_surface(x2, y2, z2, color=color, alpha=alpha, linewidth=0)

    # Side surface: θ = ze_max 固定, r ∈ [r_min, r_max]
    r = np.linspace(r_min, r_max, resolution)
    phi_side = np.linspace(0, 2 * np.pi, resolution)
    rr, pp_side = np.meshgrid(r, phi_side)

    x3 = rr * np.sin(ze_max) * np.cos(pp_side)
    y3 = rr * np.sin(ze_max) * np.sin(pp_side)
    z3 = rr * np.cos(ze_max)
    ax.plot_surface(x3, y3, z3, color=color, alpha=alpha, linewidth=0)


def draw_cylinder(ax, origin, radius, height, axis='z', resolution=64, color='blue', alpha=0.2):
    """Draw a cylinder in 3D space.
    Parameters
    ----------
    ax : Axes3D
        3D axis object (e.g., `ax = fig.add_subplot(111, projection='3d')`)
    origin : tuple
        Origin of the cylinder (x, y, z) coordinates.
    radius : float
        Radius of the cylinder in meters.
    height : float
        Height of the cylinder in meters.
    axis : str
        Axis along which the cylinder is oriented ('x', 'y', or 'z').
    resolution : int
        Number of points to sample in the azimuthal direction.
    color : str
        Color of the cylinder surface.
    alpha : float
        Transparency of the cylinder surface (0.0 to 1.0).
    """
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    theta = np.linspace(0, 2 * np.pi, resolution)
    h = np.linspace(0, height, 2)
    theta_grid, h_grid = np.meshgrid(theta, h)

    if axis == 'z':
        x = origin[0] + radius * np.cos(theta_grid)
        y = origin[1] + radius * np.sin(theta_grid)
        z = origin[2] + h_grid
    elif axis == 'y':
        x = origin[0] + radius * np.cos(theta_grid)
        z = origin[2] + radius * np.sin(theta_grid)
        y = origin[1] + h_grid
    elif axis == 'x':
        y = origin[1] + radius * np.cos(theta_grid)
        z = origin[2] + radius * np.sin(theta_grid)
        x = origin[0] + h_grid
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def draw_box(ax, xmin, xmax, ymin, ymax, zmin, zmax, color='red', alpha=0.3):
    """Draw a 3D box defined by its bounds.
    Parameters
    ----------
    ax : Axes3D
        3D axis object (e.g., `ax = fig.add_subplot(111, projection='3d')`)
    xmin : float
        Minimum x-coordinate of the box.
    xmax : float
        Maximum x-coordinate of the box.
    ymin : float
        Minimum y-coordinate of the box.
    ymax : float
        Maximum y-coordinate of the box.
    zmin : float
        Minimum z-coordinate of the box.
    zmax : float
        Maximum z-coordinate of the box.
    color : str
        Color of the box surface.
    alpha : float
        Transparency of the box surface (0.0 to 1.0).
    """

    # Define the corners of the box.
    corners = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax]
    ])

    # Vertex indices for each surface of the box.
    faces_idx = [
        [0, 1, 2, 3], [4, 5, 6, 7],  # 下・上
        [0, 1, 5, 4], [2, 3, 7, 6],  # 側面
        [1, 2, 6, 5], [0, 3, 7, 4]
    ]

    faces = [[corners[i] for i in face] for face in faces_idx]

    box = Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=alpha, linewidths=1)
    ax.add_collection3d(box)


def set_axes_equal(ax):
    """Make 3D axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_volume(ax, volume, cylinder: bool = True, box: bool = True):
    """Quicklook of the volume.
    
    Parameters
    ==========
    ax : Axes3D
        3D axis object (e.g., `ax = fig.add_subplot(111, projection='3d')`)
    volume : dict
        Dictionary containing the volume parameters, generated by `compute_inner_volume_from_min_ze_max`.
    cylinder : bool, optional
        If True, draw a cylinder inscribed in the volume. Default is True.
    box : bool, optional
        If True, draw a box inscribed in the volume. Default is True.
    """
    import numpy as np
    from matplotlib.patches import Patch
    print(f"ze_max required: {np.rad2deg(volume['ze_max']):.4f} deg")

    # Legend handles for each plot.
    if cylinder:
        patch_cylinder = [Patch(facecolor='blue', edgecolor='none', label='Cylinder inscribed in the volume', alpha=0.2)]
    else:
        patch_cylinder = []
    if box:
        patch_box = [Patch(facecolor='red', edgecolor='k', label='Box inscribed in the volume', alpha=0.3)]
    else:
        patch_box = []
    handles = [
        Patch(facecolor='green', edgecolor='none', label='Radar scan volume', alpha=0.1),
        *patch_cylinder,
        *patch_box
    ]

    r_min = volume["r_min"]
    
    draw_conical_shell(ax, ze_max=volume["ze_max"], r_min=r_min, r_max=volume["r_max"])
    if cylinder:
        draw_cylinder(ax, origin=(0, 0, r_min), radius=volume["r_in"], height=volume["h_in"], axis='z', color='blue', alpha=0.2)
    if box:
        draw_box(
            ax,
            -volume["x_in"], volume["x_in"],
            -volume["y_in"], volume["y_in"],
            r_min, r_min + volume["h_in"],
            color='red', alpha=0.2)
    set_axes_equal(ax)
    ax.legend(handles=handles)
