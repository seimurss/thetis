"""
bathymetry_utils.py
====================
Analytically generates the tidal inlet bathymetry and interpolates it onto
the Firedrake mesh - no external bathymetry files required.

Domain geometry (Warner et al. 2008, 15 km × 14 km):
  - x : 0 → 15 000 m   (along-shore,  uniform depth)
  - y : 0 → 14 000 m   (cross-shore,  depth varies)

Depth profile (y-only, inferred from tidal_inlet_bathymetry.nc):
  ┌─────────────────────────────────────────────────────┐
  │  y ∈ [0,      y_shelf]   →  h = h_shelf  (flat)    │
  │  y ∈ [y_shelf, y_ocean]  →  h = linear   (ramp)    │
  │  y > y_ocean             →  h = h_ocean  (clamped) │
  └─────────────────────────────────────────────────────┘
  Default values:
    h_shelf  =  4.0 m   (lagoon / inner shelf)
    h_ocean  = 15.0 m   (open ocean)
    y_shelf  =  6800 m  (shelf break / start of slope)
    y_ocean  = 14000 m  (seaward boundary)

Usage (inside inlet.py):
    from bathymetry_utils import generate_bathymetry
    bathymetry_2d = generate_bathymetry(mesh2d, P1_2d)
"""

import numpy as np


def _depth_profile(y_coords,
                   h_shelf=4.0,
                   h_ocean=15.0,
                   y_shelf=6800.0,
                   y_ocean=14000.0):
    """
    Return water depth [m] at each point in y_coords using the piecewise
    linear profile of the tidal inlet domain.

    Parameters
    ----------
    y_coords : array_like
        Cross-shore coordinate values [m].
    h_shelf  : float
        Constant depth over the inner shelf / lagoon [m].
    h_ocean  : float
        Depth at the open-ocean boundary [m].
    y_shelf  : float
        y-coordinate where the slope begins [m].
    y_ocean  : float
        y-coordinate of the ocean boundary [m].

    Returns
    -------
    depth : np.ndarray
        Depth values, same shape as y_coords.
    """
    y = np.asarray(y_coords, dtype=float)
    depth = np.where(
        y <= y_shelf,
        h_shelf,                                                      # flat shelf
        h_shelf + (h_ocean - h_shelf) * (y - y_shelf) / (y_ocean - y_shelf)  # linear ramp
    )
    # clamp to [h_shelf, h_ocean] for any out-of-range nodes
    depth = np.clip(depth, h_shelf, h_ocean)
    return depth


def generate_bathymetry(mesh2d, P1_2d,
                        h_shelf=4.0,
                        h_ocean=15.0,
                        y_shelf=6800.0,
                        y_ocean=14000.0,
                        name='Bathymetry'):
    """
    Generate the tidal inlet bathymetry analytically and populate a
    Firedrake P1 Function on mesh2d.

    The depth depends only on the cross-shore coordinate y  (uniform
    in x), matching the structure of the Warner et al. (2008) test case:
      - flat shelf of h_shelf metres from y = 0 to y = y_shelf
      - linear ramp from h_shelf to h_ocean over y = y_shelf to y_ocean

    Parameters
    ----------
    mesh2d   : firedrake.Mesh
        The 2-D Firedrake mesh.
    P1_2d    : firedrake.FunctionSpace
        CG1 function space on mesh2d.
    h_shelf  : float, optional
        Depth on the shallow (lagoon) side [m]. Default 4.0.
    h_ocean  : float, optional
        Depth at the open-ocean boundary [m]. Default 15.0.
    y_shelf  : float, optional
        y-coordinate where the continental slope begins [m]. Default 6800.
    y_ocean  : float, optional
        y-coordinate of the seaward domain edge [m]. Default 14000.
    name     : str, optional
        Name for the returned Firedrake Function.

    Returns
    -------
    bathymetry_2d : firedrake.Function
        P1 Function populated with analytically computed depth values,
        correctly aligned to every mesh node regardless of any internal
        node reordering performed by Firedrake's Mesh().
    """
    from firedrake import Function

    # mesh2d.coordinates.dat.data has shape (n_nodes, 2): columns are (x, y)
    mesh_coords = mesh2d.coordinates.dat.data
    y_nodes     = mesh_coords[:, 1]  # cross-shore coordinate

    bathy_values = _depth_profile(y_nodes,
                                  h_shelf=h_shelf,
                                  h_ocean=h_ocean,
                                  y_shelf=y_shelf,
                                  y_ocean=y_ocean)

    bathymetry_2d = Function(P1_2d, name=name)
    bathymetry_2d.dat.data[:] = bathy_values
    return bathymetry_2d
