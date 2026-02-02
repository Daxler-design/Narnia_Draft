


import numpy as np
import debug_utils as debug
import json
from pathlib import Path


from sklearn.cluster import KMeans
from scipy.spatial import cKDTree, Voronoi
from typing import Tuple, Optional, List
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt

def compute_voronoi_sdf(shape: Tuple[int, int], centroids: np.ndarray) -> np.ndarray:
    """
    Compute a Voronoi-based scalar field: value = dist_to_2nd_nearest - dist_to_nearest.
    
    Creates ridge patterns at Voronoi cell boundaries where value is maximized.
    
    Args:
        shape: Output field shape (ny, nx)
        centroids: Array of centroid positions (k, 2) in [y, x] coordinates
    
    Returns:
        2D array (ny, nx) with Voronoi SDF values
        - Negative values near cell boundaries (ridges = material)
        - Positive values inside cells (void)
        - Zero at the ridge boundary surface
    
    Example:
        >>> centroids = np.array([[10, 10], [40, 40], [10, 40]])
        >>> sdf = compute_voronoi_sdf((50, 50), centroids)
        >>> sdf.shape
        (50, 50)
    
    Note:
        - Uses cKDTree for efficient nearest-neighbor queries
        - Output range depends on centroid spacing
        - Ridges occur where distances to two nearest centroids are similar
    """
    ny, nx = shape
    Y, X = np.indices(shape)
    grid_points = np.stack([Y.ravel(), X.ravel()], axis=-1)
    tree = cKDTree(centroids)
    dists, _ = tree.query(grid_points, k=2)
    # Convert to SDF: negate so ridges (d2≈d1) become negative (material)
    # Add small thickness (0.5) so ridge centers are at -0.5
    sdf_flat = -(dists[:, 1] - dists[:, 0]) + 0.5
    return sdf_flat.reshape(shape)




def poly_to_true_sdf(poly_xyz, bbox_min, bbox_max, nx=256, ny=256):
    
    """
    poly_xyz: (m,3) polyline points, assumed closed or will be closed
    bbox_min/max: [x,y,z] from json
    returns: phi (ny,nx) signed distance in WORLD UNITS
    """
    poly_xyz = np.asarray(poly_xyz, dtype=float)
    poly_xy = poly_xyz[:, :2]

    # ensure closed
    if np.linalg.norm(poly_xy[0] - poly_xy[-1]) > 1e-9:
        poly_xy = np.vstack([poly_xy, poly_xy[0]])

    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # inside mask via point-in-polygon
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    inside = MplPath(poly_xy).contains_points(pts).reshape(ny, nx)

    # EDT (true distance in world units because sampling=(dy,dx))
    dist_in  = distance_transform_edt(inside,  sampling=(dy, dx))
    dist_out = distance_transform_edt(~inside, sampling=(dy, dx))

    # signed: inside negative, outside positive
    phi = dist_out - dist_in
    return phi

def load_sdf_list_from_inshapes(json_path, nx=256, ny=256, branch_index=0):
    """
    Returns: list of SDFs [phi0, phi1, ...] for shapes[branch_index]["polys"]
    """
    data = json.loads(Path(json_path).read_text())
    bbox_min = data["bbox"]["minbb"]
    bbox_max = data["bbox"]["maxbb"]

    polys = data["shapes"][branch_index]["polys"]
    sdfs = [poly_to_true_sdf(poly, bbox_min, bbox_max, nx=nx, ny=ny) for poly in polys]
    return sdfs

def interpolate_true_sdf_profiles_world(
    sdf_profiles: List[np.ndarray],
    bbox_min: List[float],
    bbox_max: List[float],
    target_count: int,
    method: str = "linear",
    redistance: bool = True,
) -> np.ndarray:
    """
    Input: sdf_profiles (list of np.ndarray), bbox_min (list[float]),
           bbox_max (list[float]), target_count (int), method (str), redistance (bool)
    Output: np.ndarray, shape (target_count, ny, nx) true SDF in world units.
    """
    if target_count <= 0:
        raise ValueError("target_count must be > 0.")

    stack = np.asarray(sdf_profiles, dtype=float)
    if stack.ndim != 3:
        raise ValueError("sdf_profiles must be a list of 2D arrays (num_slices, ny, nx).")

    source_count, ny, nx = stack.shape
    if source_count == 0:
        raise ValueError("sdf_profiles is empty.")

    if method != "linear":
        raise ValueError("Only linear interpolation is supported for now.")

    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])
    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 0.0
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 0.0

    def _redistance(slice_2d: np.ndarray) -> np.ndarray:
        mask = slice_2d <= 0.0
        dist_in = distance_transform_edt(mask, sampling=(dy, dx))
        dist_out = distance_transform_edt(~mask, sampling=(dy, dx))
        return dist_out - dist_in

    if target_count == source_count:
        return stack.copy()

    new_stack = np.zeros((target_count, ny, nx), dtype=float)
    if target_count == 1:
        interp = stack[0]
        new_stack[0] = _redistance(interp) if redistance else interp
        return new_stack

    for i in range(target_count):
        pos = (i / float(target_count - 1)) * (source_count - 1)
        i0 = int(np.floor(pos))
        i1 = int(np.ceil(pos))
        if i0 == i1:
            interp = stack[i0]
        else:
            t = pos - i0
            interp = (1.0 - t) * stack[i0] + t * stack[i1]
        new_stack[i] = _redistance(interp) if redistance else interp

    print(
        f"Interpolate (method={method}, target_count={target_count}) "
        f"slices: {source_count} -> {new_stack.shape[0]}"
    )
    return new_stack


def interpolate_true_sdf_profiles_insert_between(
    sdf_profiles: List[np.ndarray],
    bbox_min: List[float],
    bbox_max: List[float],
    insert_count: int = 1,
    redistance: bool = True,
) -> np.ndarray:
    """
    Insert interpolated slices between neighbors without reparameterizing endpoints.
    """
    stack = np.asarray(sdf_profiles, dtype=float)
    if stack.ndim != 3:
        raise ValueError("sdf_profiles must be a list of 2D arrays (num_slices, ny, nx).")

    source_count, ny, nx = stack.shape
    if source_count == 0:
        raise ValueError("sdf_profiles is empty.")

    insert_count = int(insert_count)
    if insert_count <= 0:
        print(
            f"Interpolate (method=insert_between, insert_count=0) "
            f"slices: {source_count} -> {source_count}"
        )
        return stack.copy()

    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])
    dx = (xmax - xmin) / (nx - 1) if nx > 1 else 0.0
    dy = (ymax - ymin) / (ny - 1) if ny > 1 else 0.0

    def _redistance(slice_2d: np.ndarray) -> np.ndarray:
        mask = slice_2d <= 0.0
        dist_in = distance_transform_edt(mask, sampling=(dy, dx))
        dist_out = distance_transform_edt(~mask, sampling=(dy, dx))
        return dist_out - dist_in

    new_slices = []
    for i in range(source_count - 1):
        s0 = stack[i]
        s1 = stack[i + 1]
        new_slices.append(s0)
        for j in range(1, insert_count + 1):
            t = j / float(insert_count + 1)
            interp = (1.0 - t) * s0 + t * s1
            new_slices.append(_redistance(interp) if redistance else interp)
    new_slices.append(stack[-1])

    new_stack = np.stack(new_slices, axis=0)
    print(
        f"Interpolate (method=insert_between, insert_count={insert_count}) "
        f"slices: {source_count} -> {new_stack.shape[0]}"
    )
    return new_stack


# --------------------------
# Bracing Generation Voronoi 
# --------------------------

def generate_centroids(mask: np.ndarray, k: int = 3, prev_centroids: Optional[np.ndarray] = None, seed: int = 42) -> np.ndarray:
    """
    Generate k centroids for the given boolean mask using K-Means clustering.
    
    Args:
        mask: Boolean mask (ny, nx) indicating valid region
        k: Number of centroids to generate
        prev_centroids: Previous centroids for temporal coherence (optional)
        seed: Random seed for K-means initialization
    
    Returns:
        Array of centroids with shape (k, 2) in [y, x] coordinates
    
    Example:
        >>> mask = np.ones((50, 50), dtype=bool)
        >>> centroids = generate_centroids(mask, k=5)
        >>> centroids.shape
        (5, 2)
    
    Note:
        - If mask has fewer points than k, returns prev_centroids or default center
        - Uses k-means++ initialization for better distribution
        - Coordinates are in image space (row, col) = (y, x)
    """
    coords = np.argwhere(mask)
    if len(coords) < k:
        if prev_centroids is not None:
            return prev_centroids
        else:
            ny, nx = mask.shape
            return np.array([[ny/2, nx/2]] * k)

    if prev_centroids is not None:
        # Validate prev_centroids shape
        if prev_centroids.shape[0] == k:
            # Perfect match - use as init
            kmeans = KMeans(n_clusters=k, init=prev_centroids, n_init=1, random_state=seed)
        else:
            # Mismatch - don't use warm start, let k-means++ handle it
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
    else:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
        
    kmeans.fit(coords)
    return kmeans.cluster_centers_



def constrain_centroids_to_mask(centroids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Ensure centroids are strictly inside the mask by snapping to nearest valid point.
    
    Args:
        centroids: Array of centroid positions (k, 2) in [y, x] coordinates
        mask: Boolean mask (ny, nx) indicating valid region
    
    Returns:
        Constrained centroids array (k, 2)
    
    Example:
        >>> mask = np.zeros((50, 50), dtype=bool)
        >>> mask[10:40, 10:40] = True  # Valid region
        >>> centroids = np.array([[5, 5], [25, 25]])  # One outside, one inside
        >>> constrained = constrain_centroids_to_mask(centroids, mask)
        >>> mask[int(constrained[0, 0]), int(constrained[0, 1])]  # Now inside
        True
    
    Note:
        - Centroids already inside mask are unchanged
        - Outside centroids snap to nearest valid point
        - Returns input unchanged if mask is empty
    """
    ny, nx = mask.shape
    valid_coords = np.argwhere(mask)
    if len(valid_coords) == 0:
        return centroids
    constrained = []
    for c in centroids:
        y, x = int(c[0]), int(c[1])
        if 0 <= y < ny and 0 <= x < nx and mask[y, x]:
            constrained.append(c)
        else:
            dists = np.sum((valid_coords - c)**2, axis=1)
            nearest_idx = np.argmin(dists)
            constrained.append(valid_coords[nearest_idx])
    return np.array(constrained)

# convert vornoroi field to a true sdf

def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute minimum distance from point(s) p to line segment ab.
    
    Args:
        p: Points array (n, 2)
        a: Segment start (2,)
        b: Segment end (2,)
    
    Returns:
        Array of distances (n,)
    """
    ab = b - a  # (2,)
    ap = p - a  # (n, 2)
    ab_len_sq = np.dot(ab, ab)  # scalar
    
    if ab_len_sq < 1e-10:  # degenerate segment
        return np.linalg.norm(ap, axis=-1)
    
    # Project p onto line ab: t = (ap · ab) / |ab|²
    # ap is (n, 2), ab is (2,) -> use matrix multiplication
    t = np.sum(ap * ab, axis=-1) / ab_len_sq  # (n,)
    t = np.clip(t, 0, 1)  # clamp to segment
    
    # Nearest point on segment: a + t*ab
    nearest = a + t[:, np.newaxis] * ab  # (n, 2)
    return np.linalg.norm(p - nearest, axis=-1)  # (n,)


def compute_voronoi_ridge_sdf(shape: Tuple[int, int], centroids: np.ndarray, ridge_thickness: float = 1.0) -> np.ndarray:
    """
    Compute a true signed distance field to Voronoi cell ridges.
    
    Creates ridge patterns at Voronoi cell boundaries with accurate distance values.
    
    Args:
        shape: Output field shape (ny, nx)
        centroids: Array of centroid positions (k, 2) in [y, x] coordinates
        ridge_thickness: Half-width of the ridge (default: 1.0)
    
    Returns:
        2D array (ny, nx) with true SDF values
        - Negative values inside ridges (material)
        - Positive values outside ridges (void)
        - Zero at ridge boundary surface
        - Unit gradient: |∇SDF| ≈ 1
    
    Example:
        >>> centroids = np.array([[10, 10], [40, 40], [10, 40]])
        >>> sdf = compute_voronoi_ridge_sdf((50, 50), centroids, ridge_thickness=2.0)
        >>> sdf.shape
        (50, 50)
    
    Note:
        - Uses scipy.spatial.Voronoi for accurate ridge geometry
        - Computes actual distance to nearest ridge segment
        - More expensive than compute_voronoi_sdf but geometrically correct
        - Ridge thickness controls the width of the material zone
    """
    ny, nx = shape
    
    # Compute Voronoi diagram
    vor = Voronoi(centroids)
    
    # Extract finite ridge segments
    ridge_segments = []
    for ridge_vertices in vor.ridge_vertices:
        if -1 not in ridge_vertices:  # Skip infinite ridges
            v0 = vor.vertices[ridge_vertices[0]]
            v1 = vor.vertices[ridge_vertices[1]]
            ridge_segments.append((v0, v1))
    
    if not ridge_segments:
        # No finite ridges - return all positive (void)
        return np.ones(shape) * ridge_thickness
    
    # Create grid points
    Y, X = np.indices(shape)
    grid_points = np.stack([Y.ravel(), X.ravel()], axis=-1)
    
    # Compute distance to nearest ridge segment
    min_dists = np.full(len(grid_points), np.inf)
    
    for v0, v1 in ridge_segments:
        dists = point_to_segment_distance(grid_points, v0, v1)
        min_dists = np.minimum(min_dists, dists)
    
    # Convert to signed distance (negative inside ridge, positive outside)
    sdf_flat = min_dists - ridge_thickness
    
    return sdf_flat.reshape(shape)



def voronoi_ridge_band_sdf_world(shape, centroids_px, bbox_min, bbox_max, sigma_world):
    ny, nx = shape
    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # centroids px (y,x) -> world (x,y)
    cx = xmin + (centroids_px[:, 1] / (nx - 1)) * (xmax - xmin)
    cy = ymin + (centroids_px[:, 0] / (ny - 1)) * (ymax - ymin)
    pts = np.stack([cx, cy], axis=1)  # (x,y) world

    # grid world (x,y)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    Xw, Yw = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([Xw.ravel(), Yw.ravel()], axis=-1)

    # nearest two distances in world
    tree = cKDTree(pts)
    dists, _ = tree.query(grid, k=2)  # (N,2), dists[:,0]=d1, dists[:,1]=d2
    d1, d2 = dists[:, 0], dists[:, 1]

    # ridge band: (d2 - d1)/2 <= sigma_world
    band = (d2 - d1) <= (2.0 * sigma_world)
    band = band.reshape((ny, nx))

    # true signed distance to band boundary (world units)
    dist_in  = distance_transform_edt(band,  sampling=(dy, dx))
    dist_out = distance_transform_edt(~band, sampling=(dy, dx))
    phi = dist_out - dist_in  # inside negative, outside positive
    return phi


def sigma_list_generate(min_sigma, max_sigma, num_slices, method="linear"):
    """
    Generate a sigma list with smooth interpolation.
    method: "linear", "exp", "ease", "cosine"
    """
    if num_slices <= 0:
        return []
    if num_slices == 1:
        return [float(min_sigma)]

    min_sigma = float(min_sigma)
    max_sigma = float(max_sigma)
    t = np.linspace(0.0, 1.0, num_slices)
    m = str(method).lower()

    if m == "linear":
        w = t
    elif m in ("exp", "exponential"):
        k = 3.0
        w = (np.exp(k * t) - 1.0) / (np.exp(k) - 1.0)
    elif m in ("ease", "ease_in_out"):
        w = t * t * (3.0 - 2.0 * t)
    elif m == "cosine":
        w = 0.5 - 0.5 * np.cos(np.pi * t)
    else:
        raise ValueError(f"Unknown method '{method}'. Use linear, exp, ease, or cosine.")

    sigmas = min_sigma + (max_sigma - min_sigma) * w
    return sigmas.tolist()


        
def generate_bracing_cavity_static_world(sdf_profiles, iso_level_offset, nx, ny, k, seed=42, sigma_world=None, debug_interval=10, debugOutput=False):
    """
    Docstring for generate_bracing_cavity_static_world
    
    :param sdf_profiles: Original SDF profiles
    2D array (num_slices, ny, nx)
    :param iso_level_offset: world distance (only for true sdf) offset to apply to the original SDF profiles, positive value is offset inward
    :param nx: shape x dimension,aligned with sdf_profiles
    
    :param ny: shape y dimension , aligned with sdf_profiles
    :param k: centroid count per slice
    :param seed: Random seed for centroid generation
    :param sigma_world: real world units for voronoi ridge band thickness
                         - float: constant for all slices
                         - list/tuple: interpolated across slices
                           * [v0, v1, ...] -> evenly spaced keyframes
                           * [(idx, v), ...] -> explicit keyframes by slice index
    :param debug_interval: Interval for debug output (in slices)
    :param debugOutput: Whether to output debug plots

    :return: bracing cavities SDF profiles
    """
    
    num_slice = len(sdf_profiles)
    bracing_cavities_sdf = np.zeros((num_slice, ny, nx))
    prev_centroids = None

    for i in range(num_slice):
        slice_2d = sdf_profiles[i]
        
        # use profile to creat mask
        mask = slice_2d <= 0.0

        centroids = generate_centroids(mask, k=k, prev_centroids=None, seed=seed)
        centroids = constrain_centroids_to_mask(centroids, mask)
        
        if isinstance(sigma_world, (list, tuple, np.ndarray)):
            sigma_i = sigma_world[i]
        else:
            sigma_i = sigma_world
        V= voronoi_ridge_band_sdf_world(slice_2d.shape,
                                        centroids_px=centroids,
                                        bbox_min=bbox_min,
                                        bbox_max=bbox_max,
                                        sigma_world=sigma_i)
        
        profile_slice_offset = slice_2d + iso_level_offset
        bracing_cavity = np.maximum(profile_slice_offset, -V)
        bracing_cavities_sdf[i] = bracing_cavity

        if debugOutput and (i % debug_interval ==0):

            debug.output_debug_plot_world_offsets(
            slice_2d, 
            bbox_min=bbox_min, bbox_max=bbox_max, nx=nx, ny=ny, slice_idx=i,
            offset_step_world=0.05,offset_max_world=0.5,
            file_name_prefix="profile_sdf")

            debug.output_debug_plot_world_offsets(
            V, 
            bbox_min=bbox_min, bbox_max=bbox_max, nx=nx, ny=ny, slice_idx=i,
            offset_step_world=0.001,offset_max_world=0.001,
            file_name_prefix="Voronoi_sdf")

            
            debug.output_debug_plot_world_offsets(
                bracing_cavity, 
                bbox_min=bbox_min, bbox_max=bbox_max, nx=nx, ny=ny, slice_idx=i,
                centroids=centroids,
                offset_step_world=0.01,
                file_name_prefix="bracing_cavity_sdf")

    return bracing_cavities_sdf

# --------------------------
# Bracing Generation Curve Guides
# --------------------------

"""
This method will introduce n curves as bracing gudies defined in inShapes.json, 
in each slice read from inShapes.json, there will be coresponding bracing guide cuvres,
logic is simple, for each slice, we will create true sdfs based on world units.

profile_sdf from the profiles curves,
profile_sdf_offset = profile_sdf + iso_level_offset (positive value is offset inward), this offset value reflects the thin wall thickness (m) we want to keep
bracing_ridge_sdf from the bracing ridge guide curve, with ridge thickness defined in world units (m), this should be masked inside the profile_sdf_offset area
bracing_cavity_sdf = np.maximum(profile_sdf_offset, -bracing_ridge_sdf)



"""

def _normalize_polys_to_slices(entries, branch_index=None):
    """
    Normalize inShapes JSON entries into a list of slices,
    where each slice is a list of polylines (each polyline is a list of points).
    """
    if not entries:
        return []

    if branch_index is not None:
        entries = [entries[branch_index]]

    if len(entries) == 1:
        polys = entries[0].get("polys", [])
        if not polys:
            return []
        first = polys[0]
        if first and isinstance(first[0], (int, float)):
            return [[polys]]
        return [[poly] for poly in polys]

    return [entry.get("polys", []) for entry in entries]


def load_profile_sdf_slices_from_inshapes(json_path, nx=256, ny=256, branch_index=None):
    """
    Load profile polylines from inShapes.json and return per-slice true SDFs.
    """
    data = json.loads(Path(json_path).read_text())
    bbox_min = data["bbox"]["minbb"]
    bbox_max = data["bbox"]["maxbb"]

    slice_polys = _normalize_polys_to_slices(data.get("shapes", []), branch_index)
    if not slice_polys:
        raise ValueError("No shape polylines found in inShapes.json.")

    sdfs = []
    for polys in slice_polys:
        if not polys:
            raise ValueError("Encountered empty shape slice in inShapes.json.")
        if len(polys) == 1:
            sdf = poly_to_true_sdf(polys[0], bbox_min, bbox_max, nx=nx, ny=ny)
        else:
            sdf_list = [poly_to_true_sdf(poly, bbox_min, bbox_max, nx=nx, ny=ny) for poly in polys]
            sdf = np.minimum.reduce(sdf_list)
        sdfs.append(sdf)
    return sdfs, bbox_min, bbox_max


def load_ridge_polylines_from_inshapes(json_path, branch_index=None):
    """
    Load ridge guide polylines from inShapes.json and return per-slice lists.
    """
    data = json.loads(Path(json_path).read_text())
    return _normalize_polys_to_slices(data.get("ridge", []), branch_index)


def polyline_ridge_sdf_world(polys, bbox_min, bbox_max, nx=256, ny=256, ridge_thickness=0.02):
    """
    Compute true SDF of ridge polylines with world-unit thickness.
    Negative inside ridge band, positive outside.
    """
    if not polys:
        return np.ones((ny, nx), dtype=float) * float(ridge_thickness)

    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    Xw, Yw = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([Xw.ravel(), Yw.ravel()], axis=-1)

    min_dists = np.full(grid.shape[0], np.inf, dtype=float)
    for poly in polys:
        pts = np.asarray(poly, dtype=float)[:, :2]
        if pts.shape[0] < 2:
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(a - b) < 1e-12:
                continue
            dists = point_to_segment_distance(grid, a, b)
            min_dists = np.minimum(min_dists, dists)

    if not np.isfinite(min_dists).any():
        return np.ones((ny, nx), dtype=float) * float(ridge_thickness)

    sdf_flat = min_dists - float(ridge_thickness)
    return sdf_flat.reshape((ny, nx))


def _extend_point_to_bbox_2d(point_xy, direction_xy, bbox_min, bbox_max, eps=1e-9):
    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    if abs(dx) < eps and abs(dy) < eps:
        return np.asarray(point_xy, dtype=float)

    candidates = []
    if abs(dx) >= eps:
        for x in (xmin, xmax):
            t = (x - point_xy[0]) / dx
            if t > eps:
                y = point_xy[1] + t * dy
                if ymin - eps <= y <= ymax + eps:
                    candidates.append((t, x, y))
    if abs(dy) >= eps:
        for y in (ymin, ymax):
            t = (y - point_xy[1]) / dy
            if t > eps:
                x = point_xy[0] + t * dx
                if xmin - eps <= x <= xmax + eps:
                    candidates.append((t, x, y))

    if not candidates:
        return np.asarray(point_xy, dtype=float)

    _, x_hit, y_hit = min(candidates, key=lambda c: c[0])
    return np.array([x_hit, y_hit], dtype=float)


def extend_polyline_to_bbox(poly_xyz, bbox_min, bbox_max, eps=1e-9):
    """
    Extend polyline endpoints to the bbox along the tangent directions.
    """
    pts = np.asarray(poly_xyz, dtype=float)
    if pts.shape[0] < 2:
        return poly_xyz

    def _first_non_degenerate(start, step):
        idx = start + step
        while 0 <= idx < pts.shape[0]:
            if np.linalg.norm(pts[start, :2] - pts[idx, :2]) > eps:
                return pts[idx, :2]
            idx += step
        return None

    start_next = _first_non_degenerate(0, 1)
    end_prev = _first_non_degenerate(pts.shape[0] - 1, -1)

    start_dir = pts[0, :2] - start_next if start_next is not None else np.array([0.0, 0.0])
    end_dir = pts[-1, :2] - end_prev if end_prev is not None else np.array([0.0, 0.0])

    new_start_xy = _extend_point_to_bbox_2d(pts[0, :2], start_dir, bbox_min, bbox_max)
    new_end_xy = _extend_point_to_bbox_2d(pts[-1, :2], end_dir, bbox_min, bbox_max)

    new_pts = pts.copy()
    if np.linalg.norm(new_start_xy - pts[0, :2]) > eps:
        z = pts[0, 2] if pts.shape[1] > 2 else 0.0
        new_pts = np.vstack([np.array([new_start_xy[0], new_start_xy[1], z]), new_pts])
    if np.linalg.norm(new_end_xy - pts[-1, :2]) > eps:
        z = pts[-1, 2] if pts.shape[1] > 2 else 0.0
        new_pts = np.vstack([new_pts, np.array([new_end_xy[0], new_end_xy[1], z])])

    return new_pts.tolist()


def generate_bracing_cavity_guided_world(
    sdf_profiles,
    ridge_sdf_slices,
    bbox_min,
    bbox_max,
    nx,
    ny,
    iso_level_offset,
    debug_interval=10,
    debugOutput=False,
):
    """
    Generate bracing cavities using precomputed ridge SDF slices (world units).
    """
    num_slice = len(sdf_profiles)
    bracing_cavities_sdf = np.zeros((num_slice, ny, nx))

    if ridge_sdf_slices is None or len(ridge_sdf_slices) == 0:
        ridge_sdf_slices = [np.zeros((ny, nx), dtype=float) for _ in range(num_slice)]
    elif len(ridge_sdf_slices) != num_slice:
        raise ValueError(
            f"ridge_sdf_slices length {len(ridge_sdf_slices)} does not match num_slice {num_slice}."
        )

    for i in range(num_slice):
        slice_2d = sdf_profiles[i]
        ridge_sdf = np.asarray(ridge_sdf_slices[i], dtype=float)

        profile_slice_offset = slice_2d + iso_level_offset
        bracing_cavity = np.maximum(profile_slice_offset, -ridge_sdf)
        bracing_cavities_sdf[i] = bracing_cavity

        if debugOutput and (i % debug_interval == 0):
            debug.output_debug_plot_world_offsets(
                ridge_sdf,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                nx=nx,
                ny=ny,
                slice_idx=i,
                offset_step_world=0.01,
                offset_max_world=0.01,
                file_name_prefix="ridge_sdf",
            )

            debug.output_debug_plot_world_offsets(
                bracing_cavity,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                nx=nx,
                ny=ny,
                slice_idx=i,
                offset_step_world=0.01,
                file_name_prefix="bracing_cavity_sdf",
            )

    return bracing_cavities_sdf



# --------------------------
# IO utils
# --------------------------
def save_npz_for_gui(
    output_path,
    result_fields_3d,
    profile_fields_3d=None,
    bracing_fields_3d=None,
    bounds_min=None,
    bounds_max=None,
    iso_level=0.0,
    total_height=None,
):
    """
    Save SDF stacks to NPZ matching the GUI's expected format.
    """
    result_fields_3d = np.asarray(result_fields_3d)
    if result_fields_3d.ndim != 3:
        raise ValueError("result_fields_3d must be (num_slices, ny, nx)")

    num_slices = result_fields_3d.shape[0]
    result_fields_2d = result_fields_3d.reshape(num_slices, -1)

    safe_total_height = None
    if total_height is not None:
        try:
            safe_total_height = float(total_height)
        except (TypeError, ValueError):
            safe_total_height = None
    if safe_total_height is None or safe_total_height <= 0.0:
        safe_total_height = float(max(num_slices - 1, 1))

    save_dict = {
        "result_fields": result_fields_2d,
        "iso_level": float(iso_level),
        "bounds_min": np.array(bounds_min) if bounds_min is not None else None,
        "bounds_max": np.array(bounds_max) if bounds_max is not None else None,
        "total_height": safe_total_height,
        "slice_count": int(num_slices),
    }

    if profile_fields_3d is not None:
        profile_fields_3d = np.asarray(profile_fields_3d)
        save_dict["profile_fields"] = profile_fields_3d.reshape(num_slices, -1)

    if bracing_fields_3d is not None:
        bracing_fields_3d = np.asarray(bracing_fields_3d)
        save_dict["bracing_fields"] = bracing_fields_3d.reshape(num_slices, -1)

    np.savez(output_path, **save_dict)




# ------------------------------------------------------------------------------


# 1. need convert polylines to true sdfs
nx = 512
ny = 512
sdf_profiles, bbox_min, bbox_max = load_profile_sdf_slices_from_inshapes(
    "alice_result/inShapes.json",
    nx=nx,
    ny=ny,
)
print(f"Loaded {len(sdf_profiles)} SDF profiles from inShapes.json \n")
# pre-intepolate to target number of slices
use_insert_interpolation = False
insert_between = 1
target_num_slices = len(sdf_profiles)  # set when use_insert_interpolation is False

if use_insert_interpolation:
    if target_num_slices is not None:
        raise ValueError("insert_between is enabled; set target_num_slices=None.")
    sdf_profiles = interpolate_true_sdf_profiles_insert_between(
        sdf_profiles,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        insert_count=insert_between,
        redistance=True,
    )
else:
    if target_num_slices is None:
        raise ValueError("target_num_slices is required when insert_between is disabled.")
    sdf_profiles = interpolate_true_sdf_profiles_world(
        sdf_profiles,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        target_count=target_num_slices,
        redistance=True,
    )

print(f"Loaded {len(sdf_profiles)} SDF profiles interpolated \n")
for i, sdf in enumerate(sdf_profiles):
    if i % 10 == 0:
        # check sdf profiles and contours
        print(f"sdf {i} is a type of {type(sdf)}, shape: {sdf.shape}, min: {sdf.min()}, max: {sdf.max()}")
        debug.output_debug_plot_world_offsets(
            sdf, 
            bbox_min=bbox_min, bbox_max=bbox_max, nx=nx, ny=ny, slice_idx=i,
            offset_step_world=0.05,offset_max_world=0.5,
            file_name_prefix="profile_sdf")
        

# 2. generate bracing cavitys sdf profiles_ voronoi static method

# sigma_values = sigma_list_generate(min_sigma=0.02, max_sigma=0.035, num_slices=len(sdf_profiles), method="ease")

# bracing_cavitys_sdf = generate_bracing_cavity_static_world(
#     sdf_profiles,
#     iso_level_offset=0.02,
#     nx=nx,
#     ny=ny,
#     k=5,
#     seed=42,
#     sigma_world=sigma_values,
#     debug_interval=10,
#     debugOutput=False)

# --------------------------------------------------------        
# 2. generate bracing cavitys sdf profiles (ridge guide method)
ridge_polys_slices = load_ridge_polylines_from_inshapes("alice_result/inShapes.json")
ridge_thickness_values = sigma_list_generate(
    min_sigma=0.03,
    max_sigma=0.035,
    num_slices=len(ridge_polys_slices),
    method="ease",
)

ridge_sdf_slices = []
for i, polys in enumerate(ridge_polys_slices):
    extended_polys = [extend_polyline_to_bbox(poly, bbox_min, bbox_max) for poly in polys]
    if isinstance(ridge_thickness_values, (list, tuple, np.ndarray)):
        thickness_i = ridge_thickness_values[i]
    else:
        thickness_i = ridge_thickness_values
    ridge_sdf_slices.append(
        polyline_ridge_sdf_world(
            extended_polys,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            nx=nx,
            ny=ny,
            ridge_thickness=thickness_i,
        )
    )

# we need extend the ridge to the borders of the field, so we can get correct bracing cavitys sdf profiles after interpolation


if ridge_sdf_slices:
    if use_insert_interpolation:
        ridge_sdf_slices = interpolate_true_sdf_profiles_insert_between(
            ridge_sdf_slices,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            insert_count=insert_between,
            redistance=True,
        )
    else:
        ridge_sdf_slices = interpolate_true_sdf_profiles_world(
            ridge_sdf_slices,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            target_count=target_num_slices,
            redistance=True,
        )

ridge_thickness_values = sigma_list_generate(
    min_sigma=0.03/2,
    max_sigma=0.035/2,
    num_slices=len(ridge_sdf_slices),
    method="ease",
)

print(f"Loaded {len(ridge_sdf_slices)} ridge SDF slices \n")

bracing_cavitys_sdf = generate_bracing_cavity_guided_world(
    sdf_profiles,
    ridge_sdf_slices,
    bbox_min=bbox_min,
    bbox_max=bbox_max,
    nx=nx,
    ny=ny,
    iso_level_offset=0.02,
    debug_interval=10,
    debugOutput=False,
)

bracing_sdf_result = np.maximum(np.asarray(sdf_profiles), -bracing_cavitys_sdf)

# 3. output thin wall profile with bracing cavity sdf profiles (debug previews)
for i in range(len(bracing_sdf_result)):
    if i % 10 == 0:
        bracing_slice = bracing_sdf_result[i]
        debug.output_debug_plot_world_offsets(
            bracing_slice,
            bbox_min=bbox_min, bbox_max=bbox_max, nx=nx, ny=ny, slice_idx=i,
            offset_step_world=0.01,
            show_offset=False,
            file_name_prefix="bracing_field")

# 4. save NPZ for GUI viewer
save_npz_for_gui(
    output_path="output/debug_bracing_results.npz",
    result_fields_3d=bracing_sdf_result,
    profile_fields_3d=sdf_profiles,
    bracing_fields_3d=bracing_cavitys_sdf,
    bounds_min=bbox_min,
    bounds_max=bbox_max,
    iso_level=0.0,
    total_height=0.0,# z is not needed in this case
)



# grasshopper CPython helper: load a slice from NPZ and return Rhino.Geometry curves
def npz_slice_to_rhino_curves(
    npz_path,
    field="result_fields",
    slice_idx=0,
    iso_level=0.0,
):
    """
    Grasshopper CPython helper: load a slice from NPZ and return Rhino.Geometry curves.
    Returns a list of Rhino.Geometry.PolylineCurve.
    """
    import numpy as _np
    import Rhino.Geometry as _rg #type: ignore

    data = _np.load(npz_path, allow_pickle=True)
    fields = data[field]
    num_slices, values_per_field = fields.shape
    n = int(_np.sqrt(values_per_field))
    if n * n != values_per_field:
        raise ValueError(f"Cannot infer square grid from {values_per_field} values.")
    nx = ny = n

    if slice_idx < 0 or slice_idx >= num_slices:
        raise IndexError(f"slice_idx {slice_idx} out of range 0..{num_slices - 1}")

    bmin = data.get("bounds_min", _np.array([0.0, 0.0, 0.0]))
    bmax = data.get("bounds_max", _np.array([1.0, 1.0, 1.0]))
    xmin, ymin = float(bmin[0]), float(bmin[1])
    xmax, ymax = float(bmax[0]), float(bmax[1])

    slice_2d = fields[slice_idx].reshape((ny, nx))

    def _interp(p1, p2, v1, v2):
        if abs(v2 - v1) < 1e-12:
            t = 0.5
        else:
            t = (iso_level - v1) / (v2 - v1)
        return p1 + t * (p2 - p1)

    # Marching squares edges per cell
    segments = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            v00 = slice_2d[iy, ix]
            v10 = slice_2d[iy, ix + 1]
            v11 = slice_2d[iy + 1, ix + 1]
            v01 = slice_2d[iy + 1, ix]

            case = 0
            if v00 >= iso_level:
                case |= 1
            if v10 >= iso_level:
                case |= 2
            if v11 >= iso_level:
                case |= 4
            if v01 >= iso_level:
                case |= 8
            if case == 0 or case == 15:
                continue

            # cell corner coords in grid space
            p00 = _np.array([ix, iy], dtype=float)
            p10 = _np.array([ix + 1, iy], dtype=float)
            p11 = _np.array([ix + 1, iy + 1], dtype=float)
            p01 = _np.array([ix, iy + 1], dtype=float)

            # edge interpolation points
            e0 = _interp(p00, p10, v00, v10)  # top
            e1 = _interp(p10, p11, v10, v11)  # right
            e2 = _interp(p11, p01, v11, v01)  # bottom
            e3 = _interp(p01, p00, v01, v00)  # left

            # case table (segments)
            if case in (1, 14):
                segments.append((e3, e0))
            elif case in (2, 13):
                segments.append((e0, e1))
            elif case in (3, 12):
                segments.append((e3, e1))
            elif case in (4, 11):
                segments.append((e1, e2))
            elif case in (5, 10):
                segments.append((e3, e0))
                segments.append((e1, e2))
            elif case in (6, 9):
                segments.append((e0, e2))
            elif case in (7, 8):
                segments.append((e3, e2))

    if not segments:
        return []

    # Map grid coords to world XY
    def _to_world(pt):
        x = xmin + (pt[0] / (nx - 1)) * (xmax - xmin)
        y = ymin + (pt[1] / (ny - 1)) * (ymax - ymin)
        return _rg.Point3d(x, y, 0.0)

    # Stitch segments into polylines
    tol = 1e-6
    def _key(pt):
        return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

    seg_map = {}
    for a, b in segments:
        ka = _key(a)
        kb = _key(b)
        seg_map.setdefault(ka, []).append(b)
        seg_map.setdefault(kb, []).append(a)

    curves = []
    visited = set()
    for a, b in segments:
        ka = _key(a)
        kb = _key(b)
        if (ka, kb) in visited or (kb, ka) in visited:
            continue

        poly = [a, b]
        visited.add((ka, kb))

        # extend forward
        while True:
            last = poly[-1]
            kl = _key(last)
            neighbors = seg_map.get(kl, [])
            next_pt = None
            for npt in neighbors:
                kn = _key(npt)
                if (kl, kn) not in visited and (kn, kl) not in visited:
                    next_pt = npt
                    visited.add((kl, kn))
                    break
            if next_pt is None:
                break
            poly.append(next_pt)

        # extend backward
        while True:
            first = poly[0]
            kf = _key(first)
            neighbors = seg_map.get(kf, [])
            next_pt = None
            for npt in neighbors:
                kn = _key(npt)
                if (kn, kf) not in visited and (kf, kn) not in visited:
                    next_pt = npt
                    visited.add((kn, kf))
                    break
            if next_pt is None:
                break
            poly.insert(0, next_pt)

        if len(poly) >= 2:
            pts = [_to_world(p) for p in poly]
            curves.append(_rg.PolylineCurve(pts))

    return curves


