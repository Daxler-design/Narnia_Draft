


import numpy as np
import core
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




# ------------------------------------------------------------------------------

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



# # this method has problems with infinite ridges
# def compute_voronoi_ridge_sdf_world(
#     shape: Tuple[int, int],
#     centroids: np.ndarray,                 # (k,2) in (y,x) pixel coords
#     ridge_thickness: float = 1.0,          # if bbox provided: WORLD units; else: PIXEL units
#     bbox_min: Optional[list] = None,       # [x,y,z]
#     bbox_max: Optional[list] = None,       # [x,y,z]
# ) -> np.ndarray:
#     ny, nx = shape

#     # -----------------------
#     # Choose coordinate system
#     # -----------------------
#     if bbox_min is None or bbox_max is None:
#         # ===== Original behavior: PIXEL space =====
#         # centroids are (y,x) but Euclidean points should be (x,y)
#         pts = np.stack([centroids[:, 1], centroids[:, 0]], axis=1)  # (x,y) in pixels

#         # Grid points in pixel coords (x,y)
#         Y, X = np.indices((ny, nx))
#         grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)      # (x,y) pixels

#         thickness = float(ridge_thickness)  # pixels

#     else:
#         # ===== World behavior: WORLD space =====
#         xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
#         xmax, ymax = float(bbox_max[0]), float(bbox_max[1])

#         # centroids: (y,x) pixel -> (x,y) world
#         cx = xmin + (centroids[:, 1] / (nx - 1)) * (xmax - xmin)
#         cy = ymin + (centroids[:, 0] / (ny - 1)) * (ymax - ymin)
#         pts = np.stack([cx, cy], axis=1)  # (x,y) world

#         # Grid points: (row,col) -> (x,y) world
#         xs = np.linspace(xmin, xmax, nx)
#         ys = np.linspace(ymin, ymax, ny)
#         Xw, Yw = np.meshgrid(xs, ys, indexing="xy")                 # (ny,nx)
#         grid_points = np.stack([Xw.ravel(), Yw.ravel()], axis=-1)    # (x,y) world

#         thickness = float(ridge_thickness)  # world units

#     # -----------------------
#     # Voronoi + finite ridge segments
#     # -----------------------
#     # vor = Voronoi(pts)
#     vor = Voronoi(pts, qhull_options="Qbb Qc Qx QJ")

#     ridge_segments = []
#     for ridge_vertices in vor.ridge_vertices:
#         if -1 in ridge_vertices:
#             continue
#         v0 = vor.vertices[ridge_vertices[0]]  # (x,y)
#         v1 = vor.vertices[ridge_vertices[1]]  # (x,y)
#         ridge_segments.append((v0, v1))

#     if not ridge_segments:
#         return np.ones((ny, nx), dtype=float) * thickness

#     # -----------------------
#     # Distance to nearest ridge segment
#     # -----------------------
#     min_dists = np.full(grid_points.shape[0], np.inf, dtype=float)
#     for v0, v1 in ridge_segments:
#         dists = point_to_segment_distance(grid_points, v0, v1)  # consistent (x,y)
#         min_dists = np.minimum(min_dists, dists)

#     # Signed distance: negative inside ridge (material), positive outside
#     sdf = (min_dists - thickness).reshape((ny, nx))
#     return sdf


# ------------------------------------------------------------------------------


#  try directtly from curve shapes
with open("alice_result/inShapes.json", 'r') as f:
    shape_data = json.load(f)
    bbox_min = shape_data["bbox"]["minbb"]
    bbox_max = shape_data["bbox"]["maxbb"]

# 1. need convert polylines to true sdfs
nx = 256
ny = 256
sdf_profiles = load_sdf_list_from_inshapes("alice_result/inShapes.json", nx=nx, ny=ny, branch_index=0)
print(f"Loaded {len(sdf_profiles)} SDF profiles from inShapes.json \n")
for i, sdf in enumerate(sdf_profiles):
    if i % 10 == 0:
        # check sdf profiles and contours
        print(f"sdf {i} is a type of {type(sdf)}, shape: {sdf.shape}, min: {sdf.min()}, max: {sdf.max()}")
        debug.output_debug_plot_world_offsets(
            sdf, 
            bbox_min=bbox_min, bbox_max=bbox_max, nx=nx, ny=ny, slice_idx=i,
            offset_step_world=0.05,offset_max_world=0.5,
            file_name_prefix="profile_sdf")
        







# Test if voronoi generation if a true sdf
sigma_world = 0.03
k = 5
seed = 42
# normalize_range = True
iso_level_mask = 0.0

# num_slices = profile_fields_2d.shape[0]

bracing_fields = np.zeros((len(sdf_profiles), ny, nx)) 
prev_centroids = None

for i in range(len(sdf_profiles)): 
    if i %10 ==0:
        slice_2d = sdf_profiles[i]
        # use profile to creat mask
        mask = slice_2d <= iso_level_mask

        centroids = generate_centroids(mask, k=k, prev_centroids=prev_centroids, seed=seed)
        centroids = constrain_centroids_to_mask(centroids, mask)
        

        # V = compute_voronoi_ridge_sdf_world(slice_2d.shape,
        #                                     centroids=centroids,
        #                                     ridge_thickness=sigma_world,
        #                                     bbox_min=bbox_min,
        #                                     bbox_max=bbox_max,)  # true voronoi sdf world units

        V= voronoi_ridge_band_sdf_world(slice_2d.shape,
                                        centroids_px=centroids,
                                        bbox_min=bbox_min,
                                        bbox_max=bbox_max,
                                        sigma_world=sigma_world)

        # prev_centroids = centroids         



    
        # debug.output_debug_voronoi(V, centroids, i)
        debug.output_debug_voronoi(V, centroids, i)
        debug.output_debug_plot_world_offsets(
            V, 
            bbox_min=bbox_min, bbox_max=bbox_max, nx=nx, ny=ny, slice_idx=i,
            offset_step_world=0.001,offset_max_world=0.001,
            file_name_prefix="Voronoi_sdf")
        

        
        

        profile_slice_offset = slice_2d + 0.01
        bracing_cavity = np.maximum(profile_slice_offset, -V)
        bracing_slice = np.maximum(slice_2d, -bracing_cavity)

        debug.output_debug_plot_world_offsets(
            bracing_slice, 
            bbox_min=bbox_min, bbox_max=bbox_max, nx=nx, ny=ny, slice_idx=i,
            centroids=centroids,
            offset_step_world=0.01,
            file_name_prefix="bracing_field")

        
        
        




        

