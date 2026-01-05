"""core.py — SDF post-processing utilities.

Coordinate + shape conventions (critical):

- Volumetric scalar fields are stored as flattened per-slice arrays and are
  conceptually shaped as:

  - P[z, y, x] where:
    - z: slice index (0..num_slices-1)
    - y: row index (0..ny-1)
    - x: col index (0..nx-1)

- Grid index order is always (row, col) == (y, x). We call this **rc**.
- World order is always (x, y, z). We call this **xyz**.
- When we store 2D point pairs in arrays (centroids, pixel coords), we store
  them as rc = [row, col] unless explicitly stated otherwise.

Use helpers:
- to_xy(rc) / to_rc(xy) for swapping conventions (no silent axis swaps).
- rc_to_world_xy(...) / world_xy_to_rc(...) for mapping between image indices
  and world XY (using bounds + nx/ny).
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d, gaussian_filter, distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RBFInterpolator
import contourpy as _contourpy
import time


def to_xy(rc: np.ndarray) -> np.ndarray:
    """Convert (row, col) -> (x, y).

    Accepts shape (..., 2). Returns same shape (..., 2).
    """
    arr = np.asarray(rc)
    if arr.shape[-1] != 2:
        raise ValueError(f"Expected last dim == 2, got shape {arr.shape}")
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 1]  # x = col
    out[..., 1] = arr[..., 0]  # y = row
    return out


def to_rc(xy: np.ndarray) -> np.ndarray:
    """Convert (x, y) -> (row, col).

    Accepts shape (..., 2). Returns same shape (..., 2).
    """
    arr = np.asarray(xy)
    if arr.shape[-1] != 2:
        raise ValueError(f"Expected last dim == 2, got shape {arr.shape}")
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 1]  # row = y
    out[..., 1] = arr[..., 0]  # col = x
    return out


def xy_grid_from_bounds(bounds_min, bounds_max, nx: int, ny: int):
    """Return (x, y, X, Y) matching array indexing.

    - x has length nx, y has length ny
    - X, Y have shape (ny, nx)
    - indexing='xy' so X varies along cols (x), Y varies along rows (y)
    """
    x = np.linspace(float(bounds_min[0]), float(bounds_max[0]), int(nx))
    y = np.linspace(float(bounds_min[1]), float(bounds_max[1]), int(ny))
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y


def rc_to_world_xy(rc, bounds_min, bounds_max, nx: int, ny: int) -> np.ndarray:
    """Map image indices (row, col) to world (x, y) using bounds.

    Uses a linear mapping where:
    - col==0 maps to x=bounds_min[0]
    - col==nx-1 maps to x=bounds_max[0]
    - row==0 maps to y=bounds_min[1]
    - row==ny-1 maps to y=bounds_max[1]

    Returns array (..., 2) in (x, y).
    """
    rc = np.asarray(rc, dtype=float)
    if rc.shape[-1] != 2:
        raise ValueError(f"Expected last dim == 2, got shape {rc.shape}")

    nx = int(nx)
    ny = int(ny)
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid nx/ny: {(nx, ny)}")

    x0, y0 = float(bounds_min[0]), float(bounds_min[1])
    x1, y1 = float(bounds_max[0]), float(bounds_max[1])

    col = rc[..., 1]
    row = rc[..., 0]
    tx = np.zeros_like(col) if nx == 1 else (col / (nx - 1))
    ty = np.zeros_like(row) if ny == 1 else (row / (ny - 1))
    x = x0 + (x1 - x0) * tx
    y = y0 + (y1 - y0) * ty
    return np.stack([x, y], axis=-1)


def world_xy_to_rc(xy, bounds_min, bounds_max, nx: int, ny: int, *, rounding: str = "round") -> np.ndarray:
    """Map world (x, y) to image indices (row, col).

    `rounding`:
    - 'round' (default): nearest integer
    - 'floor': floor
    - 'ceil': ceil

    Returns integer array (..., 2) in (row, col), clipped to image bounds.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[-1] != 2:
        raise ValueError(f"Expected last dim == 2, got shape {xy.shape}")

    nx = int(nx)
    ny = int(ny)
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Invalid nx/ny: {(nx, ny)}")

    x0, y0 = float(bounds_min[0]), float(bounds_min[1])
    x1, y1 = float(bounds_max[0]), float(bounds_max[1])

    x = xy[..., 0]
    y = xy[..., 1]
    tx = np.zeros_like(x) if x1 == x0 else (x - x0) / (x1 - x0)
    ty = np.zeros_like(y) if y1 == y0 else (y - y0) / (y1 - y0)

    col_f = tx * (nx - 1) if nx > 1 else np.zeros_like(tx)
    row_f = ty * (ny - 1) if ny > 1 else np.zeros_like(ty)

    if rounding == "floor":
        col = np.floor(col_f)
        row = np.floor(row_f)
    elif rounding == "ceil":
        col = np.ceil(col_f)
        row = np.ceil(row_f)
    elif rounding == "round":
        col = np.round(col_f)
        row = np.round(row_f)
    else:
        raise ValueError("rounding must be one of: 'round', 'floor', 'ceil'")

    col_i = np.clip(col.astype(int), 0, nx - 1)
    row_i = np.clip(row.astype(int), 0, ny - 1)
    return np.stack([row_i, col_i], axis=-1)

def stack_scalar_fields(data_dict, prefix="scalar_field_values_"):
    """
    Extract all scalar field keys from a JSON dict and stack them
    into a 2D NumPy array, sorted by their numeric index.
    """
    scalar_field_keys = sorted(
        [key for key in data_dict.keys() if key.startswith(prefix)],
        key=lambda x: int(x.split('_')[-1])
    )
    scalar_fields_2d = np.array([data_dict[key] for key in scalar_field_keys])
    return scalar_fields_2d, scalar_field_keys

def meta_data_info(data_dict):
    """
    Extract metadata information from the JSON dict.
    """
    bounds_max = data_dict.get("bounds_max", None)
    bounds_min = data_dict.get("bounds_min", None)
    iso_level = data_dict.get("iso_level", None) 
    slice_count = data_dict.get("slice_count", None)
    # Optional, but preferred. If missing, infer_grid_from_scalar_fields may be
    # forced to fall back to a square-grid inference.
    _ = data_dict.get("nx", None)
    _ = data_dict.get("ny", None)
    return iso_level, slice_count, bounds_max, bounds_min


def meta_data_dict(data_dict) -> dict:
    """Return a normalized metadata dict.

    Keys (if present): iso_level, slice_count, bounds_min, bounds_max, nx, ny.
    """
    return {
        "iso_level": data_dict.get("iso_level", None),
        "slice_count": data_dict.get("slice_count", None),
        "bounds_min": data_dict.get("bounds_min", None),
        "bounds_max": data_dict.get("bounds_max", None),
        "nx": data_dict.get("nx", None),
        "ny": data_dict.get("ny", None),
    }

def infer_grid_from_scalar_fields(scalar_fields, *, nx: int | None = None, ny: int | None = None, metadata: dict | None = None):
    """Infer grid dimensions.

    Returns (num_fields, nx, ny) where each slice is shaped (ny, nx).

    Supports:
    - scalar_fields shape (num_fields, ny, nx)  -> dimensions taken directly
    - scalar_fields shape (num_fields, nx*ny)   -> requires (nx, ny) OR metadata
      containing 'nx'/'ny'. As a legacy fallback, will infer a square grid.
    """
    arr = np.asarray(scalar_fields)
    if arr.ndim == 3:
        num_fields, ny_i, nx_i = arr.shape
        return int(num_fields), int(nx_i), int(ny_i)

    if arr.ndim != 2:
        raise ValueError(f"Expected scalar_fields with ndim 2 or 3, got shape {arr.shape}")

    num_fields, values_per_field = arr.shape

    if metadata is not None:
        nx = metadata.get("nx", nx)
        ny = metadata.get("ny", ny)

    if nx is not None and ny is not None:
        nx = int(nx)
        ny = int(ny)
        if nx <= 0 or ny <= 0:
            raise ValueError(f"Invalid nx/ny: {(nx, ny)}")
        if nx * ny != int(values_per_field):
            raise ValueError(
                f"nx*ny ({nx}*{ny}={nx*ny}) does not match values_per_field ({values_per_field})."
            )
        return int(num_fields), nx, ny

    # Legacy fallback: try square
    n = int(np.sqrt(values_per_field))
    if n * n == values_per_field:
        return int(num_fields), int(n), int(n)

    raise ValueError(
        "Cannot infer non-square grid from flattened fields without nx/ny. "
        "Provide nx, ny in metadata (JSON/NPZ) or pass nx=/ny= explicitly."
    )

def compute_sf_operation(sf_A, sf_B, iso_level_A=0.0, iso_level_B=0.0, mode="difference", swap=False):
    """
    Compute an SDF boolean-like operation between two scalar fields.
    """
    A = np.asarray(sf_A, dtype=float) - iso_level_A
    B = np.asarray(sf_B, dtype=float) - iso_level_B

    if A.shape != B.shape:
        raise ValueError("Input scalar fields must have the same shape")

    if swap:
        A, B = B, A

    m = mode.lower()
    if m in ("difference", "a_minus_b", "sub"):
        result = np.maximum(A, -B)
    elif m in ("union", "or", "min"):
        result = np.minimum(A, B)
    elif m in ("intersection", "and", "max"):
        result = np.maximum(A, B)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return result

def iso_curves_for_slice_2d(slice_2d, level, X, Y):
    """
    Extract iso-curves from a 2D slice using a fast contour engine (contourpy).

    Parameters:
        slice_2d: 2D array (ny, nx)
        level: iso value
        X, Y: coordinate grids (ny, nx) from np.meshgrid(..., indexing="xy")
    Returns:
        list of (N_i, 2) arrays [x, y]
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 2:
        x = X[0, :]
    else:
        x = X
    if Y.ndim == 2:
        y = Y[:, 0]
    else:
        y = Y

    cg = _contourpy.contour_generator(x=x, y=y, z=np.asarray(slice_2d), name="serial")
    return [np.asarray(line, dtype=float) for line in cg.lines(level)]

def get_profile_mask(field_2d, iso_level=0.0):
    """
    Returns a boolean mask where the field is inside the profile (value < iso_level).
    """
    return field_2d < iso_level

def generate_centroids(mask, k=3, prev_centroids=None, seed=42):
    """
    Generate k centroids for the given boolean mask using K-Means.
    Handles resizing of prev_centroids if k changes (interpolation support).
    """
    coords = np.argwhere(mask)
    
    # Edge case: Not enough valid pixels in mask
    if len(coords) < k:
        if prev_centroids is not None:
            # If we have previous centroids, try to reuse them
            if len(prev_centroids) >= k:
                return prev_centroids[:k]
            else:
                # Pad with the last centroid if we need more
                pad = np.tile(prev_centroids[-1:], (k - len(prev_centroids), 1))
                return np.vstack([prev_centroids, pad])
        else:
            ny, nx = mask.shape
            return np.array([[ny/2, nx/2]] * k)

    # Prepare initialization for KMeans
    init_centroids = 'k-means++'
    n_init = 10

    if prev_centroids is not None:
        prev_k = len(prev_centroids)
        if prev_k == k:
            init_centroids = prev_centroids
            n_init = 1
        elif prev_k > k:
            # Shrink: Keep the first k centroids
            # (Assuming coherence is maintained by order)
            init_centroids = prev_centroids[:k]
            n_init = 1
        else:
            # Grow: Keep existing, add new random points from mask
            num_new = k - prev_k
            rng = np.random.default_rng(seed)
            # Sample new points from valid coordinates
            new_indices = rng.choice(len(coords), size=num_new, replace=False)
            new_points = coords[new_indices]
            init_centroids = np.vstack([prev_centroids, new_points])
            n_init = 1

    kmeans = KMeans(n_clusters=k, init=init_centroids, n_init=n_init, random_state=seed)
    kmeans.fit(coords)
    return kmeans.cluster_centers_

def build_centroid_tracks(masks, k_max, seed=42, sigma=2.0):
    """
    Generate smooth 3D trajectories for centroids across all slices.
    1. Initializes k_max centroids at the slice with the largest mask area.
    2. Tracks them forward and backward using KMeans with init=prev.
    3. Smooths the resulting trajectories along the Z-axis.
    """
    num_slices, ny, nx = masks.shape
    trajectories = np.zeros((num_slices, k_max, 2))

    # 1. Find best initialization slice (largest area)
    areas = np.sum(masks, axis=(1, 2))
    start_idx = np.argmax(areas)

    # 2. Initialize at start_idx
    trajectories[start_idx] = generate_centroids(masks[start_idx], k=k_max, seed=seed)

    # 3. Forward pass
    for i in range(start_idx + 1, num_slices):
        trajectories[i] = generate_centroids(masks[i], k=k_max, prev_centroids=trajectories[i-1], seed=seed)

    # 4. Backward pass
    for i in range(start_idx - 1, -1, -1):
        trajectories[i] = generate_centroids(masks[i], k=k_max, prev_centroids=trajectories[i+1], seed=seed)

    # 5. Smooth along axis 0 (slices)
    smoothed = gaussian_filter1d(trajectories, sigma=sigma, axis=0)

    return smoothed

def compute_k_schedule(num_slices, k_min, k_max):
    """
    Returns integer array K_per_slice length num_slices.
    Linear ramp from Kmin to Kmax.
    """
    return np.linspace(k_min, k_max, num_slices).astype(int)

def compute_activation_weights(k_schedule, k_max, ramp=3):
    """
    Returns weights W shape (num_slices, Kmax) in [0,1].
    When a centroid becomes active, ramp it 0->1 over `ramp` slices.
    """
    num_slices = len(k_schedule)
    weights = np.zeros((num_slices, k_max))
    
    for i in range(num_slices):
        k_curr = k_schedule[i]
        
        # Fully active centroids
        weights[i, :k_curr] = 1.0
        
        # Handle ramp for newly active centroids
        # If k increases, we want to smooth the transition.
        # However, the simple logic is: if index j < k_curr, it's active.
        # To make it smooth, we can look at fractional k or just ramp based on index.
        # Let's use a simple heuristic:
        # If k_schedule is float, we could use fractional part.
        # But k_schedule is int.
        # Let's just use the provided ramp logic:
        # "When a centroid becomes active, ramp it 0->1 over `ramp` slices."
        
        # This requires knowing WHEN it became active.
        # Simpler: Weight based on distance from "birth slice".
        pass

    # Refined logic:
    # Iterate columns (centroids). Find first slice where k_schedule > col_idx.
    for col in range(k_max):
        # Find indices where this centroid should be active
        active_indices = np.where(k_schedule > col)[0]
        
        if len(active_indices) == 0:
            continue
            
        start_idx = active_indices[0]
        
        # Set 1.0 for all active
        weights[active_indices, col] = 1.0
        
        # Apply ramp at the beginning
        # e.g. if ramp=3, weights at start, start+1, start+2 should be 0.33, 0.66, 1.0
        ramp_len = min(ramp, len(active_indices))
        if ramp_len > 0:
            ramp_vals = np.linspace(0.0, 1.0, ramp_len + 1)[1:] # e.g. [0.33, 0.66, 1.0]
            weights[active_indices[:ramp_len], col] = ramp_vals

    return weights

def compute_weighted_voronoi_ridge(shape, centroids, weights, eps=1e-3):
    """
    Compute weighted Voronoi SDF.
    d_j = ||x - c_j|| / max(eps, weights[j])
    """
    ny, nx = shape
    Y, X = np.indices(shape)
    grid_points = np.stack([Y.ravel(), X.ravel()], axis=-1) # (N, 2)
    
    # Compute distances to all centroids
    # (N, K)
    # We can't use cKDTree easily with weights.
    # Brute force for now (N is ~256*256 = 65k, K is small ~10). Fast enough.
    
    # centroids: (K, 2)
    # grid_points: (N, 2)
    
    # Expand dims for broadcasting
    # (N, 1, 2) - (1, K, 2) -> (N, K, 2)
    diffs = grid_points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    dists = np.sqrt(dists_sq) # (N, K)
    
    # Apply weights
    # weights: (K,)
    # d_weighted = d / w
    # If w is small, d_weighted is huge (effectively infinite distance, so ignored)
    safe_weights = np.maximum(weights, eps)
    weighted_dists = dists / safe_weights[np.newaxis, :]
    
    # Find 1st and 2nd nearest
    # Sort along axis 1
    sorted_dists = np.sort(weighted_dists, axis=1)
    
    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1]
    
    sdf = d2 - d1
    return sdf.reshape(shape)

def generate_interpolated_bracing_fields(profile_fields, k_min, k_max, iso_level=0.0, ramp=3, smooth_sigma=1.0, seed=42, *, nx: int | None = None, ny: int | None = None, metadata: dict | None = None):
    """
    Orchestrates the generation of interpolated bracing fields.
    """
    num_slices, nx, ny = infer_grid_from_scalar_fields(profile_fields, nx=nx, ny=ny, metadata=metadata)
    
    # 1. Pre-calculate masks
    masks = np.zeros((num_slices, ny, nx), dtype=bool)
    for i in range(num_slices):
        slice_2d = profile_fields[i].reshape((ny, nx))
        masks[i] = get_profile_mask(slice_2d, iso_level=iso_level)
        
    # 2. Build Centroid Tracks (Fixed Kmax)
    # We track Kmax centroids throughout.
    tracks = build_centroid_tracks(masks, k_max=k_max, seed=seed, sigma=smooth_sigma)
    
    # 3. Compute Schedule and Weights
    k_schedule = compute_k_schedule(num_slices, k_min, k_max)
    weights = compute_activation_weights(k_schedule, k_max, ramp=ramp)
    
    # 4. Generate Fields
    bracing_fields = np.zeros_like(profile_fields)
    
    for i in range(num_slices):
        # Get current slice data
        current_centroids = tracks[i] # (Kmax, 2)
        current_weights = weights[i]  # (Kmax,)
        
        # Constrain centroids to mask (projection step)
        # Note: We constrain ALL centroids, even inactive ones, to keep tracks valid.
        # But strictly speaking, we only need active ones inside.
        # Let's constrain all for safety.
        current_centroids = constrain_centroids_to_mask(current_centroids, masks[i])
        
        # Compute Weighted Voronoi
        sdf = compute_weighted_voronoi_ridge((ny, nx), current_centroids, current_weights)
        bracing_fields[i] = sdf.ravel()
        
    return bracing_fields, tracks, weights

def compute_voronoi_sdf(shape, centroids):
    """
    Compute a scalar field where value = dist_to_2nd_nearest - dist_to_nearest.
    """
    ny, nx = shape
    Y, X = np.indices(shape)
    grid_points = np.stack([Y.ravel(), X.ravel()], axis=-1)
    tree = cKDTree(centroids)
    dists, _ = tree.query(grid_points, k=2)
    sdf_flat = dists[:, 1] - dists[:, 0]
    return sdf_flat.reshape(shape)

def constrain_centroids_to_mask(centroids, mask):
    """
    Ensure centroids are strictly inside the mask.
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


def compute_soft_regions(
    mask: np.ndarray,
    seeds_rc: np.ndarray,
    tau: float,
    *,
    weights: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute soft region assignment Q for one slice.

    Parameters
    - mask: bool array (ny, nx). True means inside.
    - seeds_rc: array (k, 2) in (row, col) pixel coordinates.
    - tau: softmax temperature (>0). Smaller -> sharper cells.

    - weights: optional array (k,) in [0,1] to modulate influence (ramp-in).

    Returns
    - Q: array (k, ny, nx), where Q[:, y, x] sums to 1 inside mask and is 0 outside.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (ny, nx), got shape {mask.shape}")
    ny, nx = mask.shape

    seeds = np.asarray(seeds_rc, dtype=float)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError(f"seeds_rc must have shape (k, 2), got shape {seeds.shape}")
    k = int(seeds.shape[0])
    if k <= 0:
        raise ValueError("seeds_rc must contain at least one seed")

    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and > 0")

    seeds_used = seeds
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (k,):
            raise ValueError(f"weights must have shape (k,), got {w.shape} for k={k}")
        inactive = (w <= 0.0) | (~np.isfinite(seeds).all(axis=1))
        if np.any(inactive):
            seeds_used = seeds.copy()
            seeds_used[inactive] = 0.0

    Y, X = np.indices((ny, nx), dtype=float)
    sy = seeds_used[:, 0].reshape((k, 1, 1))
    sx = seeds_used[:, 1].reshape((k, 1, 1))
    d = np.hypot(Y[np.newaxis, :, :] - sy, X[np.newaxis, :, :] - sx).astype(np.float32)

    logits = (-d / tau).astype(np.float32)
    logits -= np.max(logits, axis=0, keepdims=True)
    exps = np.exp(logits, dtype=np.float32)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        w = np.clip(w, 0.0, 1.0)
        exps *= w.reshape((k, 1, 1))
    sums = np.sum(exps, axis=0, keepdims=True)
    Q = exps / np.maximum(sums, eps)

    # Zero outside + renormalize inside
    Q[:, ~mask] = 0.0
    s_in = np.sum(Q, axis=0)
    inside = mask & (s_in > eps)
    Q[:, inside] /= s_in[inside][np.newaxis, :]
    return Q


def compute_volume_cell_walls(
    masks_zyx: np.ndarray,
    seeds_zk2: np.ndarray,
    weights_zk: np.ndarray,
    *,
    tau: float,
    wall_method: str = "entropy",
    smooth_sigma_xy: float | None = 1.0,
    threshold: float = 0.6,
    thickness_px: float = 2.0,
    sigma_z: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Build W[z,y,x] and B[z,y,x] volumes from (seeds, weights).

    Pipeline:
    - For each z: compute W_raw[z] from soft assignment Q (with ramp weights)
    - Z-regularize W with gaussian_filter1d(axis=0, sigma=sigma_z)
    - Re-apply mask: W[z] outside mask -> 0
    - For each z: compute B[z] via threshold + distance transform (thick walls)

    Returns
    - W: float32 array (z, y, x) in [0,1] (masked)
    - B: float32 array (z, y, x) signed wall distance (<=0 is brace material)
    """
    masks = np.asarray(masks_zyx, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"masks_zyx must be (z,y,x), got {masks.shape}")
    z_count, ny, nx = masks.shape

    seeds = np.asarray(seeds_zk2, dtype=float)
    weights = np.asarray(weights_zk, dtype=float)
    if seeds.ndim != 3 or seeds.shape[0] != z_count or seeds.shape[2] != 2:
        raise ValueError(f"seeds_zk2 must be (z,k,2), got {seeds.shape}")
    if weights.ndim != 2 or weights.shape[0] != z_count:
        raise ValueError(f"weights_zk must be (z,k), got {weights.shape}")
    if seeds.shape[1] != weights.shape[1]:
        raise ValueError("seeds and weights must have same k dimension")

    k_max = seeds.shape[1]
    W_raw = np.zeros((z_count, ny, nx), dtype=np.float32)

    for z in range(z_count):
        mask = masks[z]
        if not np.any(mask):
            continue
        Q = compute_soft_regions(mask, seeds[z], tau, weights=weights[z])
        W_raw[z] = compute_wall_strength(Q, mask, method=wall_method)

    W = W_raw
    sigma_z = float(sigma_z)
    if np.isfinite(sigma_z) and sigma_z > 0:
        W = gaussian_filter1d(W_raw, sigma=sigma_z, axis=0).astype(np.float32)

    # Re-apply mask after smoothing to prevent bleed outside cavity.
    for z in range(z_count):
        W[z][~masks[z]] = 0.0

    B = np.zeros_like(W, dtype=np.float32)
    for z in range(z_count):
        B[z] = thicken_wall_field(
            W[z],
            masks[z],
            smooth_sigma=smooth_sigma_xy,
            threshold=threshold,
            thickness_px=thickness_px,
        ).astype(np.float32)

    return W, B


def continuity_metrics(volume_zyx: np.ndarray, masks_zyx: np.ndarray, *, bins: int = 30) -> dict:
    """Compute simple continuity metrics across adjacent slices.

    Returns dict with:
    - mean_abs_diff_per_slice: list length z-1
    - mean_abs_diff: float
    - hist_edges, hist_counts
    """
    vol = np.asarray(volume_zyx, dtype=float)
    masks = np.asarray(masks_zyx, dtype=bool)
    if vol.ndim != 3 or masks.ndim != 3 or vol.shape != masks.shape:
        raise ValueError("volume_zyx and masks_zyx must both be (z,y,x) and same shape")

    if vol.shape[0] < 2:
        return {
            "mean_abs_diff_per_slice": [],
            "mean_abs_diff": 0.0,
            "hist_edges": np.array([0.0, 1.0]),
            "hist_counts": np.array([0]),
        }

    diffs = np.abs(vol[1:] - vol[:-1])
    pair_masks = masks[1:] | masks[:-1]
    finite = np.isfinite(diffs)

    means = []
    for i in range(diffs.shape[0]):
        m = pair_masks[i] & finite[i]
        means.append(float(np.mean(diffs[i][m])) if np.any(m) else 0.0)

    all_vals = diffs[pair_masks & finite]
    if all_vals.size == 0:
        edges = np.array([0.0, 1.0])
        counts = np.array([0])
        overall = 0.0
    else:
        vmax = float(np.quantile(all_vals, 0.99))
        vmax = vmax if vmax > 0 else float(np.max(all_vals))
        vmax = vmax if vmax > 0 else 1.0
        edges = np.linspace(0.0, vmax, int(bins) + 1)
        counts, edges = np.histogram(all_vals, bins=edges)
        overall = float(np.mean(all_vals))

    return {
        "mean_abs_diff_per_slice": means,
        "mean_abs_diff": overall,
        "hist_edges": edges,
        "hist_counts": counts,
    }


def compute_wall_strength(Q: np.ndarray, mask: np.ndarray, *, method: str = "entropy", eps: float = 1e-12) -> np.ndarray:
    """Convert soft assignment Q into wall strength W.

    Parameters
    - Q: soft assignments. Supported shapes: (k, ny, nx) or (ny, nx, k)
    - mask: bool array (ny, nx)
    - method:
        - 'entropy': normalized entropy in [0,1]
        - 'top2gap': 1 - (top1 - top2) in [0,1]

    Returns
    - W: float array (ny, nx), 0 outside mask
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (ny, nx), got shape {mask.shape}")
    ny, nx = mask.shape

    q = np.asarray(Q, dtype=float)
    if q.ndim != 3:
        raise ValueError(f"Q must be 3D, got shape {q.shape}")

    if q.shape[0] != ny or q.shape[1] != nx:
        # assume (k, ny, nx)
        if q.shape[1] != ny or q.shape[2] != nx:
            raise ValueError(f"Q must match mask shape; got Q={q.shape}, mask={mask.shape}")
        q_k = q  # (k, ny, nx)
        k = q_k.shape[0]
    else:
        # (ny, nx, k)
        k = q.shape[2]
        q_k = np.moveaxis(q, -1, 0)

    method = method.lower().strip()
    if method == "entropy":
        H = -np.sum(q_k * np.log(q_k + eps), axis=0)
        denom = np.log(float(k)) if k > 1 else 1.0
        W = (H / denom).astype(np.float32)
    elif method in ("top2gap", "top2", "gap"):
        q_xyk = np.moveaxis(q_k, 0, -1)  # (ny, nx, k)
        if k == 1:
            W = np.zeros((ny, nx), dtype=np.float32)
        else:
            top2 = np.partition(q_xyk, kth=-2, axis=-1)[..., -2:]
            top1 = top2[..., 1]
            top2v = top2[..., 0]
            delta = np.clip(top1 - top2v, 0.0, 1.0)
            W = (1.0 - delta).astype(np.float32)
    else:
        raise ValueError("method must be 'entropy' or 'top2gap'")

    W[~mask] = 0.0
    return np.clip(W, 0.0, 1.0)


def thicken_wall_field(
    W: np.ndarray,
    mask: np.ndarray,
    *,
    smooth_sigma: float | None = None,
    threshold: float = 0.6,
    thickness_px: float = 2.0,
) -> np.ndarray:
    """Convert wall strength W into a bracing-friendly signed field B.

    Steps:
    - optional gaussian smoothing
    - threshold to obtain a wall centerline mask
    - distance transform -> signed field B = dist_to_wall - thickness

    Output
    - B: float array (ny, nx) where B<=0 is "inside brace".
         Outside the profile mask, B is forced positive.
    """
    mask = np.asarray(mask, dtype=bool)
    W = np.asarray(W, dtype=float)
    if W.shape != mask.shape:
        raise ValueError(f"W and mask must match shape, got W={W.shape}, mask={mask.shape}")

    W2 = W.copy()
    if smooth_sigma is not None:
        W2 = gaussian_filter(W2, sigma=float(smooth_sigma))

    W2[~mask] = 0.0
    wall_core = (W2 >= float(threshold)) & mask

    # Distance to nearest wall pixel.
    # EDT uses zeros as features; make wall_core pixels be zeros.
    dist = distance_transform_edt(~wall_core).astype(np.float32)
    B = dist - float(thickness_px)

    if np.any(mask):
        B_out = float(np.max(B[mask]) + abs(float(thickness_px)) + 1.0)
        B[~mask] = B_out
    else:
        B[:] = 1.0

    return B


def generate_cell_wall_fields_single_slice(
    mask: np.ndarray,
    seeds_rc: np.ndarray,
    tau: float,
    *,
    wall_method: str = "entropy",
    smooth_sigma: float | None = 1.0,
    threshold: float = 0.6,
    thickness_px: float = 2.0,
):
    """Convenience wrapper that returns (Q, W, B) for a single slice."""
    Q = compute_soft_regions(mask, seeds_rc, tau)
    W = compute_wall_strength(Q, mask, method=wall_method)
    B = thicken_wall_field(W, mask, smooth_sigma=smooth_sigma, threshold=threshold, thickness_px=thickness_px)
    return Q, W, B


def _project_points_into_mask(points_rc: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Project points onto nearest valid pixel inside mask.

    Returns (projected_points_rc, num_projected).
    """
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape

    pts = np.asarray(points_rc, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_rc must be (n,2), got {pts.shape}")

    # Clip to image bounds first.
    r = np.clip(np.round(pts[:, 0]).astype(int), 0, ny - 1)
    c = np.clip(np.round(pts[:, 1]).astype(int), 0, nx - 1)

    inside = mask[r, c]
    if np.all(inside):
        return np.column_stack([r, c]).astype(float), 0

    # EDT: zeros are features. Make inside-mask be zeros.
    _, inds = distance_transform_edt(~mask, return_indices=True)
    pr = r.copy()
    pc = c.copy()
    moved = 0
    for i in range(len(r)):
        if inside[i]:
            continue
        pr[i] = int(inds[0, r[i], c[i]])
        pc[i] = int(inds[1, r[i], c[i]])
        moved += 1

    return np.column_stack([pr, pc]).astype(float), moved


def _hard_assign_regions(mask: np.ndarray, seeds_rc: np.ndarray) -> np.ndarray:
    """Assign each in-mask pixel to nearest seed. Returns labels (ny,nx), -1 outside."""
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    seeds = np.asarray(seeds_rc, dtype=float)
    k = seeds.shape[0]
    Y, X = np.indices((ny, nx), dtype=float)
    sy = seeds[:, 0].reshape((k, 1, 1))
    sx = seeds[:, 1].reshape((k, 1, 1))
    d2 = (Y[np.newaxis, :, :] - sy) ** 2 + (X[np.newaxis, :, :] - sx) ** 2
    labels = np.argmin(d2, axis=0).astype(np.int32)
    labels[~mask] = -1
    return labels


def track_seeds_with_splits(
    masks_zyx: np.ndarray,
    k_schedule: np.ndarray,
    init_seeds_rc: np.ndarray,
    *,
    z0: int = 0,
    ramp_slices: int = 5,
    use_ot_transport: bool = False,
    ot_num_samples: int = 600,
    ot_band_px: float | None = 6.0,
    ot_epsilon: float = 8.0,
    ot_max_iter: int = 200,
    ot_tol: float = 1e-3,
    ot_rbf_smooth: float = 5.0,
    ot_max_disp_px: float | None = 20.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Track seeds across slices with stable IDs and split-based births.

    Implements:
    - prediction (constant velocity)
    - projection into mask
    - Hungarian assignment to preserve IDs
    - when K increases: split largest region, new seed at farthest point
    - ramped influence weights for new seeds

    Returns
    - seeds_out: (num_slices, Kmax, 2) rc pixel coords (float)
    - weights_out: (num_slices, Kmax) in [0,1]
    - report: dict with debug metrics
    """
    masks = np.asarray(masks_zyx, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"masks_zyx must be (z, y, x), got {masks.shape}")
    num_slices, ny, nx = masks.shape

    k_schedule = np.asarray(k_schedule, dtype=int)
    if k_schedule.shape[0] != num_slices:
        raise ValueError("k_schedule length must match number of slices")
    if np.any(k_schedule < 0):
        raise ValueError("k_schedule must be non-negative")

    init_seeds = np.asarray(init_seeds_rc, dtype=float)
    if init_seeds.ndim != 2 or init_seeds.shape[1] != 2:
        raise ValueError("init_seeds_rc must be (k0,2)")

    Kmax = int(np.max(k_schedule)) if num_slices > 0 else 0
    seeds_out = np.full((num_slices, Kmax, 2), np.nan, dtype=float)
    weights_out = np.zeros((num_slices, Kmax), dtype=float)

    birth_z = np.full((Kmax,), -1, dtype=int)

    z0 = int(z0)
    if not (0 <= z0 < num_slices):
        raise ValueError("z0 out of range")
    k0 = int(k_schedule[z0])
    if k0 > len(init_seeds):
        raise ValueError("init_seeds_rc has fewer seeds than k_schedule[z0]")

    # Initialize
    seeds_out[z0, :k0] = init_seeds[:k0]
    birth_z[:k0] = z0
    weights_out[z0, :k0] = 1.0

    report = {
        "max_displacement_per_slice": [0.0] * num_slices,
        "num_projected_per_slice": [0] * num_slices,
        "split_events": [],
        "ot": {
            "enabled": bool(use_ot_transport),
            "num_samples": int(ot_num_samples),
            "timings_ms": [0.0] * num_slices,
            "failures": 0,
        },
    }

    for z in range(z0 + 1, num_slices):
        k_prev = int(k_schedule[z - 1])
        k_curr = int(k_schedule[z])
        if k_prev == 0 and k_curr == 0:
            continue

        prev = seeds_out[z - 1, :k_prev].copy()
        if z - 2 >= z0 and k_prev > 0:
            prev_prev = seeds_out[z - 2, :k_prev]
            vel = np.nan_to_num(prev - prev_prev)
        else:
            vel = np.zeros_like(prev)
        predicted = prev + vel

        # Optional OT-guided transport to reduce sliding when geometry changes.
        if use_ot_transport and k_prev > 0:
            t0 = time.perf_counter()
            try:
                warp = compute_ot_warp_rc(
                    masks[z - 1],
                    masks[z],
                    num_samples=ot_num_samples,
                    band_px=ot_band_px,
                    epsilon=ot_epsilon,
                    max_iter=ot_max_iter,
                    tol=ot_tol,
                    rbf_smooth=ot_rbf_smooth,
                )
                disp = warp(predict_seed_points_rc(predicted))
                if ot_max_disp_px is not None:
                    maxd = float(ot_max_disp_px)
                    if maxd > 0:
                        nrm = np.linalg.norm(disp, axis=1)
                        scale = np.minimum(1.0, maxd / np.maximum(nrm, 1e-9))
                        disp = disp * scale[:, None]
                predicted = predicted + disp
            except Exception:
                report["ot"]["failures"] += 1
            finally:
                report["ot"]["timings_ms"][z] = (time.perf_counter() - t0) * 1000.0

        projected, num_proj = _project_points_into_mask(predicted, masks[z])
        report["num_projected_per_slice"][z] = int(num_proj)

        # Hungarian assignment to keep stable IDs even if projections cross.
        if k_prev > 0:
            cost = np.linalg.norm(predicted[:, None, :] - projected[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            ordered = np.zeros_like(projected)
            ordered[row_ind] = projected[col_ind]
        else:
            ordered = projected

        # Handle births (splits)
        if k_curr > k_prev:
            labels = _hard_assign_regions(masks[z], ordered)
            areas = np.array([(labels == i).sum() for i in range(k_prev)], dtype=int)
            parent = int(np.argmax(areas)) if k_prev > 0 else -1

            if parent >= 0 and areas[parent] > 0:
                Y, X = np.indices((ny, nx), dtype=float)
                py, px = ordered[parent]
                d = np.hypot(Y - py, X - px)
                d[labels != parent] = -1.0
                flat_idx = int(np.argmax(d))
                new_r, new_c = np.unravel_index(flat_idx, (ny, nx))
                new_seed = np.array([float(new_r), float(new_c)], dtype=float)
            else:
                # Fallback: pick any in-mask pixel farthest from existing seeds (or center).
                if np.any(masks[z]):
                    coords = np.argwhere(masks[z])
                    new_seed = coords[len(coords) // 2].astype(float)
                else:
                    new_seed = np.array([ny / 2, nx / 2], dtype=float)

            # Append births sequentially
            for new_id in range(k_prev, k_curr):
                ordered = np.vstack([ordered, new_seed])
                birth_z[new_id] = z
                report["split_events"].append(
                    {
                        "z": int(z),
                        "parent_id": int(parent),
                        "new_id": int(new_id),
                        "new_seed_rc": (float(new_seed[0]), float(new_seed[1])),
                    }
                )

        seeds_out[z, :k_curr] = ordered[:k_curr]

        # Displacement metric
        if k_prev > 0:
            disp = np.linalg.norm(seeds_out[z, :k_prev] - seeds_out[z - 1, :k_prev], axis=1)
            report["max_displacement_per_slice"][z] = float(np.nanmax(disp))

        # Weights with ramp
        r = max(1, int(ramp_slices))
        for j in range(min(k_curr, Kmax)):
            bz = int(birth_z[j])
            if bz < 0:
                continue
            age = z - bz
            weights_out[z, j] = min(1.0, (age + 1) / r)

    return seeds_out, weights_out, report


def predict_seed_points_rc(seeds_rc: np.ndarray) -> np.ndarray:
    """Validate/clean a (k,2) rc array used as evaluation points."""
    pts = np.asarray(seeds_rc, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (k,2), got {pts.shape}")
    pts = pts.copy()
    pts[~np.isfinite(pts)] = 0.0
    return pts


def _mask_mass_candidates(mask: np.ndarray, band_px: float | None) -> np.ndarray:
    """Return candidate pixel coords (n,2) inside mask, optionally near boundary."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros((0, 2), dtype=int)

    if band_px is None:
        return np.argwhere(mask)

    band_px = float(band_px)
    if not np.isfinite(band_px) or band_px <= 0:
        return np.argwhere(mask)

    # distance to boundary within the mask
    dist_in = distance_transform_edt(mask)
    band = (dist_in <= band_px) & mask
    coords = np.argwhere(band)
    if coords.size == 0:
        coords = np.argwhere(mask)
    return coords


def sample_points_stratified(coords_rc: np.ndarray, num_samples: int, *, seed: int = 0) -> np.ndarray:
    """Stratified-ish sampling of rc coords.

    Uses a tile-based approach to avoid clumping; falls back to uniform sampling.
    """
    coords = np.asarray(coords_rc)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords_rc must be (n,2), got {coords.shape}")

    n = coords.shape[0]
    if n == 0:
        return coords.astype(float)
    m = int(num_samples)
    if m <= 0:
        return coords[:0].astype(float)
    if n <= m:
        return coords.astype(float)

    rng = np.random.default_rng(int(seed))

    r = coords[:, 0]
    c = coords[:, 1]
    r0, r1 = int(r.min()), int(r.max())
    c0, c1 = int(c.min()), int(c.max())
    span_r = max(1, r1 - r0 + 1)
    span_c = max(1, c1 - c0 + 1)

    # Tile size chosen so roughly one sample per tile.
    tile_area = max(1.0, (span_r * span_c) / float(m))
    tile = int(max(2.0, np.sqrt(tile_area)))
    tr = (r - r0) // tile
    tc = (c - c0) // tile
    key = tr.astype(np.int64) * 1_000_000 + tc.astype(np.int64)

    # For each tile, pick one random point.
    order = rng.permutation(n)
    picked = []
    seen = set()
    for idx in order:
        k = int(key[idx])
        if k in seen:
            continue
        picked.append(idx)
        seen.add(k)
        if len(picked) >= m:
            break

    if len(picked) < m:
        remaining = np.setdiff1d(np.arange(n), np.array(picked, dtype=int), assume_unique=False)
        extra = rng.choice(remaining, size=(m - len(picked)), replace=False)
        picked = np.concatenate([np.array(picked, dtype=int), extra])
    else:
        picked = np.array(picked[:m], dtype=int)

    return coords[picked].astype(float)


def sinkhorn_entropic(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    *,
    epsilon: float,
    max_iter: int = 200,
    tol: float = 1e-3,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute entropic OT coupling via Sinkhorn scaling (numpy-only).

    Returns coupling P (n,m) where row sums ~ a and col sums ~ b.
    Falls back by raising if numerical issues occur.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    C = np.asarray(C, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or C.ndim != 2:
        raise ValueError("Invalid shapes for a,b,C")
    if C.shape != (a.size, b.size):
        raise ValueError("C shape must be (len(a), len(b))")

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    a = a / max(eps, float(a.sum()))
    b = b / max(eps, float(b.sum()))

    # Stabilize cost a bit.
    C0 = C - float(np.min(C))
    K = np.exp(-C0 / epsilon)
    K = np.maximum(K, eps)

    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(int(max_iter)):
        Kv = K @ v
        u_new = a / np.maximum(Kv, eps)
        Ku = K.T @ u_new
        v_new = b / np.maximum(Ku, eps)

        if not (np.all(np.isfinite(u_new)) and np.all(np.isfinite(v_new))):
            raise FloatingPointError("Sinkhorn diverged")

        # Convergence check: marginal error
        if float(tol) > 0:
            P = (u_new[:, None] * K) * v_new[None, :]
            err = max(
                float(np.max(np.abs(P.sum(axis=1) - a))),
                float(np.max(np.abs(P.sum(axis=0) - b))),
            )
            if err < float(tol):
                u, v = u_new, v_new
                break

        u, v = u_new, v_new

    P = (u[:, None] * K) * v[None, :]
    if not np.all(np.isfinite(P)):
        raise FloatingPointError("Sinkhorn produced non-finite coupling")
    return P


def compute_ot_warp_rc(
    mask_src: np.ndarray,
    mask_tgt: np.ndarray,
    *,
    num_samples: int = 600,
    band_px: float | None = 6.0,
    epsilon: float = 8.0,
    max_iter: int = 200,
    tol: float = 1e-3,
    rbf_smooth: float = 5.0,
    seed: int = 0,
):
    """Compute a smooth 2D displacement warp u(rc) from src slice -> tgt slice.

    - Samples points from each mask (optionally near boundary band)
    - Computes entropic Sinkhorn OT coupling
    - Uses barycentric map to create correspondences
    - Fits an RBF warp (thin-plate spline) from src->disp

    Returns a callable warp(points_rc) -> disp_rc with shape (n,2).
    On failure, returns identity warp (zeros).
    """
    src_coords = _mask_mass_candidates(mask_src, band_px)
    tgt_coords = _mask_mass_candidates(mask_tgt, band_px)

    n = int(num_samples)
    if src_coords.shape[0] < 10 or tgt_coords.shape[0] < 10 or n < 10:
        return lambda pts: np.zeros((len(pts), 2), dtype=float)

    X = sample_points_stratified(src_coords, n, seed=seed)
    Y = sample_points_stratified(tgt_coords, n, seed=seed + 1)

    a = np.ones((X.shape[0],), dtype=float) / float(X.shape[0])
    b = np.ones((Y.shape[0],), dtype=float) / float(Y.shape[0])

    # Cost in pixel coordinates
    dx = X[:, None, 0] - Y[None, :, 0]
    dy = X[:, None, 1] - Y[None, :, 1]
    C = dx * dx + dy * dy

    try:
        P = sinkhorn_entropic(a, b, C, epsilon=epsilon, max_iter=max_iter, tol=tol)
        row = P.sum(axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        Yb = (P @ Y) / row
        disp = Yb - X

        # Fit thin-plate spline warp for displacement.
        # RBFInterpolator expects (n,2) and (n,2) values.
        rbf = RBFInterpolator(X, disp, kernel="thin_plate_spline", smoothing=float(rbf_smooth))

        def warp(points_rc: np.ndarray) -> np.ndarray:
            pts = predict_seed_points_rc(points_rc)
            out = rbf(pts)
            out = np.asarray(out, dtype=float)
            out[~np.isfinite(out)] = 0.0
            return out

        return warp
    except Exception:
        return lambda pts: np.zeros((len(pts), 2), dtype=float)


def vector_field_lines_rc(
    disp_rc_yx2: np.ndarray,
    *,
    stride: int = 8,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dense displacement field into polyline segments for debug.

    Input: disp_rc_yx2 with shape (ny, nx, 2) where disp[...,0]=dr, disp[...,1]=dc.
    Output:
    - points_rc: (N*2, 3) points (r,c,z=0)
    - lines: (N, 2) indices into points
    """
    u = np.asarray(disp_rc_yx2, dtype=float)
    if u.ndim != 3 or u.shape[2] != 2:
        raise ValueError("disp_rc_yx2 must be (ny,nx,2)")
    ny, nx, _ = u.shape
    s = max(1, int(stride))

    ys = np.arange(0, ny, s)
    xs = np.arange(0, nx, s)
    rr, cc = np.meshgrid(ys, xs, indexing="ij")
    base = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(float)
    disp = u[rr, cc].reshape((-1, 2)) * float(scale)
    tip = base + disp

    pts0 = np.column_stack([base[:, 0], base[:, 1], np.zeros((base.shape[0],))])
    pts1 = np.column_stack([tip[:, 0], tip[:, 1], np.zeros((tip.shape[0],))])
    points = np.vstack([pts0, pts1]).astype(np.float64)
    n = base.shape[0]
    lines = np.column_stack([np.arange(n), np.arange(n) + n]).astype(np.int32)
    return points, lines


def demo_cell_wall_single_slice_npz(
    out_path: str,
    *,
    nx: int = 256,
    ny: int = 256,
    k: int = 6,
    tau: float = 12.0,
    wall_method: str = "entropy",
    smooth_sigma: float | None = 1.0,
    threshold: float = 0.6,
    thickness_px: float = 2.0,
    seed: int = 0,
):
    """Generate a synthetic ellipse mask + random seeds and save (W,B) to NPZ."""
    nx = int(nx)
    ny = int(ny)
    rng = np.random.default_rng(int(seed))

    Y, X = np.indices((ny, nx), dtype=float)
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    a = 0.42 * nx
    b = 0.32 * ny
    mask = (((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2) <= 1.0

    coords = np.argwhere(mask)
    if len(coords) < k:
        raise ValueError("Mask too small for requested k")
    seeds_rc = coords[rng.choice(len(coords), size=k, replace=False)].astype(float)

    Q, W, B = generate_cell_wall_fields_single_slice(
        mask,
        seeds_rc,
        tau,
        wall_method=wall_method,
        smooth_sigma=smooth_sigma,
        threshold=threshold,
        thickness_px=thickness_px,
    )

    np.savez(
        out_path,
        mask=mask.astype(np.uint8),
        seeds_rc=seeds_rc,
        tau=float(tau),
        wall_method=str(wall_method),
        W=W,
        B=B,
        Q=Q,
        nx=nx,
        ny=ny,
    )
