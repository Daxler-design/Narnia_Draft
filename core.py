import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
import contourpy as _contourpy

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
    total_height = data_dict.get("total_height", None)
    return iso_level, slice_count, bounds_max, bounds_min

def infer_grid_from_scalar_fields(scalar_fields_2d):
    """
    Infer (nx, ny) from scalar_fields_2d.shape[1] assuming a square grid.
    """
    num_fields, values_per_field = scalar_fields_2d.shape
    n = int(np.sqrt(values_per_field))
    if n * n != values_per_field:
        raise ValueError(f"Cannot infer square grid from {values_per_field} values.")
    return num_fields, n, n

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

def generate_interpolated_bracing_fields(profile_fields, k_min, k_max, iso_level=0.0, ramp=3, smooth_sigma=1.0, seed=42):
    """
    Orchestrates the generation of interpolated bracing fields.
    """
    num_slices, nx, ny = infer_grid_from_scalar_fields(profile_fields)
    
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
