"""
Bracing generation using Voronoi diagrams and keyfield blending.

Functions for generating bracing patterns from profile scalar fields using:
- Static Voronoi diagrams with K-means clustering
- Dynamic keyfield blending between key slices
- Ridge response transformation for smoother patterns
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List


def get_profile_mask(field_2d: np.ndarray, iso_level: float = 0.0) -> np.ndarray:
    """
    Returns a boolean mask where the field is inside the profile (value < iso_level).
    
    Args:
        field_2d: 2D scalar field array (ny, nx)
        iso_level: Threshold value (default: 0.0)
    
    Returns:
        Boolean mask where True indicates inside profile (field < iso_level)
    
    Example:
        >>> field = np.random.rand(50, 50) - 0.5
        >>> mask = get_profile_mask(field, iso_level=0.0)
        >>> np.sum(mask)  # Count of points inside profile
        1234
    """
    return field_2d < iso_level


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
        kmeans = KMeans(n_clusters=k, init=prev_centroids, n_init=1, random_state=seed)
    else:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
        
    kmeans.fit(coords)
    return kmeans.cluster_centers_


def compute_voronoi_sdf(shape: Tuple[int, int], centroids: np.ndarray) -> np.ndarray:
    """
    Compute a Voronoi-based scalar field: value = dist_to_2nd_nearest - dist_to_nearest.
    
    Creates ridge patterns at Voronoi cell boundaries where value is maximized.
    
    Args:
        shape: Output field shape (ny, nx)
        centroids: Array of centroid positions (k, 2) in [y, x] coordinates
    
    Returns:
        2D array (ny, nx) with Voronoi SDF values
        - Positive values near cell boundaries (ridges)
        - Near zero at points closest to a single centroid
    
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
    sdf_flat = dists[:, 1] - dists[:, 0]
    return sdf_flat.reshape(shape)


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


def generate_bracing_static(profile_fields_2d, iso_level, nx, ny, k, seed=42):
    """
    Generate static bracing fields based on profile fields centroids.
    
    Uses K-means clustering per slice with temporal coherence to create
    Voronoi-based bracing patterns.
    
    Args:
        profile_fields_2d: Profile scalar fields (num_slices, nx*ny)
        iso_level: Threshold for profile mask
        nx: Grid width
        ny: Grid height
        k: Number of centroids per slice
        seed: Random seed for K-means
    
    Returns:
        Bracing fields array (num_slices, nx*ny) with Voronoi SDF values
    
    Example:
        >>> profile = np.random.rand(60, 2500) - 0.5
        >>> bracing = generate_bracing_static(profile, 0.0, 50, 50, k=5)
        >>> bracing.shape
        (60, 2500)
    
    Note:
        - Centroids from previous slice used as initial guess for next slice
        - Prints progress every 10 slices
        - Centroids constrained to lie within profile mask
    """
    num_fields = profile_fields_2d.shape[0]
    bracing_fields = np.zeros_like(profile_fields_2d)
    prev_centroids = None
    
    for i in range(num_fields):
        slice_2d = profile_fields_2d[i].reshape((ny, nx))
        mask = get_profile_mask(slice_2d, iso_level=iso_level)
        
        centroids = generate_centroids(mask, k=k, prev_centroids=prev_centroids, seed=seed)
        centroids = constrain_centroids_to_mask(centroids, mask)
        
        voronoi_sdf_flat = compute_voronoi_sdf((ny, nx), centroids)
        bracing_fields[i] = voronoi_sdf_flat.ravel()
        prev_centroids = centroids
        
        if i % 10 == 0:
            print(f"Generated bracing for slice {i}/{num_fields}")
            
    return bracing_fields


def generate_bracing_keyfield_blend(profile_fields_2d, iso_level, nx, ny, keys_config, smooth=0.0, seed=42, sigma=5.0, tau=0.5, beta=4.0):
    """
    Generate bracing fields by blending Voronoi fields between key slices.
    
    Uses Ridge Response transformation (R = exp(-(V/sigma)^2)) for smoother
    blending than raw Voronoi values. Interpolates between key slices using
    optional smoothstep and exponential blending.
    
    Args:
        profile_fields_2d: Profile scalar fields (num_slices, nx*ny)
        iso_level: Threshold for profile mask
        nx: Grid width
        ny: Grid height
        keys_config: List of (slice_idx, k_centroids) tuples defining key slices
        smooth: Smoothstep interpolation factor [0=linear, 1=smoothstep] (default: 0.0)
        seed: Random seed for K-means
        sigma: Ridge width parameter (default: 5.0)
        tau: Ridge threshold offset (default: 0.5)
        beta: Exponential blend softness (default: 4.0, use 0 for linear)
    
    Returns:
        Bracing fields array (num_slices, nx*ny) with blended ridge values
    
    Example:
        >>> profile = np.random.rand(60, 2500) - 0.5
        >>> keys = [(0, 3), (30, 4), (59, 5)]  # Variable centroid count
        >>> bracing = generate_bracing_keyfield_blend(
        ...     profile, 0.0, 50, 50, keys, smooth=0.5, sigma=5.0
        ... )
        >>> bracing.shape
        (60, 2500)
    
    Note:
        - Slices before first key use first key's field
        - Slices after last key use last key's field
        - Blending uses smooth-max when beta > 0, linear lerp when beta == 0
        - Output B = R - tau where R is normalized ridge response
        - Prints progress every 10 slices
    """
    # Sort keys by slice index
    sorted_keys = sorted(keys_config, key=lambda x: x[0])
    if not sorted_keys:
        return np.zeros_like(profile_fields_2d)

    num_fields = profile_fields_2d.shape[0]
    bracing_fields = np.zeros_like(profile_fields_2d)

    # Cache for key fields: map slice_idx -> 2D field
    key_fields_cache = {}

    def get_key_field(idx, k):
        # Helper to compute normalized Ridge Response field
        cache_key = (idx, k)
        if cache_key in key_fields_cache:
            return key_fields_cache[cache_key]
        
        slice_2d = profile_fields_2d[idx].reshape((ny, nx))
        mask = get_profile_mask(slice_2d, iso_level=iso_level)
        centroids = generate_centroids(mask, k=k, prev_centroids=None, seed=seed)
        centroids = constrain_centroids_to_mask(centroids, mask)
        
        # 1. Raw Voronoi SDF (d2 - d1)
        V = compute_voronoi_sdf((ny, nx), centroids)
        
        # 2. Convert to Ridge Response R
        R = np.exp(- (V / sigma)**2)
        
        # 3. Normalize R based on P95 inside mask
        valid_vals = R[mask]
        if valid_vals.size > 0:
            p95 = np.percentile(valid_vals, 95)
            if p95 > 1e-6:
                R = R / p95

        key_fields_cache[cache_key] = R
        return R

    # 1. Fill Before First Key
    first_idx, first_k = sorted_keys[0]
    if first_idx > 0:
        R_first = get_key_field(first_idx, first_k)
        B_first = R_first - tau
        
        for i in range(first_idx):
             slice_mask = get_profile_mask(profile_fields_2d[i].reshape((ny, nx)), iso_level=iso_level)
             field = B_first.copy().ravel()
             field[~slice_mask.ravel()] = -10.0
             bracing_fields[i] = field

    # 2. Interpolate between keys
    for i in range(len(sorted_keys) - 1):
        idx_start, k_start = sorted_keys[i]
        idx_end, k_end = sorted_keys[i+1]
        
        R_start = get_key_field(idx_start, k_start)
        R_end = get_key_field(idx_end, k_end)
        
        steps = idx_end - idx_start
        for j in range(steps):
            curr_idx = idx_start + j
            t = j / float(steps)

            # Apply smoothing to t (Linear -> Smoothstep)
            if smooth > 0:
                t_smooth = t * t * (3 - 2 * t)
                t = (1 - smooth) * t + smooth * t_smooth
            
            # Field Blending
            if beta > 0.0:
                # Smooth Max Blending
                R_curr = (1.0 / beta) * np.log((1-t) * np.exp(beta * R_start) + t * np.exp(beta * R_end))
            else:
                # Linear Blend (Lerp)
                R_curr = (1.0 - t) * R_start + t * R_end
            
            # Reconstruction B = R - tau
            B_curr = R_curr - tau
            
            # Masking for current slice
            slice_mask = get_profile_mask(profile_fields_2d[curr_idx].reshape((ny, nx)), iso_level=iso_level)
            B_curr[~slice_mask] = -10.0
            
            bracing_fields[curr_idx] = B_curr.ravel()
            
            if curr_idx % 10 == 0:
                print(f"Generated bracing (blend) for slice {curr_idx}/{num_fields}")

    # 3. Fill After Last Key
    last_idx, last_k = sorted_keys[-1]
    R_last = get_key_field(last_idx, last_k)
    B_last = R_last - tau
    
    for i in range(last_idx, num_fields):
        slice_mask = get_profile_mask(profile_fields_2d[i].reshape((ny, nx)), iso_level=iso_level)
        field = B_last.copy()
        field[~slice_mask] = -10.0
        bracing_fields[i] = field.ravel()

    return bracing_fields


def compute_k_from_area(mask: np.ndarray, area_per_seed: float, k_min: int = 2, k_max: int = 12) -> int:
    """
    Compute optimal number of centroids based on mask area.
    
    Args:
        mask: Boolean mask (ny, nx)
        area_per_seed: Target pixels per centroid (e.g., 1500.0)
        k_min: Minimum centroids (default: 2)
        k_max: Maximum centroids (default: 12)
    
    Returns:
        Number of centroids k, clamped to [k_min, k_max]
    
    Example:
        >>> mask = np.ones((100, 100), dtype=bool)  # 10000 px²
        >>> k = compute_k_from_area(mask, area_per_seed=2000, k_min=2, k_max=10)
        >>> k
        5  # 10000 / 2000 = 5 centroids
    
    Note:
        - Returns k_min if mask has no valid pixels
        - Rounds to nearest integer
        - Clamps result to [k_min, k_max] range
    """
    area = float(np.sum(mask))
    if area <= 0:
        return k_min
    
    k_ideal = area / max(area_per_seed, 1.0)
    k = int(round(k_ideal))
    return max(k_min, min(k, k_max))


def generate_bracing_adaptive(
    profile_fields_2d: np.ndarray,
    iso_level: float,
    nx: int,
    ny: int,
    area_per_seed: float = 1500.0,
    k_min: int = 2,
    k_max: int = 12,
    seed: int = 42,
    smooth_sigma: float = 1.5,
    ramp_slices: int = 3
) -> np.ndarray:
    """
    Generate shape-adaptive bracing with automatic centroid count.
    
    Automatically determines k per slice based on mask area, provides smooth
    transitions when k changes via Gaussian smoothing and ramped weights.
    Better geometric quality than keyfield blend - maintains consistent
    cell sizes across varying profile shapes.
    
    Args:
        profile_fields_2d: Profile scalar fields (num_slices, nx*ny)
        iso_level: Threshold for profile mask
        nx: Grid width
        ny: Grid height
        area_per_seed: Target pixels per centroid (default: 1500.0)
        k_min: Minimum centroids per slice (default: 2)
        k_max: Maximum centroids per slice (default: 12)
        seed: Random seed for K-means
        smooth_sigma: Gaussian sigma for Z-axis centroid smoothing (default: 1.5)
        ramp_slices: Number of slices for weight ramp when k increases (default: 3)
    
    Returns:
        Bracing fields array (num_slices, nx*ny) with Voronoi SDF values
    
    Example:
        >>> profile = np.random.rand(60, 2500) - 0.5
        >>> bracing = generate_bracing_adaptive(
        ...     profile, 0.0, 50, 50,
        ...     area_per_seed=1500, k_min=3, k_max=8
        ... )
        >>> bracing.shape
        (60, 2500)
    
    Note:
        - Adapts k to shape area → uniform cell sizes
        - Smooth centroid trajectories (Gaussian filter along Z)
        - Ramped weights prevent sudden "pop-in" of new cells
        - Progress printed every 10 slices
    """
    from scipy.ndimage import gaussian_filter1d
    
    num_slices = profile_fields_2d.shape[0]
    bracing_fields = np.zeros_like(profile_fields_2d)
    
    # Step 1: Compute k-schedule based on area
    k_schedule = np.zeros(num_slices, dtype=int)
    for z in range(num_slices):
        slice_2d = profile_fields_2d[z].reshape((ny, nx))
        mask = get_profile_mask(slice_2d, iso_level)
        k_schedule[z] = compute_k_from_area(mask, area_per_seed, k_min, k_max)
    
    max_k = int(k_schedule.max())
    if max_k == 0:
        return bracing_fields
    
    # Step 2: Generate per-slice centroids with warm-start
    all_centroids_rc = np.full((num_slices, max_k, 2), np.nan)
    for z in range(num_slices):
        slice_2d = profile_fields_2d[z].reshape((ny, nx))
        mask = get_profile_mask(slice_2d, iso_level)
        k_active = k_schedule[z]
        
        if k_active == 0:
            continue
        
        # Warm-start from previous slice
        prev_centroids = None
        if z > 0 and k_schedule[z-1] > 0:
            prev_k = k_schedule[z-1]
            if prev_k >= k_active:
                # Same or fewer centroids → use subset
                prev_centroids = all_centroids_rc[z-1, :k_active]
            else:
                # More centroids needed → use all previous + init new randomly
                prev_centroids = all_centroids_rc[z-1, :prev_k]
        
        centroids = generate_centroids(mask, k=k_active, prev_centroids=prev_centroids, seed=seed + z)
        centroids = constrain_centroids_to_mask(centroids, mask)
        all_centroids_rc[z, :k_active] = centroids
    
    # Step 3: Smooth centroid trajectories along Z with Gaussian filter
    for i in range(max_k):
        for coord_idx in [0, 1]:  # r, c
            trajectory = all_centroids_rc[:, i, coord_idx].copy()
            
            # Find valid (non-NaN) range
            valid_mask = ~np.isnan(trajectory)
            if not np.any(valid_mask):
                continue
            
            # Apply Gaussian smoothing only to valid range
            if smooth_sigma > 0:
                # Replace NaNs with forward/backward fill for smoothing
                filled = trajectory.copy()
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    first_valid = valid_indices[0]
                    last_valid = valid_indices[-1]
                    
                    # Forward fill
                    for z in range(first_valid + 1, num_slices):
                        if np.isnan(filled[z]):
                            filled[z] = filled[z-1]
                    
                    # Backward fill
                    for z in range(first_valid - 1, -1, -1):
                        if np.isnan(filled[z]):
                            filled[z] = filled[z+1]
                    
                    # Smooth
                    smoothed = gaussian_filter1d(filled, sigma=smooth_sigma, mode='nearest')
                    
                    # Restore NaN mask
                    smoothed[~valid_mask] = np.nan
                    all_centroids_rc[:, i, coord_idx] = smoothed
    
    # Step 4: Compute activation weights (ramp-up for new centroids)
    weights = np.zeros((num_slices, max_k))
    birth_z = np.full(max_k, -1, dtype=int)  # Track when each centroid first appeared
    
    for z in range(num_slices):
        k_curr = k_schedule[z]
        for i in range(k_curr):
            if not np.isnan(all_centroids_rc[z, i, 0]):
                # Check if this is first appearance
                if birth_z[i] < 0:
                    birth_z[i] = z
                
                # Ramp weight based on age
                age = z - birth_z[i]
                weights[z, i] = min(1.0, (age + 1) / max(ramp_slices, 1))
    
    # Step 5: Generate Voronoi ridge fields
    for z in range(num_slices):
        slice_2d = profile_fields_2d[z].reshape((ny, nx))
        mask = get_profile_mask(slice_2d, iso_level)
        k_active = k_schedule[z]
        
        if k_active == 0:
            continue
        
        centroids_rc = all_centroids_rc[z, :k_active]
        valid_centroids = centroids_rc[~np.isnan(centroids_rc[:, 0])]
        
        if len(valid_centroids) == 0:
            continue
        
        # Weighted Voronoi ridge
        w = weights[z, :k_active]
        ridge = _compute_weighted_voronoi_ridge((ny, nx), valid_centroids, w)
        
        ridge[~mask] = -9999  # Outside mask = solid
        bracing_fields[z] = ridge.ravel()
        
        if z % 10 == 0:
            print(f"Generated adaptive bracing for slice {z}/{num_slices} (k={k_active})")
    
    return bracing_fields


def _compute_weighted_voronoi_ridge(shape: Tuple[int, int], centroids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Internal helper: Compute weighted Voronoi ridge (d2 - d1).
    
    Args:
        shape: (ny, nx) tuple
        centroids: (k, 2) array of [y, x] positions
        weights: (k,) array of influence weights
    
    Returns:
        Ridge field (ny, nx) where positive values are cell boundaries
    """
    ny, nx = shape
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    if len(centroids) == 0:
        return np.zeros((ny, nx), dtype=float)
    
    # Compute weighted distances to all centroids
    dists = np.zeros((ny, nx, len(centroids)))
    for i, (cy, cx) in enumerate(centroids):
        w_i = weights[i] if i < len(weights) else 1.0
        # Avoid division by zero
        w_i = max(w_i, 0.01)
        dists[:, :, i] = np.sqrt((Y - cy)**2 + (X - cx)**2) / w_i
    
    # Find 1st and 2nd nearest
    if len(centroids) == 1:
        # Only one centroid → ridge is zero everywhere
        return np.zeros((ny, nx), dtype=float)
    
    sorted_dists = np.sort(dists, axis=2)
    d1 = sorted_dists[:, :, 0]
    d2 = sorted_dists[:, :, 1]
    
    ridge = d2 - d1
    return ridge

