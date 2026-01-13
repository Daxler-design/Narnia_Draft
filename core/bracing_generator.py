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
