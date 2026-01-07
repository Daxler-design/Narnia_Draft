import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
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

def generate_bracing_static(profile_fields_2d, iso_level, nx, ny, k, seed=42):
    """
    Generate static bracing fields based on profile fields centroids.
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
    keys_config: list of (slice_idx, k_centroids)
    
    Uses Ridge Response R = exp(-(V/sigma)^2) blending instead of raw Voronoi V blending.
    Output is B = R - tau.
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
        # V >= 0 usually. R in (0, 1]
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
        # Convert R -> B
        B_first = R_first - tau
        
        # Masking
        # We need to apply mask for each slice individually 
        # because the profile shape might change (though often similar)
        # Here we just iterate to fill
        for i in range(first_idx):
             slice_mask = get_profile_mask(profile_fields_2d[i].reshape((ny, nx)), iso_level=iso_level)
             field = B_first.copy().ravel()
             # Apply mask to flat array
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
                # Smooth Max Blending: (1/beta) * log((1-t)*exp(beta*R1) + t*exp(beta*R2))
                # To avoid overflow, use logaddexp logic indirectly or just clip beta*R if needed
                # Since R is in [0, 1], beta*R is comfortably within range for reasonable beta (<100)
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
