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

def generate_bracing_key_blending(profile_fields_2d, iso_level, nx, ny, k, key_step=5, smooth=0.0, seed=42):
    """
    Generate bracing fields using key-frame blending.
    """
    import itertools
    
    num_fields = profile_fields_2d.shape[0]
    bracing_fields = np.zeros_like(profile_fields_2d)
    
    # 1. Identify key slices
    key_indices = list(range(0, num_fields, key_step))
    if key_indices[-1] != num_fields - 1:
        key_indices.append(num_fields - 1)
        
    # 2. Generate centroids for key slices
    key_centroids = {}
    for idx in key_indices:
        slice_2d = profile_fields_2d[idx].reshape((ny, nx))
        mask = get_profile_mask(slice_2d, iso_level=iso_level)
        # Independent K-Means for each key frame
        centroids = generate_centroids(mask, k=k, prev_centroids=None, seed=seed)
        key_centroids[idx] = centroids

    # 3. Match and Interpolate
    for i in range(len(key_indices) - 1):
        idx_start = key_indices[i]
        idx_end = key_indices[i+1]
        
        c_start = key_centroids[idx_start]
        c_end = key_centroids[idx_end]
        
        # Match c_end to c_start using brute-force permutation (k is small)
        best_perm = None
        min_dist = float('inf')
        
        perms = list(itertools.permutations(range(k)))
        for perm in perms:
            dist = 0
            for j in range(k):
                dist += np.linalg.norm(c_start[j] - c_end[perm[j]])
            if dist < min_dist:
                min_dist = dist
                best_perm = perm
                
        c_end_ordered = c_end[list(best_perm)]
        
        # Update the stored key centroid for the next segment start
        key_centroids[idx_end] = c_end_ordered
        c_end = c_end_ordered
        
        # Interpolate
        steps = idx_end - idx_start
        for j in range(steps):
            curr_idx = idx_start + j
            t = j / float(steps)
            
            # Apply smoothing to t (Linear -> Smoothstep)
            if smooth > 0:
                t_smooth = t * t * (3 - 2 * t)
                t = (1 - smooth) * t + smooth * t_smooth
            
            # Interpolation
            c_curr = (1 - t) * c_start + t * c_end
            
            # Constrain to mask
            slice_2d = profile_fields_2d[curr_idx].reshape((ny, nx))
            mask = get_profile_mask(slice_2d, iso_level=iso_level)
            c_curr = constrain_centroids_to_mask(c_curr, mask)
            
            # Compute SDF
            voronoi_sdf_flat = compute_voronoi_sdf((ny, nx), c_curr)
            bracing_fields[curr_idx] = voronoi_sdf_flat.ravel()
            
            if curr_idx % 10 == 0:
                print(f"Generated bracing for slice {curr_idx}/{num_fields}")

    # Handle the very last frame
    last_idx = key_indices[-1]
    slice_2d = profile_fields_2d[last_idx].reshape((ny, nx))
    mask = get_profile_mask(slice_2d, iso_level=iso_level)
    c_last = constrain_centroids_to_mask(key_centroids[last_idx], mask)
    voronoi_sdf_flat = compute_voronoi_sdf((ny, nx), c_last)
    bracing_fields[last_idx] = voronoi_sdf_flat.ravel()
    
    return bracing_fields
