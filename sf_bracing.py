import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

def get_profile_mask(field_2d, iso_level=0.0):
    """
    Returns a boolean mask where the field is inside the profile (value < iso_level).
    Assumes SDF convention where negative is inside.
    """
    return field_2d < iso_level

def generate_centroids(mask, k=3, prev_centroids=None, seed=42):
    """
    Generate k centroids for the given boolean mask using K-Means.
    
    Parameters:
        mask: 2D boolean array (ny, nx)
        k: number of centroids
        prev_centroids: (k, 2) array of centroids from previous slice (for coherence)
        seed: random seed
        
    Returns:
        centroids: (k, 2) array of [y, x] coordinates (matching array indexing)
    """
    # Get coordinates of interior points
    # np.argwhere returns (row, col) -> (y, x)
    coords = np.argwhere(mask)
    
    if len(coords) < k:
        # Fallback if slice is empty or too small
        # Return prev_centroids if available, else center of image
        if prev_centroids is not None:
            return prev_centroids
        else:
            ny, nx = mask.shape
            return np.array([[ny/2, nx/2]] * k)

    # Setup K-Means
    # If prev_centroids are provided, use them as initialization to ensure coherence
    if prev_centroids is not None:
        # We assume prev_centroids matches k
        # n_init=1 because we want to stick close to the initialization
        kmeans = KMeans(n_clusters=k, init=prev_centroids, n_init=1, random_state=seed)
    else:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=seed)
        
    kmeans.fit(coords)
    centroids = kmeans.cluster_centers_
    
    # Sort centroids to maintain consistent ordering if possible (e.g. by angle or x-coord)
    # This helps if K-Means permutes them, though init=prev usually keeps order.
    # For now, we rely on init=prev.
    
    return centroids

def compute_voronoi_sdf(shape, centroids):
    """
    Compute a scalar field where value = dist_to_2nd_nearest - dist_to_nearest.
    This field is 0 at Voronoi boundaries and positive inside cells.
    
    Parameters:
        shape: (ny, nx) tuple
        centroids: (k, 2) array of [y, x]
        
    Returns:
        sdf: 2D float array
    """
    ny, nx = shape
    
    # Create grid of coordinates
    # We want (y, x) for every pixel
    Y, X = np.indices(shape)
    # Flatten to (N, 2) points
    grid_points = np.stack([Y.ravel(), X.ravel()], axis=-1)
    
    # Use cKDTree for fast distance query
    tree = cKDTree(centroids)
    
    # Query 2 nearest neighbors
    # k=2 returns distances (N, 2) and indices (N, 2)
    dists, _ = tree.query(grid_points, k=2)
    
    dist_nearest = dists[:, 0]
    dist_second = dists[:, 1]
    
    # Voronoi SDF: d2 - d1
    # Boundary is at 0. Positive inside cells.
    # We might want to normalize or scale this, but raw distance diff is standard.
    sdf_flat = dist_second - dist_nearest
    
    return sdf_flat.reshape(shape)

def constrain_centroids_to_mask(centroids, mask):
    """
    Ensure centroids are strictly inside the mask.
    If a centroid is outside, snap it to the nearest mask point.
    """
    ny, nx = mask.shape
    valid_coords = np.argwhere(mask)
    
    if len(valid_coords) == 0:
        return centroids # Nothing to do
        
    constrained = []
    for c in centroids:
        y, x = int(c[0]), int(c[1])
        # Check bounds
        if 0 <= y < ny and 0 <= x < nx and mask[y, x]:
            constrained.append(c)
        else:
            # Find nearest valid point
            # simple euclidean distance to all valid points
            dists = np.sum((valid_coords - c)**2, axis=1)
            nearest_idx = np.argmin(dists)
            constrained.append(valid_coords[nearest_idx])
            
    return np.array(constrained)
