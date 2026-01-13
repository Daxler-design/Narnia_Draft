"""
Morphological post-processing and temporal smoothing for bracing fields.

Functions for cleaning bracing patterns using:
- Binary morphological operations (dilation, erosion, closing)
- Connected component analysis and small island removal
- Temporal majority voting for temporal stability
"""

import numpy as np
from .data_utils import infer_grid_from_scalar_fields


def _binary_dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    """
    Dilate a binary mask using a circular structuring element.
    
    Pure NumPy implementation using repeated OR with shifted masks.
    
    Args:
        mask: Binary mask (h, w) to dilate
        radius: Dilation radius in pixels
    
    Returns:
        Dilated binary mask (h, w)
    
    Example:
        >>> mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        >>> dilated = _binary_dilate(mask, radius=1.0)
        >>> dilated.astype(int)
        array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]])
    
    Note:
        - Uses circular structuring element (Euclidean distance)
        - Efficient for small radii (< 10 pixels)
        - Returns unchanged mask if radius < 0.5
    """
    if radius < 0.5:
        return mask
    h, w = mask.shape
    r_int = int(radius)
    out = np.copy(mask)
    
    for dy in range(-r_int, r_int + 1):
        for dx in range(-r_int, r_int + 1):
            if dy*dy + dx*dx <= radius*radius:
                if dy == 0 and dx == 0:
                    continue
                # Shift mask using slicing for speed in pure numpy
                shifted = np.zeros_like(mask)
                y_start = max(0, dy)
                y_end = min(h, h + dy)
                x_start = max(0, dx)
                x_end = min(w, w + dx)
                y_o_start = max(0, -dy)
                y_o_end = min(h, h - dy)
                x_o_start = max(0, -dx)
                x_o_end = min(w, w - dx)
                shifted[y_start:y_end, x_start:x_end] = mask[y_o_start:y_o_end, x_o_start:x_o_end]
                out |= shifted
    return out


def _binary_erode(mask: np.ndarray, radius: float) -> np.ndarray:
    """
    Erode a binary mask using a circular structuring element.
    
    Implemented as dual of dilation: erode(M) = ~dilate(~M)
    
    Args:
        mask: Binary mask (h, w) to erode
        radius: Erosion radius in pixels
    
    Returns:
        Eroded binary mask (h, w)
    
    Example:
        >>> mask = np.ones((5, 5), dtype=bool)
        >>> eroded = _binary_erode(mask, radius=1.0)
        >>> eroded.shape
        (5, 5)
    
    Note:
        - Uses dilation-negation duality for implementation
        - Returns unchanged mask if radius < 0.5
    """
    if radius < 0.5:
        return mask
    # Erosion is the dual of dilation: erode(M) = ~dilate(~M)
    return ~_binary_dilate(~mask, radius)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask using BFS.
    
    Pure NumPy/Python implementation without scipy dependencies.
    Uses 4-connectivity (orthogonal neighbors only).
    
    Args:
        mask: Binary mask (h, w) to label
    
    Returns:
        Tuple of (labels, count) where:
        - labels: Integer array (h, w) with component labels 1..count
        - count: Number of connected components found
    
    Example:
        >>> mask = np.array([[1, 0, 1],
        ...                  [1, 0, 1],
        ...                  [0, 0, 1]], dtype=bool)
        >>> labels, count = _label_components(mask)
        >>> count
        2
        >>> labels
        array([[1, 0, 2],
               [1, 0, 2],
               [0, 0, 2]])
    
    Note:
        - Uses breadth-first search with stack-based traversal
        - Background (False) pixels labeled as 0
        - Efficient for sparse masks
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label_count = 0
    # Use 1D indices for faster stack operations
    flat_mask = mask.ravel()
    flat_labels = labels.ravel()
    
    for i in range(h * w):
        if flat_mask[i] and flat_labels[i] == 0:
            label_count += 1
            stack = [i]
            flat_labels[i] = label_count
            while stack:
                curr = stack.pop()
                cy, cx = divmod(curr, w)
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        ni = ny * w + nx
                        if flat_mask[ni] and flat_labels[ni] == 0:
                            flat_labels[ni] = label_count
                            stack.append(ni)
    return labels, label_count


def postprocess_bracing_fields(
    bracing_fields_2d: np.ndarray,
    profile_fields_2d: np.ndarray,
    iso_profile: float,
    iso_brace: float = 0.0,
    close_radius: int = 2,
    min_area: int = 120,
    temporal_window: int = 3,
    outside_value: float = -1e6
) -> np.ndarray:
    """
    Apply morphological cleaning and temporal smoothing to bracing fields.
    
    Processing pipeline:
    1. Mask by profile (only process interior region)
    2. Binarize bracing field at iso_brace threshold
    3. Morphological closing (dilation → erosion) to fill gaps
    4. Remove small connected components (< min_area)
    5. Temporal majority voting across slices for stability
    6. Reconstruct signed distance field (inside=0.5, outside=-0.5)
    
    Args:
        bracing_fields_2d: Raw bracing fields (num_slices, nx*ny)
        profile_fields_2d: Profile mask fields (num_slices, nx*ny)
        iso_profile: Profile threshold (values < iso_profile are inside)
        iso_brace: Bracing threshold for binarization (default: 0.0)
        close_radius: Morphological closing radius in pixels (default: 2)
        min_area: Minimum component area to keep (default: 120)
        temporal_window: Window size for temporal voting (default: 3)
        outside_value: Value for points outside profile (default: -1e6)
    
    Returns:
        Cleaned bracing fields (num_slices, nx*ny) with values:
        - 0.5 inside bracing (stable islands)
        - -0.5 outside bracing (but inside profile)
        - outside_value outside profile mask
    
    Example:
        >>> profile = np.random.rand(60, 2500) - 0.5
        >>> bracing = np.random.rand(60, 2500) - 0.3
        >>> cleaned = postprocess_bracing_fields(
        ...     bracing, profile, iso_profile=0.0, iso_brace=0.0,
        ...     close_radius=2, min_area=100, temporal_window=3
        ... )
        >>> cleaned.shape
        (60, 2500)
    
    Note:
        - Morphological closing removes small holes and gaps
        - Component removal eliminates noise islands
        - Temporal voting reduces flickering across slices
        - Output is suitable for direct meshing with marching cubes
    """
    num_fields, N = bracing_fields_2d.shape
    _, nx, ny = infer_grid_from_scalar_fields(bracing_fields_2d)
    
    m_stack = np.zeros((num_fields, ny, nx), dtype=bool)
    
    # Per-slice processing
    for i in range(num_fields):
        brac_slice = bracing_fields_2d[i].reshape((ny, nx))
        prof_slice = profile_fields_2d[i].reshape((ny, nx))
        
        # 1 & 2. Masking + Binarize
        mask = prof_slice < iso_profile
        M = (brac_slice >= iso_brace) & mask
        
        # 3. Morphological closing
        if close_radius > 0:
            M = _binary_dilate(M, close_radius)
            M = _binary_erode(M, close_radius)
            
        # 4. Remove small connected components
        if min_area > 0:
            labels, count = _label_components(M)
            if count > 0:
                for lbl in range(1, count + 1):
                    comp = (labels == lbl)
                    if np.sum(comp) < min_area:
                        M[comp] = False
        
        m_stack[i] = M

    # 5. Temporal stabilization (majority vote)
    if temporal_window > 1:
        m_stable = np.copy(m_stack)
        half = temporal_window // 2
        for i in range(num_fields):
            s = max(0, i - half)
            e = min(num_fields, i + half + 1)
            votes = np.sum(m_stack[s:e], axis=0)
            threshold = (e - s) // 2 + 1
            m_stable[i] = (votes >= threshold)
        m_stack = m_stable

    # 6. Reconstruct clean signed field
    out_fields = np.zeros_like(bracing_fields_2d)
    for i in range(num_fields):
        M_stable = m_stack[i]
        prof_slice = profile_fields_2d[i].reshape((ny, nx))
        mask = prof_slice < iso_profile
        
        # Inside=0.5, Outside=-0.5
        b_clean = M_stable.astype(float) - 0.5
        # Set outside profile mask to outside_value
        b_clean[~mask] = outside_value
        out_fields[i] = b_clean.ravel()
        
    return out_fields
