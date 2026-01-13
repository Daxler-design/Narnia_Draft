"""
ISO-curve extraction from 2D scalar field slices.

Functions for extracting contour lines at specific iso-values using the
contourpy library for fast contour generation.
"""

import numpy as np
import contourpy as _contourpy
from typing import List


def iso_curves_for_slice_2d(slice_2d: np.ndarray, level: float, X: np.ndarray, Y: np.ndarray) -> List[np.ndarray]:
    """
    Extract iso-curves from a 2D slice using a fast contour engine (contourpy).

    Args:
        slice_2d: 2D array with shape (ny, nx) containing scalar field values
        level: Iso-value at which to extract contours
        X: X-coordinate grid (ny, nx) from np.meshgrid(..., indexing="xy")
           or 1D array of x-coordinates
        Y: Y-coordinate grid (ny, nx) from np.meshgrid(..., indexing="xy")
           or 1D array of y-coordinates
    
    Returns:
        List of contour polylines, each as (N_i, 2) array with columns [x, y]
    
    Example:
        >>> ny, nx = 50, 50
        >>> x = np.linspace(0, 100, nx)
        >>> y = np.linspace(0, 100, ny)
        >>> X, Y = np.meshgrid(x, y, indexing="xy")
        >>> field = np.random.rand(ny, nx) - 0.5
        >>> curves = iso_curves_for_slice_2d(field, level=0.0, X=X, Y=Y)
        >>> len(curves)
        5  # Number of separate contour lines
    
    Note:
        - Uses contourpy's serial backend for thread-safe operation
        - X and Y can be either 2D grids or 1D coordinate arrays
        - Returns empty list if no contours found at the given level
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Extract 1D coordinate arrays from grids if needed
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
