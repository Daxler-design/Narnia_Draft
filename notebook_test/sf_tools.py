
# import json
# from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def stack_scalar_fields(data_dict, prefix="scalar_field_values_"):
    
    """
    Extract all scalar field keys from a JSON dict and stack them
    into a 2D NumPy array, sorted by their numeric index.

    Returns:
        scalar_fields_2d : np.ndarray (num_fields, values_per_field)
        scalar_field_keys: list of keys in sorted order
    """
    # extract and sort keys like 'scalar_field_values_0', '..._1', etc.
    scalar_field_keys = sorted(
        [key for key in data_dict.keys() if key.startswith(prefix)],
        key=lambda x: int(x.split('_')[-1])
    )

    # stack into 2D array
    scalar_fields_2d = np.array([data_dict[key] for key in scalar_field_keys])

    print(f"Shape of 2D array: {scalar_fields_2d.shape}")
    print(f"Number of scalar fields: {scalar_fields_2d.shape[0]}")
    print(f"Values per field: {scalar_fields_2d.shape[1]}")

    return scalar_fields_2d, scalar_field_keys



def meta_data_info(data_dict):

    """
    Extract metadata information from the JSON dict.
    Returns:
        iso_level: float
        slice_count: int
        bounds_max: list
        bounds_min: list
    """


    bounds_max = data_dict.get("bounds_max", None)
    bounds_min = data_dict.get("bounds_min", None)
    iso_level = data_dict.get("iso_level", None) 
    slice_count = data_dict.get("slice_count", None)
    total_height = data_dict.get("total_height", None)
    print(f"Bounds Max: {bounds_max}")
    print(f"Bounds Min: {bounds_min}")
    print(f"Iso Level: {iso_level}")
    print(f"Slice Count: {slice_count}")
    print(f"Total Height: {total_height}")

    return iso_level, slice_count,bounds_max, bounds_min



def infer_grid_from_scalar_fields(scalar_fields_2d):
    """
    Infer (nx, ny) from scalar_fields_2d.shape[1] assuming a square grid.
    Returns (num_fields, nx, ny).
    """
    num_fields, values_per_field = scalar_fields_2d.shape
    n = int(np.sqrt(values_per_field))
    if n * n != values_per_field:
        raise ValueError(
            f"Cannot infer square grid from {values_per_field} values; "
            "set (nx, ny) manually."
        )
    nx = ny = n


    return num_fields, nx, ny



def plot_scalar_overview(
    scalar_fields_2d,
    field_name="",
    ncols=10,
    cmap="Greys",
):
    """
    Compact histogram + thumbnail grid for a 2D scalar field array.

    scalar_fields_2d: np.ndarray (num_fields, values_per_field)
    field_name: label to show in titles ("Bracing", "Profile", etc.)
    """
    num_fields, nx, ny = None, None, None

    # 1. infer grid
    num_fields, nx, ny = infer_grid_from_scalar_fields(scalar_fields_2d)
    print(f"[{field_name or 'field'}] Using grid size: nx={nx}, ny={ny}")
    print(f"[{field_name or 'field'}] Shape: {scalar_fields_2d.shape}")

    # 2. histogram
    plt.figure(figsize=(4, 2.5))
    plt.hist(scalar_fields_2d.ravel(), bins=100, color="tab:grey", alpha=0.8)
    plt.xlabel("Scalar value")
    plt.ylabel("Count")
    title = "Histogram of all scalar field values"
    if field_name:
        title += f" ({field_name})"
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # 3. thumbnail grid
    nrows = int(np.ceil(num_fields / ncols))
    thumb_w, thumb_h = 3.0, 1.8
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * thumb_w, nrows * thumb_h),
        squeeze=False,
    )

    for i in range(num_fields):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        img = scalar_fields_2d[i].reshape((ny, nx))
        ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(f"slice {i}", fontsize=8)
        ax.axis("off")

    # turn off unused subplots
    for j in range(num_fields, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()



def iso_curves_for_slice_2d(slice_2d, level, X, Y):
    
    """
    slice_2d: 2D array (ny, nx)
    level: iso value
    X, Y: coordinate grids (ny, nx)
    returns: list of (N_i, 2) arrays [x, y]
    """

    from matplotlib.path import Path as MplPath
    
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, slice_2d, levels=[level])

    curves = []
    for path in cs.get_paths():
        verts = path.vertices      # (N, 2)
        codes = path.codes         # (N,) or None

        # if no codes, treat as one polyline
        if codes is None:
            curves.append(verts.copy())
            continue

        current = []
        for v, code in zip(verts, codes):
            if code == MplPath.MOVETO:
                if current:
                    curves.append(np.array(current))
                    current = []
                current.append(v)
            elif code == MplPath.LINETO:
                current.append(v)
            elif code == MplPath.CLOSEPOLY:
                # close current polyline
                if current:
                    current.append(current[0])

        if current:
            curves.append(np.array(current))

    plt.close(fig)
    return curves




def compute_sf_operation(sf_A, sf_B, iso_level_A=0.0, iso_level_B=0.0, mode="difference", swap=False):

    """
    Compute an SDF boolean-like operation between two scalar fields.

    Parameters:
        sf_A, sf_B : array-like, shape (num_fields, values_per_field)
            Input scalar fields (signed distance fields or similar).
        iso_level_A, iso_level_B : float
            Iso-level offsets to subtract from each field before the operation
            (so the iso-curve is moved to 0).
        mode : str, one of {"difference", "union", "intersection"}
            - "difference"    : A \ B  -> max(A, -B)
            - "union"         : A ∪ B  -> min(A, B)
            - "intersection"  : A ∩ B  -> max(A, B)
        swap : bool
            If True, swap A and B before performing the operation.

    Returns:
        result_scalar_fields_2d : np.ndarray with same shape as inputs
    """
    A = np.asarray(sf_A, dtype=float) - iso_level_A
    B = np.asarray(sf_B, dtype=float) - iso_level_B

    if A.shape != B.shape:
        raise ValueError("Input scalar fields must have the same shape")

    if swap:
        A, B = B, A

    m = mode.lower()
    if m in ("difference", "a_minus_b", "sub"):
        # keep points inside A but outside B
        result = np.maximum(A, -B)
    elif m in ("union", "or", "min"):
        # inside if inside A or B
        result = np.minimum(A, B)
    elif m in ("intersection", "and", "max"):
        # inside only if inside A and B
        result = np.maximum(A, B)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported: difference, union, intersection")

    return result



def show_slice(idx, scalar_field, iso_level=0.0, X=1.0  , Y=1.0):
    # compute result field with current deltas
    
    x,y, nx, ny = infer_grid_from_scalar_fields(scalar_field)[1:]

    slice_2d = scalar_field[idx].reshape((ny, nx))




    plt.figure(figsize=(4, 4))
    plt.imshow(
        slice_2d,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()]
    )
    plt.colorbar(label="Scalar value")
    plt.contour(X, Y, slice_2d, levels=[iso_level], colors="white", linewidths=1.0)
    plt.title(f"Slice {idx} with iso-level\n{round(iso_level, 6)}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
