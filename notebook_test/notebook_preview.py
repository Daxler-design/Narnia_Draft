import numpy as np
import matplotlib.pyplot as plt
import core

def plot_scalar_overview(
    scalar_fields_2d,
    field_name="",
    ncols=10,
    cmap="Greys",
):
    """
    Compact histogram + thumbnail grid for a 2D scalar field array.
    """
    num_fields, nx, ny = core.infer_grid_from_scalar_fields(scalar_fields_2d)
    
    # 1. Histogram
    plt.figure(figsize=(4, 2.5))
    plt.hist(scalar_fields_2d.ravel(), bins=100, color="tab:grey", alpha=0.8)
    plt.xlabel("Scalar value")
    plt.ylabel("Count")
    title = f"Histogram of all scalar field values ({field_name})" if field_name else "Histogram of all scalar field values"
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # 2. Thumbnail grid
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

    for j in range(num_fields, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()

def show_slice(idx, scalar_field, iso_level=0.0, X=None, Y=None):
    """
    Visualize a single slice with its iso-level contour.
    """
    num_fields, nx, ny = core.infer_grid_from_scalar_fields(scalar_field)
    slice_2d = scalar_field[idx].reshape((ny, nx))

    plt.figure(figsize=(4, 4))
    if X is not None and Y is not None:
        extent = [X.min(), X.max(), Y.min(), Y.max()]
        plt.imshow(slice_2d, origin="lower", extent=extent)
        plt.contour(X, Y, slice_2d, levels=[iso_level], colors="white", linewidths=1.0)
    else:
        plt.imshow(slice_2d, origin="lower")
        plt.contour(slice_2d, levels=[iso_level], colors="white", linewidths=1.0)
        
    plt.colorbar(label="Scalar value")
    plt.title(f"Slice {idx} with iso-level\n{round(iso_level, 6)}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
