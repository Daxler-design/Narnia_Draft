"""
Docstring for debug_utils
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # avoids GUI/OpenGL teardown issues
import matplotlib.pyplot as plt

from pathlib import Path

def output_debug_voronoi(voronoi_sdf, centroids, slice_idx):
    # DEBUG: output scalar field to the folder with colours and centroids, using plt
    
    # specify output path relative to this script's location
    debug_dir = (Path(__file__).parent / "debug_output").resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    i = slice_idx

        

    plt.figure(figsize=(10, 8))
    plt.imshow(voronoi_sdf, cmap='magma')
    plt.scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x', label='Centroids')
    plt.title(f"Slice {i} - Bracing SDF (k={len(centroids)})")
    plt.colorbar()
    
    path = debug_dir / f"debug_voronoi_{i:03d}.png"
    plt.savefig(str(path))
    plt.close()



def output_debug_plot(phi, centroids, slice_idx):
    """
    Diagnostic plot for checking whether `phi` behaves like a true signed distance field (SDF).

    Shows:
      1) signed field with diverging scale centered at 0
      2) the 0-level set contour
      3) gradient magnitude |∇phi| (should be ~1 for a true SDF)
      4) summary stats in the title
    """
    if centroids is None:
        centroids = np.empty((0, 2), dtype=float)
    debug_dir = (Path(__file__).parent / "debug_output").resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    phi = np.asarray(phi, dtype=float)

    # Gradient magnitude diagnostic (Eikonal condition)
    gy, gx = np.gradient(phi)
    g = np.sqrt(gx * gx + gy * gy)

    # Symmetric display range around 0 (so sign is obvious)
    m = float(max(abs(phi.min()), abs(phi.max())))
    if m == 0:
        m = 1.0

    # Stats for the title
    minv, maxv = float(phi.min()), float(phi.max())
    g_med = float(np.median(g))
    g_p10, g_p90 = np.percentile(g, [10, 90]).astype(float)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: signed field centered at 0, with 0-contour ---
    im0 = axs[0].imshow(phi, vmin=-m, vmax=m, cmap="coolwarm")
    axs[0].scatter(centroids[:, 1], centroids[:, 0], c="red", marker="x", label="Centroids")
    axs[0].contour(phi, levels=[0.0], colors="white", linewidths=2)  # boundary
    axs[0].set_title("Signed field (centered at 0) + 0-contour")
    axs[0].legend(loc="upper right", frameon=True)
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # --- Right: gradient magnitude ---
    # For a true SDF, |∇phi| ≈ 1 almost everywhere (except at junctions)
    # im1 = axs[1].imshow(g, cmap="viridis")
    # axs[1].set_title("|∇phi| (true SDF ≈ 1)")

    # Overlay phi contours on top of |∇phi| to see spacing distortion
    # levels = np.linspace(-m, 0, num=10)
    # axs[1].contour(
    #     phi,
    #     levels=levels,
    #     colors="white",
    #     linewidths=0.5,
    #     alhpha=0.7
    # )

    im1 = axs[1].imshow(g, cmap="viridis")
    axs[1].set_title("|∇phi| (diagnostic)")

    t_in = float(-phi.min())
    if t_in > 0:
        levels = np.linspace(-t_in, 0.0, num=10)
    else:
        levels = [0.0]

    axs[1].contour(
        phi,
        levels=levels,
        colors="white",
        linewidths=0.5,
        alpha=0.7
    )


    # make theatal lines more visible
    axs[1].contour(phi,levels=[0.0], colors="white", linewidths=2)



    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Slice {slice_idx} | min/max {minv:.3f}/{maxv:.3f} | "
        f"median|∇| {g_med:.3f} (p10 {g_p10:.3f}, p90 {g_p90:.3f}) | k={len(centroids)}",
        fontsize=12
    )
    fig.tight_layout()

    path = debug_dir / f"debug_plot_{slice_idx:03d}.png"
    plt.savefig(str(path), dpi=200)
    plt.close(fig)




def output_debug_plot_world_offsets(
    phi,
    bbox_min,
    bbox_max,
    nx,
    ny,
    slice_idx,
    offset_step_world=0.02,
    offset_max_world=None,
    inside_only=True,
    centroids=None,
    save_dir=None,
    dpi=200,
    file_name_prefix="plot",
    show_offset=True,
):
    """
    Plot an SDF with contours that correspond to exact world-space offsets.

    Args:
        phi: (ny, nx) signed distance field in WORLD UNITS (same units as bbox).
        bbox_min/bbox_max: [x,y,z] lists from JSON bbox.
        nx, ny: grid resolution used to generate phi.
        slice_idx: for filename/title.
        offset_step_world: contour spacing in WORLD UNITS (e.g. 0.02 = 2cm).
        offset_max_world: max offset distance to draw (WORLD UNITS). If None, uses available range.
        inside_only: if True, draw only inside (negative) offsets + 0 contour.
        centroids: optional (k,2) in pixel coords (row,col). Can pass None.
        save_dir: folder path; defaults to ./debug_output next to this file.
        dpi: save dpi.
        file_name_prefix: optional prefix for the saved filename.
        show_offset: if True, draw multiple offset contours; if False, only draw level=0.
    """
    if centroids is None:
        centroids = np.empty((0, 2), dtype=float)

    phi = np.asarray(phi, dtype=float)
    assert phi.shape == (ny, nx), f"phi shape {phi.shape} must match (ny,nx)=({ny},{nx})"

    xmin, ymin = float(bbox_min[0]), float(bbox_min[1])
    xmax, ymax = float(bbox_max[0]), float(bbox_max[1])

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # World-space gradient magnitude (meaningful for true SDF: ~1)
    gy, gx = np.gradient(phi, dy, dx)
    g = np.sqrt(gx * gx + gy * gy)

    # For display: center colormap at 0
    m = float(max(abs(phi.min()), abs(phi.max())))
    if m == 0:
        m = 1.0

    # Determine available inside/outside ranges
    in_max = float(-phi.min())  # max distance inside (<=0)
    out_max = float(phi.max())  # max distance outside (>=0)

    # Choose max offset to draw
    if offset_max_world is None:
        offset_max_world = in_max if inside_only else min(in_max, out_max)
    offset_max_world = float(max(0.0, offset_max_world))

    # Build contour levels in WORLD UNITS
    if offset_step_world <= 0:
        raise ValueError("offset_step_world must be > 0")

    if inside_only:
        # levels must be increasing: most negative -> 0
        neg_levels = -np.arange(offset_step_world, offset_max_world + 1e-12, offset_step_world)
        levels = np.r_[neg_levels[::-1], 0.0]  # increasing: (-big ... -step, 0)
    else:
        # symmetric around 0: (-max ... -step, 0, +step ... +max)
        pos_levels = np.arange(offset_step_world, offset_max_world + 1e-12, offset_step_world)
        neg_levels = -pos_levels
        levels = np.r_[neg_levels[::-1], 0.0, pos_levels]  # increasing overall

    if not show_offset:
        levels = np.array([0.0])

    # Setup figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # print(f"DEBUG: offset contours at world distances: {levels[-2:]}")
    # Left: SDF field (world-space offsets overlaid)
    im0 = axs[0].imshow(phi, vmin=-m, vmax=m, cmap="coolwarm", extent=[xmin, xmax, ymin, ymax], origin="lower")
    if centroids.shape[0] > 0:
        # centroids are in pixel coords (row, col); convert to world coords for plotting
        cx = xmin + (centroids[:, 1] / (nx - 1)) * (xmax - xmin)
        cy = ymin + (centroids[:, 0] / (ny - 1)) * (ymax - ymin)
        axs[0].scatter(cx, cy, c="red", marker="x", label="Centroids")
        axs[0].legend(loc="upper right", frameon=True)

    c0 = axs[0].contour(
        phi, levels=levels[-2:], colors="white",
        linewidths=0.8, alpha=0.9,
        extent=[xmin, xmax, ymin, ymax], origin="lower"
    ) # only label the last two contours for clarity

    # Label contour values as world distances
    axs[0].clabel(c0, inline=True, fontsize=8, fmt=lambda v: f"{v:.3f}")

    axs[0].set_title("SDF with world-space offset contours")
    axs[0].set_xlabel("X (world)")
    axs[0].set_ylabel("Y (world)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # Right: |∇phi| diagnostic, also in world spacing, overlay same offset contours
    im1 = axs[1].imshow(g, cmap="viridis", extent=[xmin, xmax, ymin, ymax], origin="lower")
    axs[1].contour(
        phi, levels=levels, colors="white",
        linewidths=0.6, alpha=0.8,
        extent=[xmin, xmax, ymin, ymax], origin="lower"
    )
    
    axs[1].set_title("|∇phi| (world spacing) + same offset contours")
    axs[1].set_xlabel("X (world)")
    axs[1].set_ylabel("Y (world)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # Stats
    g_med = float(np.median(g))
    g_p10, g_p90 = np.percentile(g, [10, 90]).astype(float)
    fig.suptitle(
        f"Slice {slice_idx} | dx={dx:.6g}, dy={dy:.6g} | "
        f"phi min/max {phi.min():.4g}/{phi.max():.4g} | "
        f"median|∇| {g_med:.3f} (p10 {g_p10:.3f}, p90 {g_p90:.3f}) | "
        f"offset step={offset_step_world:g}",
        fontsize=12
    )
    fig.tight_layout()


    # Save
    if save_dir is None:
        save_dir = (Path(__file__).parent / "debug_output").resolve()
    else:
        save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)


    path = save_dir / f"debug_world_offsets_{file_name_prefix}_{slice_idx:03d}.png"
    plt.savefig(str(path), dpi=dpi)
    plt.close(fig)

    return str(path)



def _to_img(field: np.ndarray, nx: int, ny: int, slice_idx: int = 0) -> np.ndarray:
    field = np.asarray(field)

    # already (ny, nx)
    if field.ndim == 2 and field.shape == (ny, nx):
        return field

    # stacked (num_slices, nx*ny)
    if field.ndim == 2 and field.shape[1] == nx * ny:
        return field[slice_idx].reshape(ny, nx)

    # flat (nx*ny,)
    if field.ndim == 1 and field.size == nx * ny:
        return field.reshape(ny, nx)

    raise ValueError(f"Can't plot field: shape={field.shape}, expected (ny,nx)=({ny},{nx}) or (S,nx*ny).")

    
def _save_debug_boolean_show(A, B, R, nx: int, ny: int, slice_idx: int = 0, tag: str = ""):
    debug_dir = (Path.cwd() / "debug_output").resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    A_img = _to_img(A, nx, ny, slice_idx)
    R_img = _to_img(R, nx, ny, slice_idx)
    B_img = _to_img(B, nx, ny, slice_idx)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    im0 = axs[0].imshow(A_img, cmap="magma", aspect="equal")
    axs[0].set_title(f"A (raw)-slice {slice_idx}")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(B_img, cmap="magma", aspect="equal")
    axs[1].set_title(f"B (slice {slice_idx})")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(R_img, cmap="magma", aspect="equal")
    axs[2].set_title(f"Result (slice {slice_idx})")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    out = debug_dir / f"Debug_boolean_show{('_'+tag) if tag else ''}_s{slice_idx:03d}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"DEBUG: wrote {out}")

