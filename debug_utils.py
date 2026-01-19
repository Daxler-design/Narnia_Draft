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

    
def _save_debug_A_offsetA_B(A, offset_A, B, nx: int, ny: int, slice_idx: int = 0, tag: str = ""):
    debug_dir = (Path.cwd() / "debug_output").resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    A_img = _to_img(A, nx, ny, slice_idx)
    off_img = _to_img(offset_A, nx, ny, slice_idx)
    B_img = _to_img(B, nx, ny, slice_idx)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    im0 = axs[0].imshow(A_img, cmap="magma", aspect="equal")
    axs[0].set_title(f"A (raw)-slice {slice_idx}")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(off_img, cmap="magma", aspect="equal")
    axs[1].set_title(f"offset_A = A - iso_level_A (slice {slice_idx})")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(B_img, cmap="magma", aspect="equal")
    axs[2].set_title(f"B (offset) (slice {slice_idx})")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    out = debug_dir / f"debug_A_offsetA_B{('_'+tag) if tag else ''}_s{slice_idx:03d}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"DEBUG: wrote {out}")

