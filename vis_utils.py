import os

# --- 1. STABILITY & GPU SETTINGS ---
# Cap common thread pools to prevent system-wide freezes on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

# --- 2. USER CONFIGURATION ---
PREFERRED_MONITOR_INDEX = 1  # Try 1 for external (dGPU), 0 for laptop (iGPU)
HEADLESS_MODE = False        # Set to True if the 3D viewer freezes your PC
FORCE_SOFTWARE_GL = False    # Set to True to attempt CPU rendering (requires Mesa)

if FORCE_SOFTWARE_GL:
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["GALLIUM_DRIVER"] = "llvmpipe"

import numpy as np
import open3d as o3d
from typing import Optional
import ctypes
from ctypes import wintypes

def print_system_diagnostics():
    print("--- System Diagnostics ---")
    try:
        monitors = get_monitors_info()
        print(f"Detected {len(monitors)} monitor(s):")
        for i, m in enumerate(monitors):
            status = "(Primary/Laptop)" if m["is_primary"] else "(Secondary/External)"
            print(f"  [{i}] {status} Bounds: {m['rect']}")
    except Exception as e:
        print(f"Could not detect monitors: {e}")
    print("--------------------------\n")

def get_monitors_info():
    """Returns a list of monitor information dictionaries using Windows API."""
    monitors = []
    
    # Define callback for EnumDisplayMonitors
    def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
        rect = lprcMonitor.contents
        
        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]
        
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        ctypes.windll.user32.GetMonitorInfoW(hMonitor, ctypes.byref(info))
        
        is_primary = bool(info.dwFlags & 1) # MONITORINFOF_PRIMARY
        
        monitors.append({
            "rect": (rect.left, rect.top, rect.right, rect.bottom),
            "is_primary": is_primary
        })
        return True

    # Define the callback type
    MONITORENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    
    ctypes.windll.user32.EnumDisplayMonitors(None, None, MONITORENUMPROC(callback), 0)
    return monitors

def generate_mesh_from_curves(all_curves, bounds_min, bounds_max):
    """Placeholder for future mesh generation."""
    raise NotImplementedError("Mesh generation from curves is not implemented yet.")


def normalize_to_u8(img_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_2d, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    vmin = np.percentile(arr[finite], 2)
    vmax = np.percentile(arr[finite], 98)
    if vmax <= vmin:
        vmax = vmin + 1.0

    scaled = (np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin)
    return (scaled * 255.0).astype(np.uint8)


def curves_to_lineset(curves_2d: list[np.ndarray], z: float) -> Optional[o3d.geometry.LineSet]:
    points = []
    lines = []
    cursor = 0
    for curve in curves_2d:
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 2 or curve.shape[0] < 2:
            continue
        pts3 = np.column_stack([curve[:, 0], curve[:, 1], np.full(curve.shape[0], z)])
        points.append(pts3)
        lines.extend([[cursor + i, cursor + i + 1] for i in range(curve.shape[0] - 1)])
        cursor += curve.shape[0]

    if not points or not lines:
        return None

    pts = np.vstack(points)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    return ls


def make_textured_plane(bounds_min, bounds_max, z: float):
    x0, y0 = float(bounds_min[0]), float(bounds_min[1])
    x1, y1 = float(bounds_max[0]), float(bounds_max[1])
    verts = np.array(
        [
            [x0, y0, z],
            [x1, y0, z],
            [x1, y1, z],
            [x0, y1, z],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uvs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs[[0, 1, 2, 0, 2, 3]])
    mesh.compute_triangle_normals()
    return mesh


def scalar_to_overlay_colors(values_2d: np.ndarray, opacity: float, bg_rgb=(0.07, 0.07, 0.07)) -> np.ndarray:
    """Map scalar values to grayscale RGB, blended with background by `opacity`.

    Open3D point clouds don't reliably support per-vertex alpha in all backends,
    so we approximate opacity by blending against the background color.
    """
    opacity = float(np.clip(opacity, 0.0, 1.0))
    u8 = normalize_to_u8(values_2d)
    gray = (u8.astype(np.float32) / 255.0)
    rgb = np.stack([gray, gray, gray], axis=-1)
    bg = np.array(bg_rgb, dtype=np.float32).reshape((1, 1, 3))
    blended = rgb * opacity + bg * (1.0 - opacity)
    return blended.reshape((-1, 3)).astype(np.float32)


def slice_z(slice_index: int, num_fields: int, bounds_min, bounds_max) -> float:
    z0 = float(bounds_min[2])
    z1 = float(bounds_max[2])
    if num_fields <= 1:
        return z0
    return z0 + (z1 - z0) * (slice_index / (num_fields - 1))
