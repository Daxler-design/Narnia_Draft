import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import core


def generate_mesh_from_curves(all_curves, bounds_min, bounds_max):
    """Placeholder for future mesh generation."""
    raise NotImplementedError("Mesh generation from curves is not implemented yet.")


def _normalize_to_u8(img_2d: np.ndarray) -> np.ndarray:
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


def _curves_to_lineset(curves_2d: list[np.ndarray], z: float) -> Optional[o3d.geometry.LineSet]:
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


def _make_textured_plane(bounds_min, bounds_max, z: float):
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


def _scalar_to_overlay_colors(values_2d: np.ndarray, opacity: float, bg_rgb=(0.07, 0.07, 0.07)) -> np.ndarray:
    """Map scalar values to grayscale RGB, blended with background by `opacity`.

    Open3D point clouds don't reliably support per-vertex alpha in all backends,
    so we approximate opacity by blending against the background color.
    """
    opacity = float(np.clip(opacity, 0.0, 1.0))
    u8 = _normalize_to_u8(values_2d)
    gray = (u8.astype(np.float32) / 255.0)
    rgb = np.stack([gray, gray, gray], axis=-1)
    bg = np.array(bg_rgb, dtype=np.float32).reshape((1, 1, 3))
    blended = rgb * opacity + bg * (1.0 - opacity)
    return blended.reshape((-1, 3)).astype(np.float32)


class NarniaCurveViewer:
    def __init__(self):
        self._window = gui.Application.instance.create_window("Narnia Viewer", 1400, 900)

        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(self._window.renderer)
        self._scene_widget.scene.set_background([0.07, 0.07, 0.07, 1.0])

        # Small axis inset (bottom-left of viewport)
        self._axis_widget = gui.SceneWidget()
        self._axis_widget.scene = rendering.Open3DScene(self._window.renderer)
        self._axis_widget.scene.set_background([0.0, 0.0, 0.0, 0.0])
        self._axis_widget.scene.add_geometry(
            "axis_frame",
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
            rendering.MaterialRecord(),
        )

        # Initialize axis camera looking at origin; we'll keep it synced to main view.
        axis_bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([-0.5, -0.5, -0.5], dtype=float),
            max_bound=np.array([0.5, 0.5, 0.5], dtype=float),
        )
        self._axis_widget.setup_camera(60.0, axis_bbox, [0.0, 0.0, 0.0])

        self._panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self._panel.preferred_width = 360

        self._profile_path = gui.TextEdit()
        self._profile_path.text_value = "./alice_result/251120/ext/waveStackFields.json"

        self._chk_generate_bracing = gui.Checkbox("Generate Bracing from Centroids")
        self._chk_generate_bracing.checked = True
        self._chk_generate_bracing.set_on_checked(self._on_generate_toggled)

        self._bracing_path = gui.TextEdit()
        self._bracing_path.text_value = "./alice_result/251120/bracing/waveStackFields.json"
        self._bracing_path.enabled = False

        self._num_centroids_slider = gui.Slider(gui.Slider.INT)
        self._num_centroids_slider.set_limits(1, 20)
        self._num_centroids_slider.int_value = 5

        self._btn_compute = gui.Button("Load / Re-generate Bracing")
        self._btn_fit = gui.Button("Fit Camera")

        self._op_mode_combo = gui.Combobox()
        self._op_mode_combo.add_item("difference")
        self._op_mode_combo.add_item("union")
        self._op_mode_combo.add_item("intersection")
        self._op_mode_combo.set_on_selection_changed(self._on_boolean_param_changed)

        self._profile_offset_slider = gui.Slider(gui.Slider.DOUBLE)
        self._profile_offset_slider.set_limits(-5.0, 5.0)
        self._profile_offset_slider.double_value = 0.0
        self._profile_offset_slider.set_on_value_changed(self._on_boolean_param_changed)

        self._bracing_offset_slider = gui.Slider(gui.Slider.DOUBLE)
        self._bracing_offset_slider.set_limits(-5.0, 5.0)
        self._bracing_offset_slider.double_value = 0.0
        self._bracing_offset_slider.set_on_value_changed(self._on_boolean_param_changed)

        self._slice_slider = gui.Slider(gui.Slider.INT)
        self._slice_slider.set_limits(0, 0)
        self._slice_slider.int_value = 0

        self._iso_slider = gui.Slider(gui.Slider.DOUBLE)
        self._iso_slider.set_limits(-1.0, 1.0)
        self._iso_slider.double_value = 0.0

        self._status = gui.Label("Load data to start.")

        self._panel.add_child(gui.Label("Profile JSON path"))
        self._panel.add_child(self._profile_path)
        self._panel.add_fixed(6)
        self._panel.add_child(gui.Label("Bracing JSON path"))
        self._panel.add_child(self._bracing_path)
        self._panel.add_fixed(10)
        self._panel.add_child(self._chk_generate_bracing)
        self._panel.add_child(gui.Label("Num Centroids"))
        self._panel.add_child(self._num_centroids_slider)
        self._panel.add_fixed(10)
        self._panel.add_child(self._btn_compute)
        self._panel.add_fixed(16)
        self._panel.add_child(gui.Label("--- Boolean Operation ---"))
        self._panel.add_child(gui.Label("Mode"))
        self._panel.add_child(self._op_mode_combo)
        self._panel.add_child(gui.Label("Profile Offset"))
        self._panel.add_child(self._profile_offset_slider)
        self._panel.add_child(gui.Label("Bracing Offset"))
        self._panel.add_child(self._bracing_offset_slider)
        self._panel.add_fixed(16)
        self._panel.add_child(gui.Label("--- Visualization ---"))
        self._panel.add_child(gui.Label("Slice"))
        self._panel.add_child(self._slice_slider)
        self._panel.add_child(gui.Label("Result Iso threshold"))
        self._panel.add_child(self._iso_slider)
        self._panel.add_fixed(10)
        self._panel.add_child(self._btn_fit)
        self._panel.add_fixed(10)
        self._panel.add_child(self._status)

        self._window.add_child(self._scene_widget)
        self._window.add_child(self._panel)
        self._window.add_child(self._axis_widget)
        self._window.set_on_layout(self._on_layout)

        self._btn_compute.set_on_clicked(self._on_compute)
        self._btn_fit.set_on_clicked(self._on_fit)
        self._slice_slider.set_on_value_changed(self._on_slice_changed)
        self._iso_slider.set_on_value_changed(self._on_iso_changed)

        # Sync axis inset as the user navigates the main viewport.
        self._scene_widget.set_on_mouse(self._on_mouse)

        self._profile_fields: Optional[np.ndarray] = None
        self._bracing_fields: Optional[np.ndarray] = None
        self._result_fields_2d: Optional[np.ndarray] = None
        
        self._iso_p_base = 0.0
        self._iso_b_base = 0.0

        self._bounds_min = None
        self._bounds_max = None
        self._X = None
        self._Y = None
        self._nx = None
        self._ny = None

        self._curves_geom: Optional[o3d.geometry.LineSet] = None
        self._profile_geom: Optional[o3d.geometry.LineSet] = None
        self._overlay_geom: Optional[o3d.geometry.Geometry] = None
        self._current_bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = None

    def _on_generate_toggled(self, checked):
        self._bracing_path.enabled = not checked

    def _on_boolean_param_changed(self, *args):
        if self._profile_fields is None or self._bracing_fields is None:
            return
        self._update_boolean_result()
        self._update_scene(fit_camera=False)

    def _update_boolean_result(self):
        mode = self._op_mode_combo.get_item(self._op_mode_combo.selected_index)
        off_p = self._profile_offset_slider.double_value
        off_b = self._bracing_offset_slider.double_value

        self._result_fields_2d = core.compute_sf_operation(
            self._profile_fields,
            self._bracing_fields,
            iso_level_A=self._iso_p_base + off_p,
            iso_level_B=self._iso_b_base + off_b,
            mode=mode,
        )

    def _on_layout(self, layout_context):
        r = self._window.content_rect
        self._panel.frame = gui.Rect(r.x, r.y, self._panel.preferred_width, r.height)
        self._scene_widget.frame = gui.Rect(
            r.x + self._panel.preferred_width,
            r.y,
            r.width - self._panel.preferred_width,
            r.height,
        )

        # Place a small axis widget in the bottom-left of the 3D viewport.
        axis_size = 140
        self._axis_widget.frame = gui.Rect(
            r.x + self._panel.preferred_width + 10,
            r.y + r.height - axis_size - 10,
            axis_size,
            axis_size,
        )

        # Ensure the axis camera matches current viewport orientation.
        self._sync_axis_camera()

    def _on_mouse(self, event):
        # Let SceneWidget handle orbit/pan/zoom, but keep axis inset updated.
        self._sync_axis_camera()
        return gui.Widget.EventCallbackResult.IGNORED

    def _sync_axis_camera(self):
        try:
            V = np.asarray(self._scene_widget.scene.camera.get_view_matrix(), dtype=float)
            if V.shape != (4, 4):
                return

            # For a typical lookAt view matrix: V[:3,:3] = R^T, so R = (V[:3,:3])^T.
            R = V[:3, :3].T
            forward = -R[:, 2]
            up = R[:, 1]

            center = np.array([0.0, 0.0, 0.0], dtype=float)
            eye = center - forward * 2.0

            self._axis_widget.scene.camera.look_at(center, eye, up)
        except Exception:
            # Don't let inset syncing break navigation.
            return

    def _on_compute(self):
        try:
            self._compute_from_paths(self._bracing_path.text_value, self._profile_path.text_value)
            self._status.text = "Computed. Use slice/iso controls."
            self._update_scene(fit_camera=True)
        except Exception as e:
            self._status.text = f"Compute failed: {e}"

    def _on_fit(self):
        if self._bounds_min is None or self._bounds_max is None:
            self._status.text = "No data loaded to fit camera."
            return
        self._fit_camera_to_current()

    def _on_slice_changed(self, _):
        if self._result_fields_2d is None:
            return
        self._update_scene(fit_camera=False)

    def _on_iso_changed(self, _):
        if self._result_fields_2d is None:
            return
        self._update_scene(fit_camera=False)

    def _compute_from_paths(self, bracing_json_path: str, profile_json_path: str):
        pp = Path(profile_json_path)
        if not pp.exists():
            raise FileNotFoundError(f"Profile JSON not found: {pp}")

        with open(pp, "r") as f:
            data_profile = json.load(f)

        self._iso_p_base, _, bmax_p, bmin_p = core.meta_data_info(data_profile)
        self._profile_fields, _ = core.stack_scalar_fields(data_profile)
        
        self._bounds_min = bmin_p
        self._bounds_max = bmax_p
        num_fields, nx, ny = core.infer_grid_from_scalar_fields(self._profile_fields)
        self._nx, self._ny = nx, ny

        if self._chk_generate_bracing.checked:
            k = int(self._num_centroids_slider.int_value)
            print(f"Generating bracing from centroids (k={k})...")
            self._bracing_fields = np.zeros_like(self._profile_fields)
            prev_centroids = None
            
            for i in range(num_fields):
                slice_2d = self._profile_fields[i].reshape((ny, nx))
                mask = core.get_profile_mask(slice_2d, iso_level=self._iso_p_base or 0.0)
                
                centroids = core.generate_centroids(mask, k=k, prev_centroids=prev_centroids)
                centroids = core.constrain_centroids_to_mask(centroids, mask)
                
                voronoi_sdf_flat = core.compute_voronoi_sdf((ny, nx), centroids)
                self._bracing_fields[i] = voronoi_sdf_flat.ravel()
                prev_centroids = centroids
                
                if i % 10 == 0:
                    print(f"Generated bracing for slice {i}/{num_fields}")
            
            self._iso_b_base = 0.0
        else:
            bp = Path(bracing_json_path)
            if not bp.exists():
                raise FileNotFoundError(f"Bracing JSON not found: {bp}")
            with open(bp, "r") as f:
                data_bracing = json.load(f)
            self._iso_b_base, _, _, _ = core.meta_data_info(data_bracing)
            self._bracing_fields, _ = core.stack_scalar_fields(data_bracing)

        if self._bracing_fields.shape != self._profile_fields.shape:
            raise ValueError("Bracing/Profile scalar field arrays must have the same shape")

        self._update_boolean_result()

        x = np.linspace(self._bounds_min[0], self._bounds_max[0], nx)
        y = np.linspace(self._bounds_min[1], self._bounds_max[1], ny)
        self._X, self._Y = np.meshgrid(x, y, indexing="xy")

        self._slice_slider.set_limits(0, max(0, num_fields - 1))
        self._slice_slider.int_value = min(self._slice_slider.int_value, max(0, num_fields - 1))

        # Provide a reasonable iso slider range based on data spread.
        sample = self._result_fields_2d.ravel()
        finite = sample[np.isfinite(sample)]
        if finite.size:
            lo = float(np.percentile(finite, 5))
            hi = float(np.percentile(finite, 95))
            if hi <= lo:
                hi = lo + 1.0
            self._iso_slider.set_limits(lo, hi)
            self._iso_slider.double_value = 0.0

    def set_data(self, result_fields_2d: np.ndarray, bounds_min, bounds_max, iso_level: float = 0.0, profile_fields: Optional[np.ndarray] = None, iso_p: float = 0.0):
        # Note: set_data is used when launching from main.py with pre-computed result.
        self._result_fields_2d = np.asarray(result_fields_2d)
        self._profile_fields = profile_fields
        self._bracing_fields = None
        
        self._iso_p_base = iso_p
        self._iso_b_base = 0.0
        
        self._bounds_min = bounds_min
        self._bounds_max = bounds_max

        try:
            num_fields, nx, ny = core.infer_grid_from_scalar_fields(self._result_fields_2d)
        except ValueError as e:
            self._status.text = f"Load error: {e}"
            return

        self._nx, self._ny = nx, ny

        x = np.linspace(self._bounds_min[0], self._bounds_max[0], nx)
        y = np.linspace(self._bounds_min[1], self._bounds_max[1], ny)
        self._X, self._Y = np.meshgrid(x, y, indexing="xy")

        self._slice_slider.set_limits(0, max(0, num_fields - 1))
        self._slice_slider.int_value = 0

        # Set iso slider range based on data spread, then set threshold value
        sample = self._result_fields_2d.ravel()
        finite = sample[np.isfinite(sample)]
        if finite.size:
            lo = float(np.percentile(finite, 5))
            hi = float(np.percentile(finite, 95))
            if hi <= lo:
                hi = lo + 1.0
            self._iso_slider.set_limits(lo, hi)
        
        self._iso_slider.double_value = float(iso_level)
        self._status.text = "Data loaded. Use slice/iso controls."
        self._update_scene(fit_camera=True)

    def _slice_z(self, slice_index: int) -> float:
        z0 = float(self._bounds_min[2])
        z1 = float(self._bounds_max[2])
        n = int(self._result_fields_2d.shape[0])
        if n <= 1:
            return z0
        return z0 + (z1 - z0) * (slice_index / (n - 1))

    def _update_scene(self, fit_camera: bool):
        if self._result_fields_2d is None or self._nx is None or self._ny is None:
            return

        idx = int(self._slice_slider.int_value)
        iso = float(self._iso_slider.double_value)
        z = self._slice_z(idx)

        # Remove old geometries
        for name in ["curves", "profile", "overlay"]:
            try:
                self._scene_widget.scene.remove_geometry(name)
            except Exception:
                pass

        self._curves_geom = None
        self._profile_geom = None
        self._overlay_geom = None
        self._current_bbox = None

        # 1. Extract and show Boolean Result Curves
        slice_2d = self._result_fields_2d[idx].reshape((self._ny, self._nx))
        curves = core.iso_curves_for_slice_2d(slice_2d, iso, self._X, self._Y)
        ls = _curves_to_lineset(curves, z)
        if ls is not None:
            self._curves_geom = ls
            mat = rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 2.0
            mat.base_color = [0.1, 0.7, 0.95, 1.0]  # Cyan-ish
            self._scene_widget.scene.add_geometry("curves", ls, mat)

        # 2. Extract and show Profile Curves (Reference)
        if self._profile_fields is not None:
            off_p = self._profile_offset_slider.double_value
            p_slice_2d = self._profile_fields[idx].reshape((self._ny, self._nx))
            p_curves = core.iso_curves_for_slice_2d(p_slice_2d, self._iso_p_base + off_p, self._X, self._Y)
            p_ls = _curves_to_lineset(p_curves, z)
            if p_ls is not None:
                self._profile_geom = p_ls
                p_mat = rendering.MaterialRecord()
                p_mat.shader = "unlitLine"
                p_mat.line_width = 1.0
                p_mat.base_color = [0.8, 0.8, 0.8, 0.6]  # Semi-transparent light gray
                self._scene_widget.scene.add_geometry("profile", p_ls, p_mat)

        self._status.text = f"Slice {idx} | iso {iso:.6g} | curves {len(curves)}"

        # Compute bbox from current geometries
        bbox = None
        for g in (self._curves_geom, self._profile_geom, self._overlay_geom):
            if g is None:
                continue
            try:
                gb = g.get_axis_aligned_bounding_box()
            except Exception:
                continue
            
            if bbox is None:
                bbox = gb
            else:
                # Manual union since + operator might not be supported in all Open3D versions
                min_b = np.minimum(bbox.min_bound, gb.min_bound)
                max_b = np.maximum(bbox.max_bound, gb.max_bound)
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)

        self._current_bbox = bbox

        if fit_camera:
            self._fit_camera_to_current()

    def _fit_camera_to_current(self):
        bbox = self._current_bbox
        if bbox is None:
            if self._bounds_min is None or self._bounds_max is None:
                return
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array([self._bounds_min[0], self._bounds_min[1], self._bounds_min[2]], dtype=float),
                max_bound=np.array([self._bounds_max[0], self._bounds_max[1], self._bounds_max[2]], dtype=float),
            )

        self._scene_widget.setup_camera(60.0, bbox, bbox.get_center())


def run_app(
    bracing_json_path: str = "./alice_result/251120/bracing/waveStackFields.json",
    profile_json_path: str = "./alice_result/251120/ext/waveStackFields.json",
):
    app = gui.Application.instance
    app.initialize()
    viewer = NarniaCurveViewer()
    viewer._bracing_path.text_value = bracing_json_path
    viewer._profile_path.text_value = profile_json_path
    app.run()


def run_app_from_data(result_fields_2d: np.ndarray, bounds_min, bounds_max, iso_level: float = 0.0, profile_fields: Optional[np.ndarray] = None, iso_p: float = 0.0):
    app = gui.Application.instance
    app.initialize()
    viewer = NarniaCurveViewer()
    # Defer scene updates until the GUI main thread is running.
    gui.Application.instance.post_to_main_thread(
        viewer._window,
        lambda: viewer.set_data(result_fields_2d, bounds_min, bounds_max, iso_level=iso_level, profile_fields=profile_fields, iso_p=iso_p),
    )
    app.run()


if __name__ == "__main__":
    # If an output NPZ exists, use it; otherwise allow the user to compute from JSON in the UI.
    output_path = Path("./output/processed_sdf_results.npz")
    profile_json_path = Path("./alice_result/251120/ext/waveStackFields.json")

    if output_path.exists() and profile_json_path.exists():
        data = np.load(output_path, allow_pickle=True)
        result_fields = data["result_fields"]
        iso_level = float(data["iso_level"])
        with open(profile_json_path, "r") as f:
            meta = json.load(f)
        iso_p, _, bounds_max, bounds_min = core.meta_data_info(meta)
        
        # Try to load profile fields for preview if they exist in the same folder as the JSON
        profile_fields, _ = core.stack_scalar_fields(meta)
        
        run_app_from_data(result_fields, bounds_min, bounds_max, iso_level=iso_level, profile_fields=profile_fields, iso_p=iso_p)
    else:
        run_app()
