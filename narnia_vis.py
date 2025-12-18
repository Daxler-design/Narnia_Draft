import json
from pathlib import Path
from typing import Optional
from datetime import datetime

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
    def _create_slider_row(self, label_text, min_val, max_val, init_val, on_change_callback, is_int=False):
        v = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        v.add_child(gui.Label(label_text))
        h = gui.Horiz(5)
        
        num_edit = gui.NumberEdit(gui.NumberEdit.INT if is_int else gui.NumberEdit.DOUBLE)
        slider = gui.Slider(gui.Slider.INT if is_int else gui.Slider.DOUBLE)
        slider.set_limits(min_val, max_val)

        if is_int:
            num_edit.int_value = int(init_val)
            slider.int_value = int(init_val)
        else:
            num_edit.double_value = float(init_val)
            slider.double_value = float(init_val)

        def on_slider(val):
            if is_int:
                num_edit.int_value = int(val)
            else:
                num_edit.double_value = float(val)
            if on_change_callback:
                on_change_callback(val)

        def on_edit(val):
            if is_int:
                slider.int_value = int(val)
            else:
                slider.double_value = float(val)
            if on_change_callback:
                on_change_callback(val)

        slider.set_on_value_changed(on_slider)
        num_edit.set_on_value_changed(on_edit)
        
        h.add_child(num_edit)
        h.add_child(slider)
        v.add_child(h)
        return v, slider, num_edit

    def _create_file_input_row(self, label_text, default_path, on_browse_callback):
        v = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        v.add_child(gui.Label(label_text))
        h = gui.Horiz(4)
        
        tedit = gui.TextEdit()
        tedit.text_value = default_path
        
        btn = gui.Button("...")
        btn.horizontal_padding_em = 0.5
        btn.set_on_clicked(on_browse_callback)
        
        h.add_child(tedit)
        h.add_child(btn)
        v.add_child(h)
        return v, tedit

    def __init__(self):
        self._base_dir = Path.cwd()
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

        self._tabs = gui.TabControl()

        # --- TAB 1: COMPUTE ---
        self._compute_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # Profile Path
        row_prof, self._profile_path = self._create_file_input_row(
            "Profile JSON path", 
            "./alice_result/251120/ext/waveStackFields.json",
            self._on_select_profile
        )

        # Bracing Path
        row_brac, self._bracing_path = self._create_file_input_row(
            "Bracing JSON path",
            "./alice_result/251120/bracing/waveStackFields.json",
            self._on_select_bracing
        )
        self._bracing_path.enabled = False

        self._chk_generate_bracing = gui.Checkbox("Generate Bracing from Centroids")
        self._chk_generate_bracing.checked = True
        self._chk_generate_bracing.set_on_checked(self._on_generate_toggled)

        # Num Centroids Slider (Limit 1-6)
        row_k, self._num_centroids_slider, self._num_centroids_edit = self._create_slider_row(
            "Num Centroids", 1, 6, 5, None, is_int=True
        )

        self._btn_compute = gui.Button("Load / Re-generate Bracing")

        self._op_mode_combo = gui.Combobox()
        self._op_mode_combo.add_item("difference")
        self._op_mode_combo.add_item("union")
        self._op_mode_combo.add_item("intersection")
        self._op_mode_combo.set_on_selection_changed(self._on_boolean_param_changed)

        # Profile Offset
        row_poff, self._profile_offset_slider, self._profile_offset_edit = self._create_slider_row(
            "Profile Offset", -2.0, 2.0, 0.0, self._on_boolean_param_changed
        )

        # Bracing Offset
        row_boff, self._bracing_offset_slider, self._bracing_offset_edit = self._create_slider_row(
            "Bracing Offset", -2.0, 2.0, 0.0, self._on_boolean_param_changed
        )

        # Result Iso (Compute)
        row_c_iso, self._c_iso_slider, self._c_iso_edit = self._create_slider_row(
            "Result Iso threshold", -2.0, 2.0, 0.0, self._on_iso_changed
        )

        self._compute_panel.add_child(row_prof)
        self._compute_panel.add_fixed(6)
        self._compute_panel.add_child(row_brac)
        self._compute_panel.add_fixed(10)
        self._compute_panel.add_child(self._chk_generate_bracing)
        self._compute_panel.add_child(row_k)
        self._compute_panel.add_fixed(10)
        self._compute_panel.add_child(self._btn_compute)
        self._compute_panel.add_fixed(16)
        self._compute_panel.add_child(gui.Label("--- Boolean Operation ---"))
        self._compute_panel.add_child(gui.Label("Mode"))
        self._compute_panel.add_child(self._op_mode_combo)
        self._compute_panel.add_child(row_poff)
        self._compute_panel.add_child(row_boff)
        self._compute_panel.add_fixed(10)
        self._compute_panel.add_child(row_c_iso)

        self._tabs.add_tab("Compute", self._compute_panel)

        # --- TAB 2: NPZ VIEWER ---
        self._viewer_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        
        # NPZ Path
        row_npz, self._npz_path = self._create_file_input_row(
            "NPZ File Path",
            "./output/processed_sdf_results.npz",
            self._on_select_npz
        )
        
        self._btn_load_npz = gui.Button("Load NPZ File")

        # View Channel Selector (NPZ)
        self._n_view_selector = gui.Combobox()
        self._n_view_selector.add_item("Result")
        self._n_view_selector.set_on_selection_changed(self._on_view_channel_changed)

        # Result Iso (NPZ)
        row_n_iso, self._n_iso_slider, self._n_iso_edit = self._create_slider_row(
            "Result Iso threshold", -1.0, 1.0, 0.0, self._on_iso_changed
        )

        self._viewer_panel.add_child(row_npz)
        self._viewer_panel.add_fixed(10)
        self._viewer_panel.add_child(self._btn_load_npz)
        self._viewer_panel.add_fixed(16)
        self._viewer_panel.add_child(gui.Label("View Channel"))
        self._viewer_panel.add_child(self._n_view_selector)
        self._viewer_panel.add_fixed(10)
        self._viewer_panel.add_child(row_n_iso)
        self._viewer_panel.add_fixed(16)
        self._viewer_panel.add_child(gui.Label("Note: Loading NPZ skips compute."))

        self._tabs.add_tab("NPZ Viewer", self._viewer_panel)

        # --- SHARED CONTROLS (Bottom) ---
        self._panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self._panel.preferred_width = 360
        self._panel.add_child(self._tabs)
        self._panel.add_fixed(16)
        self._panel.add_child(gui.Label("--- Visualization ---"))

        self._slice_slider = gui.Slider(gui.Slider.INT)
        self._slice_slider.set_limits(0, 0)
        self._slice_slider.int_value = 0

        self._btn_fit = gui.Button("Fit Camera")

        self._output_dir = gui.TextEdit()
        self._output_dir.text_value = "./output"
        self._btn_select_output = gui.Button("Select Folder")
        
        self._chk_export_profile = gui.Checkbox("Export Profile")
        self._chk_export_profile.checked = False
        self._chk_export_bracing = gui.Checkbox("Export Bracing")
        self._chk_export_bracing.checked = False
        
        self._btn_export = gui.Button("Export Results (.npz)")

        self._status = gui.Label("Load data to start.")

        self._panel.add_child(gui.Label("Slice"))
        self._panel.add_child(self._slice_slider)
        self._panel.add_fixed(10)
        self._panel.add_child(self._btn_fit)
        self._panel.add_fixed(16)
        self._panel.add_child(gui.Label("--- Export ---"))
        self._panel.add_child(gui.Label("Output Directory"))
        
        h = gui.Horiz(4)
        h.add_child(self._output_dir)
        h.add_child(self._btn_select_output)
        self._panel.add_child(h)
        
        h_opts = gui.Horiz(10)
        h_opts.add_child(self._chk_export_profile)
        h_opts.add_child(self._chk_export_bracing)
        self._panel.add_child(h_opts)
        self._panel.add_fixed(10)
        
        self._panel.add_child(self._btn_export)
        self._panel.add_fixed(10)
        self._panel.add_child(self._status)

        self._window.add_child(self._scene_widget)
        self._window.add_child(self._panel)
        self._window.add_child(self._axis_widget)
        self._window.set_on_layout(self._on_layout)

        self._btn_compute.set_on_clicked(self._on_compute)
        self._btn_fit.set_on_clicked(self._on_fit)
        self._btn_select_output.set_on_clicked(self._on_select_output)
        self._btn_export.set_on_clicked(self._on_export)
        # self._btn_select_npz is now handled in _create_file_input_row callback
        self._btn_load_npz.set_on_clicked(self._on_load_npz)
        self._slice_slider.set_on_value_changed(self._on_slice_changed)
        self._tabs.set_on_selected_tab_changed(self._on_tab_changed)

        # Sync axis inset as the user navigates the main viewport.
        self._scene_widget.set_on_mouse(self._on_mouse)

        # --- State: Compute Tab ---
        self._c_profile: Optional[np.ndarray] = None
        self._c_bracing: Optional[np.ndarray] = None
        self._c_result: Optional[np.ndarray] = None
        self._c_iso_p_base = 0.0
        self._c_iso_b_base = 0.0
        self._c_bounds_min = None
        self._c_bounds_max = None
        self._c_grid = (None, None, None, None) # nx, ny, X, Y

        # --- State: NPZ Tab ---
        self._n_result: Optional[np.ndarray] = None
        self._n_profile: Optional[np.ndarray] = None
        self._n_bracing: Optional[np.ndarray] = None
        self._n_bounds_min = None
        self._n_bounds_max = None
        self._n_grid = (None, None, None, None) # nx, ny, X, Y
        self._n_iso_base = 0.0

        self._curves_geom: Optional[o3d.geometry.LineSet] = None
        self._profile_geom: Optional[o3d.geometry.LineSet] = None
        self._bracing_geom: Optional[o3d.geometry.LineSet] = None
        self._overlay_geom: Optional[o3d.geometry.Geometry] = None
        self._current_bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = None

    def _on_view_channel_changed(self, name, index):
        self._update_scene(fit_camera=False)
        self._update_slider_ranges()

    def _on_tab_changed(self, index):
        # Refresh scene when switching tabs to show the correct data
        self._update_scene(fit_camera=False)
        self._update_slider_ranges()

    def _update_slider_ranges(self):
        idx_tab = self._tabs.selected_tab_index
        res = self._c_result if idx_tab == 0 else self._n_result
        slider = self._c_iso_slider if idx_tab == 0 else self._n_iso_slider
        edit = self._c_iso_edit if idx_tab == 0 else self._n_iso_edit
        
        if res is None:
            return

        num_fields = res.shape[0]
        self._slice_slider.set_limits(0, max(0, num_fields - 1))
        
        # Keep slider limits fixed to -1.0 to 1.0 as requested, 
        # instead of dynamically adjusting to data range.
        # sample = res.ravel()
        # finite = sample[np.isfinite(sample)]
        # if finite.size:
        #     lo = float(np.percentile(finite, 5))
        #     hi = float(np.percentile(finite, 95))
        #     if hi <= lo:
        #         hi = lo + 1.0
        #     slider.set_limits(lo, hi)
        #     # Sync edit with slider's current value (which might have been clamped)
        #     edit.double_value = slider.double_value

    def _on_generate_toggled(self, checked):
        self._bracing_path.enabled = not checked

    def _on_boolean_param_changed(self, *args):
        if self._c_profile is None or self._c_bracing is None:
            return
        # If user is tweaking compute params, ensure we are on the compute tab
        if self._tabs.selected_tab_index != 0:
            self._tabs.selected_tab_index = 0
            
        self._update_boolean_result()
        self._update_scene(fit_camera=False)

    def _update_boolean_result(self):
        mode = self._op_mode_combo.get_item(self._op_mode_combo.selected_index)
        off_p = self._profile_offset_slider.double_value
        off_b = self._bracing_offset_slider.double_value

        self._c_result = core.compute_sf_operation(
            self._c_profile,
            self._c_bracing,
            iso_level_A=self._c_iso_p_base + off_p,
            iso_level_B=self._c_iso_b_base + off_b,
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
            self._tabs.selected_tab_index = 0
            self._compute_from_paths(self._bracing_path.text_value, self._profile_path.text_value)
            self._status.text = "Computed. Use slice/iso controls."
            self._update_scene(fit_camera=True)
        except Exception as e:
            self._status.text = f"Compute failed: {e}"

    def _on_fit(self):
        idx_tab = self._tabs.selected_tab_index
        bmin = self._c_bounds_min if idx_tab == 0 else self._n_bounds_min
        bmax = self._c_bounds_max if idx_tab == 0 else self._n_bounds_max
        
        if bmin is None or bmax is None:
            self._status.text = "No data loaded to fit camera."
            return
        self._fit_camera_to_current()

    def _on_select_output(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Select Output Folder", self._window.theme)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_output_dir_done)
        self._window.show_dialog(dlg)

    def _on_output_dir_done(self, path):
        self._output_dir.text_value = path
        self._window.close_dialog()

    def _on_select_profile(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select Profile JSON", self._window.theme)
        dlg.add_filter(".json", "JSON files (.json)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_profile_done)
        self._window.show_dialog(dlg)

    def _on_profile_done(self, path):
        self._profile_path.text_value = path
        self._window.close_dialog()

    def _on_select_bracing(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select Bracing JSON", self._window.theme)
        dlg.add_filter(".json", "JSON files (.json)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_bracing_done)
        self._window.show_dialog(dlg)

    def _on_bracing_done(self, path):
        self._bracing_path.text_value = path
        self._window.close_dialog()

    def _on_select_npz(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select NPZ File", self._window.theme)
        dlg.add_filter(".npz", "NPZ files (.npz)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_npz_file_done)
        self._window.show_dialog(dlg)

    def _on_npz_file_done(self, path):
        self._npz_path.text_value = path
        self._window.close_dialog()

    def _on_file_dialog_cancel(self):
        self._window.close_dialog()

    def _on_file_dialog_done(self, path):
        # Deprecated in favor of specific callbacks
        self._window.close_dialog()

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        # Resolve relative to the directory where the app started
        return (self._base_dir / p).resolve()

    def _on_load_npz(self):
        self._status.text = "Loading NPZ file..."
        self._window.set_needs_layout()
        
        path = self._resolve_path(self._npz_path.text_value)
        if not path.exists():
            self._status.text = f"File not found: {path}"
            return
        try:
            data = np.load(path, allow_pickle=True)
            res_fields = data["result_fields"]
            iso = float(data.get("iso_level", 0.0))
            
            # Try to load optional fields
            prof_fields = data.get("profile_fields", None)
            # Handle 0-d array case if saved as None or empty
            if prof_fields is not None and prof_fields.ndim == 0:
                prof_fields = None
                
            brac_fields = data.get("bracing_fields", None)
            if brac_fields is not None and brac_fields.ndim == 0:
                brac_fields = None

            # Try to get bounds from NPZ, fallback to current if available
            current_bmin = self._n_bounds_min if self._n_bounds_min is not None else self._c_bounds_min
            current_bmax = self._n_bounds_max if self._n_bounds_max is not None else self._c_bounds_max
            
            bmin = data.get("bounds_min", current_bmin)
            bmax = data.get("bounds_max", current_bmax)
            
            if bmin is None or bmax is None:
                bmin = np.array([0.0, 0.0, 0.0])
                bmax = np.array([100.0, 100.0, 100.0])

            self._n_result = res_fields
            self._n_profile = prof_fields
            self._n_bracing = brac_fields
            
            self._n_bounds_min = bmin
            self._n_bounds_max = bmax
            self._n_iso_base = iso
            
            num_fields, nx, ny = core.infer_grid_from_scalar_fields(res_fields)
            x = np.linspace(bmin[0], bmax[0], nx)
            y = np.linspace(bmin[1], bmax[1], ny)
            X, Y = np.meshgrid(x, y, indexing="xy")
            self._n_grid = (nx, ny, X, Y)

            # Update View Selector
            self._n_view_selector.clear_items()
            self._n_view_selector.add_item("Result")
            if self._n_profile is not None:
                self._n_view_selector.add_item("Profile")
            if self._n_bracing is not None:
                self._n_view_selector.add_item("Bracing")
            self._n_view_selector.selected_index = 0

            self._tabs.selected_tab_index = 1
            self._update_slider_ranges()
            self._n_iso_slider.double_value = iso
            self._n_iso_edit.double_value = iso
            self._update_scene(fit_camera=True)
            self._status.text = f"Loaded NPZ: {path.name}"
        except Exception as e:
            self._status.text = f"Load NPZ failed: {e}"

    def _on_export(self):
        idx_tab = self._tabs.selected_tab_index
        res = self._c_result if idx_tab == 0 else self._n_result
        bmin = self._c_bounds_min if idx_tab == 0 else self._n_bounds_min
        bmax = self._c_bounds_max if idx_tab == 0 else self._n_bounds_max
        slider = self._c_iso_slider if idx_tab == 0 else self._n_iso_slider

        if res is None:
            self._status.text = "No result to export. Compute or load first."
            return

        try:
            out_dir = self._resolve_path(self._output_dir.text_value)
            out_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_sdf_results_{timestamp}.npz"
            output_path = out_dir / filename

            iso_level = float(slider.double_value)
            
            save_dict = {
                "result_fields": res,
                "iso_level": iso_level,
                "bounds_min": bmin,
                "bounds_max": bmax,
            }
            
            # Add optional fields if requested
            if self._chk_export_profile.checked:
                prof = self._c_profile if idx_tab == 0 else self._n_profile
                if prof is not None:
                    save_dict["profile_fields"] = prof
            
            if self._chk_export_bracing.checked:
                brac = self._c_bracing if idx_tab == 0 else self._n_bracing
                if brac is not None:
                    save_dict["bracing_fields"] = brac

            np.savez(output_path, **save_dict)
            self._status.text = f"Exported: {filename}"
            print(f"Successfully exported results to {output_path}")
        except Exception as e:
            self._status.text = f"Export failed: {e}"

    def _on_slice_changed(self, _):
        idx_tab = self._tabs.selected_tab_index
        res = self._c_result if idx_tab == 0 else self._n_result
        if res is None:
            return
        self._update_scene(fit_camera=False)

    def _on_iso_changed(self, _):
        idx_tab = self._tabs.selected_tab_index
        res = self._c_result if idx_tab == 0 else self._n_result
        if res is None:
            return
        self._update_scene(fit_camera=False)

    def _compute_from_paths(self, bracing_json_path: str, profile_json_path: str):
        self._status.text = "Loading profile data..."
        self._window.set_needs_layout()
        
        pp = self._resolve_path(profile_json_path)
        if not pp.exists():
            raise FileNotFoundError(f"Profile JSON not found: {pp}")

        with open(pp, "r") as f:
            data_profile = json.load(f)

        self._c_iso_p_base, _, bmax_p, bmin_p = core.meta_data_info(data_profile)
        self._c_profile, _ = core.stack_scalar_fields(data_profile)
        
        self._c_bounds_min = bmin_p
        self._c_bounds_max = bmax_p
        num_fields, nx, ny = core.infer_grid_from_scalar_fields(self._c_profile)

        if self._chk_generate_bracing.checked:
            self._status.text = "Generating bracing..."
            self._window.set_needs_layout()
            
            k = int(self._num_centroids_slider.int_value)
            print(f"Generating bracing from centroids (k={k})...")
            self._c_bracing = np.zeros_like(self._c_profile)
            prev_centroids = None
            
            for i in range(num_fields):
                slice_2d = self._c_profile[i].reshape((ny, nx))
                mask = core.get_profile_mask(slice_2d, iso_level=self._c_iso_p_base or 0.0)
                
                centroids = core.generate_centroids(mask, k=k, prev_centroids=prev_centroids)
                centroids = core.constrain_centroids_to_mask(centroids, mask)
                
                voronoi_sdf_flat = core.compute_voronoi_sdf((ny, nx), centroids)
                self._c_bracing[i] = voronoi_sdf_flat.ravel()
                prev_centroids = centroids
                
                if i % 10 == 0:
                    print(f"Generated bracing for slice {i}/{num_fields}")
            
            self._c_iso_b_base = 0.0
        else:
            self._status.text = "Loading bracing data..."
            self._window.set_needs_layout()
            
            bp = self._resolve_path(bracing_json_path)
            if not bp.exists():
                raise FileNotFoundError(f"Bracing JSON not found: {bp}")
            with open(bp, "r") as f:
                data_bracing = json.load(f)
            self._c_iso_b_base, _, _, _ = core.meta_data_info(data_bracing)
            self._c_bracing, _ = core.stack_scalar_fields(data_bracing)

        if self._c_bracing.shape != self._c_profile.shape:
            raise ValueError("Bracing/Profile scalar field arrays must have the same shape")

        self._status.text = "Computing boolean operation..."
        self._window.set_needs_layout()
        
        self._update_boolean_result()

        x = np.linspace(self._c_bounds_min[0], self._c_bounds_max[0], nx)
        y = np.linspace(self._c_bounds_min[1], self._c_bounds_max[1], ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        self._c_grid = (nx, ny, X, Y)

        self._update_slider_ranges()
        self._c_iso_slider.double_value = 0.0
        self._c_iso_edit.double_value = 0.0

    def set_data(self, result_fields_2d: np.ndarray, bounds_min, bounds_max, iso_level: float = 0.0, profile_fields: Optional[np.ndarray] = None, iso_p: float = 0.0):
        # Note: set_data is used when launching from main.py with pre-computed result.
        self._c_result = np.asarray(result_fields_2d)
        self._c_profile = profile_fields
        self._c_bracing = None
        
        self._c_iso_p_base = iso_p
        self._c_iso_b_base = 0.0
        
        self._c_bounds_min = bounds_min
        self._c_bounds_max = bounds_max

        try:
            num_fields, nx, ny = core.infer_grid_from_scalar_fields(self._c_result)
        except ValueError as e:
            self._status.text = f"Load error: {e}"
            return

        x = np.linspace(self._c_bounds_min[0], self._c_bounds_max[0], nx)
        y = np.linspace(self._c_bounds_min[1], self._c_bounds_max[1], ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        self._c_grid = (nx, ny, X, Y)

        self._tabs.selected_tab_index = 0
        self._update_slider_ranges()
        self._c_iso_slider.double_value = float(iso_level)
        self._status.text = "Data loaded. Use slice/iso controls."
        self._update_scene(fit_camera=True)

    def _slice_z(self, slice_index: int, num_fields: int, bounds_min, bounds_max) -> float:
        z0 = float(bounds_min[2])
        z1 = float(bounds_max[2])
        if num_fields <= 1:
            return z0
        return z0 + (z1 - z0) * (slice_index / (num_fields - 1))

    def _update_scene(self, fit_camera: bool):
        idx_tab = self._tabs.selected_tab_index
        
        # Initialize variables
        res = None
        prof = None
        brac = None
        bmin, bmax = None, None
        nx, ny, X, Y = None, None, None, None
        iso = 0.0
        iso_p_base = 0.0
        iso_b_base = 0.0
        curve_color = [1.0, 1.0, 1.0, 1.0]

        if idx_tab == 0: # Compute
            res = self._c_result
            prof = self._c_profile
            # brac remains None for Compute tab to preserve existing behavior
            
            bmin, bmax = self._c_bounds_min, self._c_bounds_max
            nx, ny, X, Y = self._c_grid
            iso_p_base = self._c_iso_p_base
            iso_b_base = self._c_iso_b_base
            iso = float(self._c_iso_slider.double_value)
            curve_color = [0.1, 0.7, 0.95, 1.0]  # Cyan-ish
        else: # NPZ
            # Determine which field to show based on selector
            sel_idx = self._n_view_selector.selected_index
            sel_text = self._n_view_selector.get_item(sel_idx) if sel_idx >= 0 else "Result"
            
            bmin, bmax = self._n_bounds_min, self._n_bounds_max
            nx, ny, X, Y = self._n_grid
            iso = float(self._n_iso_slider.double_value)
            
            if sel_text == "Profile":
                res = self._n_profile
                curve_color = [0.8, 0.8, 0.8, 1.0] # Gray
            elif sel_text == "Bracing":
                res = self._n_bracing
                curve_color = [0.2, 0.8, 0.2, 1.0] # Green
            else: # Result
                res = self._n_result
                prof = self._n_profile
                brac = self._n_bracing
                curve_color = [1.0, 0.5, 0.0, 1.0]  # Orange-ish

        if res is None or nx is None or ny is None:
            return

        idx = int(self._slice_slider.int_value)
        z = self._slice_z(idx, res.shape[0], bmin, bmax)

        # Remove old geometries
        for name in ["curves", "profile", "bracing", "overlay"]:
            try:
                self._scene_widget.scene.remove_geometry(name)
            except Exception:
                pass

        self._curves_geom = None
        self._profile_geom = None
        self._bracing_geom = None
        self._overlay_geom = None
        self._current_bbox = None

        # 1. Extract and show Main Curves
        slice_2d = res[idx].reshape((ny, nx))
        curves = core.iso_curves_for_slice_2d(slice_2d, iso, X, Y)
        ls = _curves_to_lineset(curves, z)
        if ls is not None:
            self._curves_geom = ls
            mat = rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 2.0
            mat.base_color = curve_color
            self._scene_widget.scene.add_geometry("curves", ls, mat)

        # 2. Extract and show Profile Curves (Reference)
        if prof is not None:
            off_p = self._profile_offset_slider.double_value if idx_tab == 0 else 0.0
            p_slice_2d = prof[idx].reshape((ny, nx))
            p_curves = core.iso_curves_for_slice_2d(p_slice_2d, iso_p_base + off_p, X, Y)
            p_ls = _curves_to_lineset(p_curves, z)
            if p_ls is not None:
                self._profile_geom = p_ls
                p_mat = rendering.MaterialRecord()
                p_mat.shader = "unlitLine"
                p_mat.line_width = 1.0
                p_mat.base_color = [0.8, 0.8, 0.8, 0.6]  # Semi-transparent light gray
                self._scene_widget.scene.add_geometry("profile", p_ls, p_mat)

        # 3. Extract and show Bracing Curves (Reference)
        if brac is not None:
            off_b = self._bracing_offset_slider.double_value if idx_tab == 0 else 0.0
            b_slice_2d = brac[idx].reshape((ny, nx))
            b_curves = core.iso_curves_for_slice_2d(b_slice_2d, iso_b_base + off_b, X, Y)
            b_ls = _curves_to_lineset(b_curves, z)
            if b_ls is not None:
                self._bracing_geom = b_ls
                b_mat = rendering.MaterialRecord()
                b_mat.shader = "unlitLine"
                b_mat.line_width = 1.0
                b_mat.base_color = [0.2, 0.8, 0.2, 0.6]  # Semi-transparent Green
                self._scene_widget.scene.add_geometry("bracing", b_ls, b_mat)

        self._status.text = f"Slice {idx} | iso {iso:.6g} | curves {len(curves)}"

        # Compute bbox from current geometries
        bbox = None
        for g in (self._curves_geom, self._profile_geom, self._bracing_geom, self._overlay_geom):
            if g is None:
                continue
            try:
                gb = g.get_axis_aligned_bounding_box()
            except Exception:
                continue
            
            if bbox is None:
                bbox = gb
            else:
                min_b = np.minimum(bbox.min_bound, gb.min_bound)
                max_b = np.maximum(bbox.max_bound, gb.max_bound)
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)

        self._current_bbox = bbox

        if fit_camera:
            self._fit_camera_to_current()

    def _fit_camera_to_current(self):
        bbox = self._current_bbox
        if bbox is None:
            idx_tab = self._tabs.selected_tab_index
            bmin = self._c_bounds_min if idx_tab == 0 else self._n_bounds_min
            bmax = self._c_bounds_max if idx_tab == 0 else self._n_bounds_max
            
            if bmin is None or bmax is None:
                return
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array([bmin[0], bmin[1], bmin[2]], dtype=float),
                max_bound=np.array([bmax[0], bmax[1], bmax[2]], dtype=float),
            )

        self._scene_widget.setup_camera(60.0, bbox, bbox.get_center())


def run_app(
    bracing_json_path: str = "./alice_result/251120/bracing/waveStackFields.json",
    profile_json_path: str = "./alice_result/251120/ext/waveStackFields.json",
    output_dir: str = "./output",
):
    app = gui.Application.instance
    app.initialize()
    viewer = NarniaCurveViewer()
    viewer._bracing_path.text_value = bracing_json_path
    viewer._profile_path.text_value = profile_json_path
    viewer._output_dir.text_value = output_dir
    app.run()


def run_app_from_data(
    result_fields_2d: np.ndarray,
    bounds_min,
    bounds_max,
    iso_level: float = 0.0,
    profile_fields: Optional[np.ndarray] = None,
    iso_p: float = 0.0,
    output_dir: str = "./output",
):
    app = gui.Application.instance
    app.initialize()
    viewer = NarniaCurveViewer()
    viewer._output_dir.text_value = output_dir
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
