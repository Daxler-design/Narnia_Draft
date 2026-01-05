import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import core
import vis_utils as vut
import vis_widgets as vwg


class NarniaCurveViewer:
    def __init__(self, monitor_index: int = 0):
        self._base_dir = Path.cwd()
        
        x, y = 0, 0
        try:
            monitors = vut.get_monitors_info()
            if 0 <= monitor_index < len(monitors):
                rect = monitors[monitor_index]["rect"]
                x, y = rect[0], rect[1]
                # Add a small offset to ensure it's visible
                x += 50
                y += 50
        except Exception as e:
            print(f"Could not get monitor info: {e}")

        self._window = gui.Application.instance.create_window("Narnia Viewer", 1400, 900, x, y)

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
        row_prof, self._profile_path = vwg.create_file_input_row(
            "Profile JSON path", 
            "./alice_result/251120/ext/waveStackFields.json",
            self._on_select_profile
        )

        # Bracing Path
        row_brac, self._bracing_path = vwg.create_file_input_row(
            "Bracing JSON path",
            "./alice_result/251120/bracing/waveStackFields.json",
            self._on_select_bracing
        )
        self._bracing_path.enabled = False

        self._chk_generate_bracing = gui.Checkbox("Generate Bracing from Centroids")
        self._chk_generate_bracing.checked = True
        self._chk_generate_bracing.set_on_checked(self._on_generate_toggled)

        # Bracing Method
        self._bracing_method_combo = gui.Combobox()
        self._bracing_method_combo.add_item("voronoi")
        self._bracing_method_combo.add_item("cell_wall")
        self._bracing_method_combo.selected_index = 1  # Default to cell_wall

        # --- Cell Wall Settings ---
        self._cell_wall_settings = gui.CollapsableVert("Cell Wall Settings", 0, gui.Margins(10, 0, 0, 0))
        
        row_tau, self._cell_tau_slider, self._cell_tau_edit = vwg.create_slider_row(
            "Tau (Softmax Temp)", 1.0, 30.0, 12.0, None
        )
        
        self._cell_wall_method_combo = gui.Combobox()
        self._cell_wall_method_combo.add_item("entropy")
        self._cell_wall_method_combo.add_item("top2gap")
        
        row_smooth_xy, self._cell_smooth_xy_slider, self._cell_smooth_xy_edit = vwg.create_slider_row(
            "Smooth XY Sigma", 0.0, 5.0, 1.0, None
        )
        
        row_wall_thresh, self._cell_wall_threshold_slider, self._cell_wall_threshold_edit = vwg.create_slider_row(
            "Wall Threshold", 0.0, 1.0, 0.6, None
        )
        
        row_wall_thick, self._cell_wall_thickness_slider, self._cell_wall_thickness_edit = vwg.create_slider_row(
            "Wall Thickness (px)", 0.5, 10.0, 2.0, None
        )
        
        row_smooth_z, self._cell_smooth_z_slider, self._cell_smooth_z_edit = vwg.create_slider_row(
            "Smooth Z Sigma", 0.0, 5.0, 0.75, None
        )
        
        row_ramp_slices, self._cell_ramp_slices_slider, self._cell_ramp_slices_edit = vwg.create_slider_row(
            "Ramp Slices", 1, 20, 5, None, is_int=True
        )

        self._cell_wall_settings.add_child(row_tau)
        self._cell_wall_settings.add_child(gui.Label("Wall Method"))
        self._cell_wall_settings.add_child(self._cell_wall_method_combo)
        self._cell_wall_settings.add_child(row_smooth_xy)
        self._cell_wall_settings.add_child(row_wall_thresh)
        self._cell_wall_settings.add_child(row_wall_thick)
        self._cell_wall_settings.add_child(row_smooth_z)
        self._cell_wall_settings.add_child(row_ramp_slices)

        # --- OT Transport Settings ---
        self._ot_settings = gui.CollapsableVert("OT Transport Settings", 0, gui.Margins(10, 0, 0, 0))
        
        self._use_ot_transport_chk = gui.Checkbox("Use OT Transport")
        self._use_ot_transport_chk.checked = True
        
        row_ot_samples, self._ot_num_samples_slider, self._ot_num_samples_edit = vwg.create_slider_row(
            "Num Samples", 100, 2000, 600, None, is_int=True
        )
        
        row_ot_band, self._ot_band_px_slider, self._ot_band_px_edit = vwg.create_slider_row(
            "Band Px", 1.0, 20.0, 6.0, None
        )
        
        row_ot_eps, self._ot_epsilon_slider, self._ot_epsilon_edit = vwg.create_slider_row(
            "Epsilon", 1.0, 50.0, 12.0, None
        )
        
        row_ot_iter, self._ot_max_iter_slider, self._ot_max_iter_edit = vwg.create_slider_row(
            "Max Iter", 100, 2000, 400, None, is_int=True
        )
        
        row_ot_tol, self._ot_tol_slider, self._ot_tol_edit = vwg.create_slider_row(
            "Tolerance", 1e-4, 1e-1, 1e-3, None
        )
        
        row_ot_rbf, self._ot_rbf_smooth_slider, self._ot_rbf_smooth_edit = vwg.create_slider_row(
            "RBF Smooth", 0.1, 20.0, 5.0, None
        )
        
        row_ot_disp, self._ot_max_disp_px_slider, self._ot_max_disp_px_edit = vwg.create_slider_row(
            "Max Disp Px", 1.0, 100.0, 20.0, None
        )

        self._ot_settings.add_child(self._use_ot_transport_chk)
        self._ot_settings.add_child(row_ot_samples)
        self._ot_settings.add_child(row_ot_band)
        self._ot_settings.add_child(row_ot_eps)
        self._ot_settings.add_child(row_ot_iter)
        self._ot_settings.add_child(row_ot_tol)
        self._ot_settings.add_child(row_ot_rbf)
        self._ot_settings.add_child(row_ot_disp)

        # Start K Slider
        row_k_start, self._k_start_slider, self._k_start_edit = vwg.create_slider_row(
            "Start K", 1, 12, 3, None, is_int=True
        )
        # End K Slider
        row_k_end, self._k_end_slider, self._k_end_edit = vwg.create_slider_row(
            "End K", 1, 12, 8, None, is_int=True
        )

        # Ramp Slider
        row_ramp, self._ramp_slider, self._ramp_edit = vwg.create_slider_row(
            "Ramp", 1, 20, 5, None, is_int=True
        )

        # Smooth Sigma Slider
        row_smooth, self._smooth_slider, self._smooth_edit = vwg.create_slider_row(
            "Smooth Sigma", 0.1, 5.0, 2.0, None
        )

        self._btn_compute = gui.Button("Load / Re-generate Bracing")

        self._op_mode_combo = gui.Combobox()
        self._op_mode_combo.add_item("difference")
        self._op_mode_combo.add_item("union")
        self._op_mode_combo.add_item("intersection")
        self._op_mode_combo.set_on_selection_changed(self._on_boolean_param_changed)

        # Profile Offset
        row_poff, self._profile_offset_slider, self._profile_offset_edit = vwg.create_slider_row(
            "Profile Offset", -2.0, 2.0, 0.0, self._on_boolean_param_changed
        )

        # Bracing Offset
        row_boff, self._bracing_offset_slider, self._bracing_offset_edit = vwg.create_slider_row(
            "Bracing Offset", -2.0, 2.0, 0.0, self._on_boolean_param_changed
        )

        # Result Iso (Compute)
        row_c_iso, self._c_iso_slider, self._c_iso_edit = vwg.create_slider_row(
            "Result Iso threshold", -2.0, 2.0, 0.0, self._on_iso_changed
        )

        self._compute_panel.add_child(row_prof)
        self._compute_panel.add_fixed(6)
        self._compute_panel.add_child(row_brac)
        self._compute_panel.add_fixed(10)
        self._compute_panel.add_child(self._chk_generate_bracing)
        self._compute_panel.add_child(gui.Label("Bracing Method"))
        self._compute_panel.add_child(self._bracing_method_combo)
        self._compute_panel.add_fixed(5)
        self._compute_panel.add_child(row_k_start)
        self._compute_panel.add_child(row_k_end)
        self._compute_panel.add_child(row_ramp)
        self._compute_panel.add_child(row_smooth)
        self._compute_panel.add_fixed(5)
        self._compute_panel.add_child(self._cell_wall_settings)
        self._compute_panel.add_child(self._ot_settings)
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
        row_npz, self._npz_path = vwg.create_file_input_row(
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
        row_n_iso, self._n_iso_slider, self._n_iso_edit = vwg.create_slider_row(
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
        self._n_W: Optional[np.ndarray] = None
        self._n_B: Optional[np.ndarray] = None
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

            W_fields = data.get("W", None)
            if W_fields is not None and W_fields.ndim == 0:
                W_fields = None

            B_fields = data.get("B", None)
            if B_fields is not None and B_fields.ndim == 0:
                B_fields = None

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
            self._n_W = W_fields
            self._n_B = B_fields
            
            self._n_bounds_min = bmin
            self._n_bounds_max = bmax
            self._n_iso_base = iso

            nx_meta = data.get("nx", None)
            ny_meta = data.get("ny", None)
            num_fields, nx, ny = core.infer_grid_from_scalar_fields(res_fields, nx=nx_meta, ny=ny_meta)
            _, _, X, Y = core.xy_grid_from_bounds(bmin, bmax, nx, ny)
            self._n_grid = (nx, ny, X, Y)

            # Update View Selector
            self._n_view_selector.clear_items()
            self._n_view_selector.add_item("Result")
            if self._n_profile is not None:
                self._n_view_selector.add_item("Profile")
            if self._n_bracing is not None:
                self._n_view_selector.add_item("Bracing")
            if self._n_W is not None:
                self._n_view_selector.add_item("WallStrength")
            if self._n_B is not None:
                self._n_view_selector.add_item("WallField")
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
        num_fields, nx, ny = core.infer_grid_from_scalar_fields(self._c_profile, metadata=data_profile)

        if self._chk_generate_bracing.checked:
            self._status.text = "Generating bracing..."
            self._window.set_needs_layout()
            
            k_start = int(self._k_start_slider.int_value)
            k_end = int(self._k_end_slider.int_value)
            ramp_val = int(self._ramp_slider.int_value)
            smooth_val = float(self._smooth_slider.double_value)
            
            method_idx = self._bracing_method_combo.selected_index
            method_name = self._bracing_method_combo.get_item(method_idx)
            
            if method_name == "cell_wall":
                print(f"Generating bracing (Cell Wall, Centroids: {k_start} -> {k_end})...")
                
                # Gather Cell Wall Params
                c_tau = float(self._cell_tau_slider.double_value)
                c_method = self._cell_wall_method_combo.get_item(self._cell_wall_method_combo.selected_index)
                c_smooth_xy = float(self._cell_smooth_xy_slider.double_value)
                c_thresh = float(self._cell_wall_threshold_slider.double_value)
                c_thick = float(self._cell_wall_thickness_slider.double_value)
                c_smooth_z = float(self._cell_smooth_z_slider.double_value)
                c_ramp = int(self._cell_ramp_slices_slider.int_value)
                
                # Gather OT Params
                use_ot = self._use_ot_transport_chk.checked
                ot_samples = int(self._ot_num_samples_slider.int_value)
                ot_band = float(self._ot_band_px_slider.double_value)
                ot_eps = float(self._ot_epsilon_slider.double_value)
                ot_iter = int(self._ot_max_iter_slider.int_value)
                ot_tol = float(self._ot_tol_slider.double_value)
                ot_rbf = float(self._ot_rbf_smooth_slider.double_value)
                ot_disp = float(self._ot_max_disp_px_slider.double_value)
                
                num_slices = self._c_profile.shape[0]
                masks = np.zeros((num_slices, ny, nx), dtype=bool)
                for i in range(num_slices):
                    masks[i] = core.get_profile_mask(self._c_profile[i].reshape((ny, nx)), iso_level=self._c_iso_p_base)

                k_schedule = np.linspace(k_start, k_end, num_slices).astype(int)
                k0 = int(k_schedule[0]) if num_slices > 0 else 0

                init_seeds = core.generate_centroids(masks[0], k=k0, seed=42)
                seeds, ramp_weights, report = core.track_seeds_with_splits(
                    masks,
                    k_schedule,
                    init_seeds,
                    ramp_slices=c_ramp,
                    use_ot_transport=use_ot,
                    ot_num_samples=ot_samples,
                    ot_band_px=ot_band,
                    ot_epsilon=ot_eps,
                    ot_max_iter=ot_iter,
                    ot_tol=ot_tol,
                    ot_rbf_smooth=ot_rbf,
                    ot_max_disp_px=ot_disp,
                )
                W, B = core.compute_volume_cell_walls(
                    masks,
                    seeds,
                    ramp_weights,
                    tau=c_tau,
                    wall_method=c_method,
                    smooth_sigma_xy=c_smooth_xy,
                    threshold=c_thresh,
                    thickness_px=c_thick,
                    sigma_z=c_smooth_z,
                )
                self._c_bracing = B.reshape((num_slices, -1))
                
            else:
                print(f"Generating bracing (Voronoi, Centroids: {k_start} -> {k_end}, Ramp: {ramp_val}, Smooth: {smooth_val})...")
                
                # Use the existing interpolated pipeline
                self._c_bracing, _, _ = core.generate_interpolated_bracing_fields(
                    self._c_profile, 
                    k_min=k_start, 
                    k_max=k_end,
                    iso_level=self._c_iso_p_base or 0.0,
                    ramp=ramp_val,
                    smooth_sigma=smooth_val,
                    nx=nx,
                    ny=ny,
                    metadata=data_profile,
                )
            
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

        _, _, X, Y = core.xy_grid_from_bounds(self._c_bounds_min, self._c_bounds_max, nx, ny)
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

        _, _, X, Y = core.xy_grid_from_bounds(self._c_bounds_min, self._c_bounds_max, nx, ny)
        self._c_grid = (nx, ny, X, Y)

        self._tabs.selected_tab_index = 0
        self._update_slider_ranges()
        self._c_iso_slider.double_value = float(iso_level)
        self._status.text = "Data loaded. Use slice/iso controls."
        self._update_scene(fit_camera=True)

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
            elif sel_text == "WallStrength":
                res = self._n_W
                curve_color = [0.95, 0.95, 0.95, 1.0]
            elif sel_text == "WallField":
                res = self._n_B
                curve_color = [0.95, 0.95, 0.95, 1.0]
            else: # Result
                res = self._n_result
                prof = self._n_profile
                brac = self._n_bracing
                curve_color = [1.0, 0.5, 0.0, 1.0]  # Orange-ish

        if res is None or nx is None or ny is None:
            return

        idx = int(self._slice_slider.int_value)
        z = vut.slice_z(idx, res.shape[0], bmin, bmax)

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
        ls = vut.curves_to_lineset(curves, z)
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
            p_ls = vut.curves_to_lineset(p_curves, z)
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
            b_ls = vut.curves_to_lineset(b_curves, z)
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
    monitor_index: int = 0,
):
    app = gui.Application.instance
    app.initialize()
    viewer = NarniaCurveViewer(monitor_index=monitor_index)
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
    monitor_index: int = 0,
):
    app = gui.Application.instance
    app.initialize()
    viewer = NarniaCurveViewer(monitor_index=monitor_index)
    viewer._output_dir.text_value = output_dir
    # Defer scene updates until the GUI main thread is running.
    gui.Application.instance.post_to_main_thread(
        viewer._window,
        lambda: viewer.set_data(result_fields_2d, bounds_min, bounds_max, iso_level=iso_level, profile_fields=profile_fields, iso_p=iso_p),
    )
    app.run()


# if __name__ == "__main__":
#     # If an output NPZ exists, use it; otherwise allow the user to compute from JSON in the UI.
#     output_path = Path("./output/processed_sdf_results.npz")
#     profile_json_path = Path("./alice_result/251120/ext/waveStackFields.json")

#     if output_path.exists() and profile_json_path.exists():
#         data = np.load(output_path, allow_pickle=True)
#         result_fields = data["result_fields"]
#         iso_level = float(data["iso_level"])
#         with open(profile_json_path, "r") as f:
#             meta = json.load(f)
#         iso_p, _, bounds_max, bounds_min = core.meta_data_info(meta)
        
#         # Try to load profile fields for preview if they exist in the same folder as the JSON
#         profile_fields, _ = core.stack_scalar_fields(meta)
        
#         run_app_from_data(result_fields, bounds_min, bounds_max, iso_level=iso_level, profile_fields=profile_fields, iso_p=iso_p)
#     else:
#         run_app()
