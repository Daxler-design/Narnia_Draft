import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering

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

        # Num Centroids Slider (Limit 1-6)
        row_k, self._num_centroids_slider, self._num_centroids_edit = vwg.create_slider_row(
            "Num Centroids", 1, 6, 5, None, is_int=True
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

        # --- State ---
        self.compute_state = vut.ViewState()
        self.viewer_state = vut.ViewState()

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
        state = self.compute_state if idx_tab == 0 else self.viewer_state
        res = state.result
        
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
        if self.compute_state.profile is None or self.compute_state.bracing is None:
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

        self.compute_state.result = core.compute_sf_operation(
            self.compute_state.profile,
            self.compute_state.bracing,
            iso_level_A=self.compute_state.iso_p_base + off_p,
            iso_level_B=self.compute_state.iso_b_base + off_b,
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
        state = self.compute_state if idx_tab == 0 else self.viewer_state
        bmin = state.bounds_min
        bmax = state.bounds_max
        
        if bmin is None or bmax is None:
            self._status.text = "No data loaded to fit camera."
            return
        self._fit_camera_to_current()

    def _on_select_output(self):
        vwg.show_file_dialog(
            self._window, 
            "Select Output Folder", 
            self._on_output_dir_done, 
            mode=gui.FileDialog.OPEN_DIR
        )

    def _on_output_dir_done(self, path):
        self._output_dir.text_value = path

    def _on_select_profile(self):
        vwg.show_file_dialog(
            self._window,
            "Select Profile JSON",
            self._on_profile_done,
            filters=[(".json", "JSON files (.json)")]
        )

    def _on_profile_done(self, path):
        self._profile_path.text_value = path

    def _on_select_bracing(self):
        vwg.show_file_dialog(
            self._window,
            "Select Bracing JSON",
            self._on_bracing_done,
            filters=[(".json", "JSON files (.json)")]
        )

    def _on_bracing_done(self, path):
        self._bracing_path.text_value = path

    def _on_select_npz(self):
        vwg.show_file_dialog(
            self._window,
            "Select NPZ File",
            self._on_npz_file_done,
            filters=[(".npz", "NPZ files (.npz)")]
        )

    def _on_npz_file_done(self, path):
        self._npz_path.text_value = path

    def _on_file_dialog_cancel(self):
        # Deprecated, handled by vwg.show_file_dialog
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
            current_bmin = self.viewer_state.bounds_min if self.viewer_state.bounds_min is not None else self.compute_state.bounds_min
            current_bmax = self.viewer_state.bounds_max if self.viewer_state.bounds_max is not None else self.compute_state.bounds_max
            
            bmin = data.get("bounds_min", current_bmin)
            bmax = data.get("bounds_max", current_bmax)
            
            if bmin is None or bmax is None:
                bmin = np.array([0.0, 0.0, 0.0])
                bmax = np.array([100.0, 100.0, 100.0])

            self.viewer_state.result = res_fields
            self.viewer_state.profile = prof_fields
            self.viewer_state.bracing = brac_fields
            
            self.viewer_state.bounds_min = bmin
            self.viewer_state.bounds_max = bmax
            self.viewer_state.iso_p_base = iso
            
            num_fields, nx, ny = core.infer_grid_from_scalar_fields(res_fields)
            x = np.linspace(bmin[0], bmax[0], nx)
            y = np.linspace(bmin[1], bmax[1], ny)
            X, Y = np.meshgrid(x, y, indexing="xy")
            self.viewer_state.grid = (nx, ny, X, Y)

            # Update View Selector
            self._n_view_selector.clear_items()
            self._n_view_selector.add_item("Result")
            if self.viewer_state.profile is not None:
                self._n_view_selector.add_item("Profile")
            if self.viewer_state.bracing is not None:
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
        state = self.compute_state if idx_tab == 0 else self.viewer_state
        res = state.result
        bmin = state.bounds_min
        bmax = state.bounds_max
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
                prof = state.profile
                if prof is not None:
                    save_dict["profile_fields"] = prof
            
            if self._chk_export_bracing.checked:
                brac = state.bracing
                if brac is not None:
                    save_dict["bracing_fields"] = brac

            np.savez(output_path, **save_dict)
            self._status.text = f"Exported: {filename}"
            print(f"Successfully exported results to {output_path}")
        except Exception as e:
            self._status.text = f"Export failed: {e}"

    def _on_slice_changed(self, _):
        idx_tab = self._tabs.selected_tab_index
        state = self.compute_state if idx_tab == 0 else self.viewer_state
        if state.result is None:
            return
        self._update_scene(fit_camera=False)

    def _on_iso_changed(self, _):
        idx_tab = self._tabs.selected_tab_index
        state = self.compute_state if idx_tab == 0 else self.viewer_state
        if state.result is None:
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

        iso_p, _, bmax_p, bmin_p = core.meta_data_info(data_profile)
        self.compute_state.iso_p_base = iso_p
        self.compute_state.profile, _ = core.stack_scalar_fields(data_profile)
        
        self.compute_state.bounds_min = bmin_p
        self.compute_state.bounds_max = bmax_p
        num_fields, nx, ny = core.infer_grid_from_scalar_fields(self.compute_state.profile)

        if self._chk_generate_bracing.checked:
            self._status.text = "Generating bracing..."
            self._window.set_needs_layout()
            
            k = int(self._num_centroids_slider.int_value)
            print(f"Generating bracing from centroids (k={k})...")
            self.compute_state.bracing = np.zeros_like(self.compute_state.profile)
            prev_centroids = None
            
            for i in range(num_fields):
                slice_2d = self.compute_state.profile[i].reshape((ny, nx))
                mask = core.get_profile_mask(slice_2d, iso_level=self.compute_state.iso_p_base or 0.0)
                
                centroids = core.generate_centroids(mask, k=k, prev_centroids=prev_centroids)
                centroids = core.constrain_centroids_to_mask(centroids, mask)
                
                voronoi_sdf_flat = core.compute_voronoi_sdf((ny, nx), centroids)
                self.compute_state.bracing[i] = voronoi_sdf_flat.ravel()
                prev_centroids = centroids
                
                if i % 10 == 0:
                    print(f"Generated bracing for slice {i}/{num_fields}")
            
            self.compute_state.iso_b_base = 0.0
        else:
            self._status.text = "Loading bracing data..."
            self._window.set_needs_layout()
            
            bp = self._resolve_path(bracing_json_path)
            if not bp.exists():
                raise FileNotFoundError(f"Bracing JSON not found: {bp}")
            with open(bp, "r") as f:
                data_bracing = json.load(f)
            iso_b, _, _, _ = core.meta_data_info(data_bracing)
            self.compute_state.iso_b_base = iso_b
            self.compute_state.bracing, _ = core.stack_scalar_fields(data_bracing)

        if self.compute_state.bracing.shape != self.compute_state.profile.shape:
            raise ValueError("Bracing/Profile scalar field arrays must have the same shape")

        self._status.text = "Computing boolean operation..."
        self._window.set_needs_layout()
        
        self._update_boolean_result()

        x = np.linspace(self.compute_state.bounds_min[0], self.compute_state.bounds_max[0], nx)
        y = np.linspace(self.compute_state.bounds_min[1], self.compute_state.bounds_max[1], ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        self.compute_state.grid = (nx, ny, X, Y)

        self._update_slider_ranges()
        self._c_iso_slider.double_value = 0.0
        self._c_iso_edit.double_value = 0.0

    def set_data(self, result_fields_2d: np.ndarray, bounds_min, bounds_max, iso_level: float = 0.0, profile_fields: Optional[np.ndarray] = None, iso_p: float = 0.0):
        # Note: set_data is used when launching from main.py with pre-computed result.
        self.compute_state.result = np.asarray(result_fields_2d)
        self.compute_state.profile = profile_fields
        self.compute_state.bracing = None
        
        self.compute_state.iso_p_base = iso_p
        self.compute_state.iso_b_base = 0.0
        
        self.compute_state.bounds_min = bounds_min
        self.compute_state.bounds_max = bounds_max

        try:
            num_fields, nx, ny = core.infer_grid_from_scalar_fields(self.compute_state.result)
        except ValueError as e:
            self._status.text = f"Load error: {e}"
            return

        x = np.linspace(self.compute_state.bounds_min[0], self.compute_state.bounds_max[0], nx)
        y = np.linspace(self.compute_state.bounds_min[1], self.compute_state.bounds_max[1], ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        self.compute_state.grid = (nx, ny, X, Y)

        self._tabs.selected_tab_index = 0
        self._update_slider_ranges()
        self._c_iso_slider.double_value = float(iso_level)
        self._status.text = "Data loaded. Use slice/iso controls."
        self._update_scene(fit_camera=True)

    def _collect_scene_inputs(self):
        idx_tab = self._tabs.selected_tab_index
        inputs = {
            "res": None, "prof": None, "brac": None,
            "bmin": None, "bmax": None,
            "nx": None, "ny": None, "X": None, "Y": None,
            "iso": 0.0, "iso_p": 0.0, "iso_b": 0.0,
            "curve_color": [1.0, 1.0, 1.0, 1.0],
            "slice_idx": int(self._slice_slider.int_value)
        }

        if idx_tab == 0: # Compute
            state = self.compute_state
            inputs["res"] = state.result
            inputs["prof"] = state.profile
            # brac remains None for Compute tab
            
            inputs["bmin"], inputs["bmax"] = state.bounds_min, state.bounds_max
            inputs["nx"], inputs["ny"], inputs["X"], inputs["Y"] = state.grid
            
            iso_p_base = state.iso_p_base
            iso_b_base = state.iso_b_base
            inputs["iso"] = float(self._c_iso_slider.double_value)
            inputs["curve_color"] = [0.1, 0.7, 0.95, 1.0]
            
            # Offsets
            off_p = self._profile_offset_slider.double_value
            off_b = self._bracing_offset_slider.double_value
            inputs["iso_p"] = iso_p_base + off_p
            inputs["iso_b"] = iso_b_base + off_b
            
        else: # NPZ
            state = self.viewer_state
            sel_idx = self._n_view_selector.selected_index
            sel_text = self._n_view_selector.get_item(sel_idx) if sel_idx >= 0 else "Result"
            
            inputs["bmin"], inputs["bmax"] = state.bounds_min, state.bounds_max
            inputs["nx"], inputs["ny"], inputs["X"], inputs["Y"] = state.grid
            inputs["iso"] = float(self._n_iso_slider.double_value)
            
            if sel_text == "Profile":
                inputs["res"] = state.profile
                inputs["curve_color"] = [0.8, 0.8, 0.8, 1.0]
            elif sel_text == "Bracing":
                inputs["res"] = state.bracing
                inputs["curve_color"] = [0.2, 0.8, 0.2, 1.0]
            else: # Result
                inputs["res"] = state.result
                inputs["prof"] = state.profile
                inputs["brac"] = state.bracing
                inputs["curve_color"] = [1.0, 0.5, 0.0, 1.0]
            
            # No offsets in NPZ viewer
            inputs["iso_p"] = state.iso_p_base
            inputs["iso_b"] = state.iso_b_base

        return inputs

    def _build_geometries(self, inputs):
        res = inputs["res"]
        if res is None or inputs["nx"] is None:
            return None

        idx = inputs["slice_idx"]
        ny, nx = inputs["ny"], inputs["nx"]
        X, Y = inputs["X"], inputs["Y"]
        z = vut.slice_z(idx, res.shape[0], inputs["bmin"], inputs["bmax"])
        
        geoms = {
            "curves": None, "profile": None, "bracing": None,
            "curve_count": 0
        }

        # 1. Main Curves
        slice_2d = res[idx].reshape((ny, nx))
        curves = core.iso_curves_for_slice_2d(slice_2d, inputs["iso"], X, Y)
        geoms["curve_count"] = len(curves)
        ls = vut.curves_to_lineset(curves, z)
        if ls is not None:
            mat = rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 2.0
            mat.base_color = inputs["curve_color"]
            geoms["curves"] = (ls, mat)

        # 2. Profile Curves
        if inputs["prof"] is not None:
            p_slice_2d = inputs["prof"][idx].reshape((ny, nx))
            p_curves = core.iso_curves_for_slice_2d(p_slice_2d, inputs["iso_p"], X, Y)
            p_ls = vut.curves_to_lineset(p_curves, z)
            if p_ls is not None:
                p_mat = rendering.MaterialRecord()
                p_mat.shader = "unlitLine"
                p_mat.line_width = 1.0
                p_mat.base_color = [0.8, 0.8, 0.8, 0.6]
                geoms["profile"] = (p_ls, p_mat)

        # 3. Bracing Curves
        if inputs["brac"] is not None:
            b_slice_2d = inputs["brac"][idx].reshape((ny, nx))
            b_curves = core.iso_curves_for_slice_2d(b_slice_2d, inputs["iso_b"], X, Y)
            b_ls = vut.curves_to_lineset(b_curves, z)
            if b_ls is not None:
                b_mat = rendering.MaterialRecord()
                b_mat.shader = "unlitLine"
                b_mat.line_width = 1.0
                b_mat.base_color = [0.2, 0.8, 0.2, 0.6]
                geoms["bracing"] = (b_ls, b_mat)
                
        return geoms

    def _apply_geometries(self, geoms, inputs, fit_camera):
        if geoms is None:
            return

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

        # Add new geometries
        if geoms["curves"]:
            self._curves_geom = geoms["curves"][0]
            self._scene_widget.scene.add_geometry("curves", *geoms["curves"])
            
        if geoms["profile"]:
            self._profile_geom = geoms["profile"][0]
            self._scene_widget.scene.add_geometry("profile", *geoms["profile"])
            
        if geoms["bracing"]:
            self._bracing_geom = geoms["bracing"][0]
            self._scene_widget.scene.add_geometry("bracing", *geoms["bracing"])

        self._status.text = f"Slice {inputs['slice_idx']} | iso {inputs['iso']:.6g} | curves {geoms['curve_count']}"

        # Compute bbox
        bbox = None
        for g in (self._curves_geom, self._profile_geom, self._bracing_geom):
            if g is None: continue
            try:
                gb = g.get_axis_aligned_bounding_box()
            except Exception: continue
            
            if bbox is None:
                bbox = gb
            else:
                min_b = np.minimum(bbox.min_bound, gb.min_bound)
                max_b = np.maximum(bbox.max_bound, gb.max_bound)
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)

        self._current_bbox = bbox

        if fit_camera:
            self._fit_camera_to_current()

    def _update_scene(self, fit_camera: bool):
        inputs = self._collect_scene_inputs()
        geoms = self._build_geometries(inputs)
        self._apply_geometries(geoms, inputs, fit_camera)

    def _fit_camera_to_current(self):
        bbox = self._current_bbox
        if bbox is None:
            idx_tab = self._tabs.selected_tab_index
            state = self.compute_state if idx_tab == 0 else self.viewer_state
            bmin = state.bounds_min
            bmax = state.bounds_max
            
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


