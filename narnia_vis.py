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
from gui import widgets as vwg
from gui import mesh_builders as mesh_build


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

        self._chk_generate_bracing = gui.Checkbox("Generate Bracing")
        self._chk_generate_bracing.checked = True
        self._chk_generate_bracing.set_on_checked(self._on_generate_toggled)

        # --- Bracing Generation Options ---
        self._gen_options_container = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        
        self._gen_options_container.add_child(gui.Label("Generation Method"))
        self._gen_method_combo = gui.Combobox()
        self._gen_method_combo.add_item("static-bracing")
        self._gen_method_combo.add_item("Key-Field Blend")
        self._gen_method_combo.add_item("Shape-Adaptive")
        self._gen_method_combo.add_item("Binary Splitting")
        self._gen_method_combo.set_on_selection_changed(self._on_gen_method_changed)
        self._gen_options_container.add_child(self._gen_method_combo)
        self._gen_options_container.add_fixed(5)

        # 1. Static Params (Num Centroids)
        self._static_params = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        row_k, self._num_centroids_slider, self._num_centroids_edit = vwg.create_slider_row(
            "Num Centroids", 1, 6, 5, None, is_int=True
        )
        self._static_params.add_child(row_k)
        
        # 2. KeyBlending Params
        self._keyblending_params = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        
        # Key Config Text
        self._keyblending_params.add_child(gui.Label("Keys (slice:k, ...)"))
        self._kb_keys_edit = gui.TextEdit()
        self._kb_keys_edit.text_value = "0:3, 30:4, 59:5"
        self._keyblending_params.add_child(self._kb_keys_edit)
        self._keyblending_params.add_fixed(5)

        # Smooth (Blend Factor)
        row_kb, self._kb_slider, self._kb_edit = vwg.create_slider_row(
            "Smooth Factor", 0.0, 1.0, 0.0, None
        )
        self._keyblending_params.add_child(row_kb)

        # Sigma (Ridge Width)
        row_sig, self._sigma_slider, self._sigma_edit = vwg.create_slider_row(
            "Sigma (Ridge Width)", 1.0, 20.0, 5.0, None
        )
        self._keyblending_params.add_child(row_sig)
        
        # Tau (Threshold)
        row_tau, self._tau_slider, self._tau_edit = vwg.create_slider_row(
            "Tau (Threshold)", 0.0, 1.0, 0.5, None
        )
        self._keyblending_params.add_child(row_tau)

        # Beta (SmoothMax)
        row_beta, self._beta_slider, self._beta_edit = vwg.create_slider_row(
            "Beta (Blend Softness)", 0.0, 20.0, 4.0, None
        )
        self._keyblending_params.add_child(row_beta)
        
        # 3. Adaptive Params
        self._adaptive_params = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        
        # Area per seed
        row_area, self._area_slider, self._area_edit = vwg.create_slider_row(
            "Area per Seed (px²)", 200.0, 3000.0, 600.0, None
        )
        self._adaptive_params.add_child(row_area)
        
        # K min
        row_kmin, self._kmin_slider, self._kmin_edit = vwg.create_slider_row(
            "K Min", 1, 10, 1, None, is_int=True
        )
        self._adaptive_params.add_child(row_kmin)
        
        # K max
        row_kmax, self._kmax_slider, self._kmax_edit = vwg.create_slider_row(
            "K Max", 3, 20, 12, None, is_int=True
        )
        self._adaptive_params.add_child(row_kmax)
        
        # Smooth sigma
        row_asigma, self._adaptive_smooth_slider, self._adaptive_smooth_edit = vwg.create_slider_row(
            "Z Smooth (sigma)", 0.0, 5.0, 1.5, None
        )
        self._adaptive_params.add_child(row_asigma)
        
        # Ramp slices
        row_ramp, self._ramp_slider, self._ramp_edit = vwg.create_slider_row(
            "Ramp Slices", 1, 10, 3, None, is_int=True
        )
        self._adaptive_params.add_child(row_ramp)
        
        # 4. Binary Splitting Params
        self._binary_params = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        
        # Info label
        self._binary_params.add_child(gui.Label("(Z-based: k grows evenly over slices)"))
        self._binary_params.add_fixed(5)
        
        # K start (dropdown: 1, 2, 4, 8, 16)
        self._binary_params.add_child(gui.Label("K Start (cells)"))
        self._binary_k_start_combo = gui.Combobox()
        for k in [1, 2, 4, 8, 16]:
            self._binary_k_start_combo.add_item(str(k))
        self._binary_k_start_combo.selected_index = 1  # Default to 2
        self._binary_params.add_child(self._binary_k_start_combo)
        self._binary_params.add_fixed(5)
        
        # K max (dropdown: 1, 2, 4, 8, 16, 32)
        self._binary_params.add_child(gui.Label("K Max (cells)"))
        self._binary_k_max_combo = gui.Combobox()
        for k in [1, 2, 4, 8, 16, 32]:
            self._binary_k_max_combo.add_item(str(k))
        self._binary_k_max_combo.selected_index = 4  # Default to 16
        self._binary_params.add_child(self._binary_k_max_combo)
        self._binary_params.add_fixed(5)
        
        # Split offset
        row_offset, self._split_offset_slider, self._split_offset_edit = vwg.create_slider_row(
            "Split Offset", 1.0, 20.0, 5.0, None
        )
        self._binary_params.add_child(row_offset)
        
        # Smooth sigma
        row_bsigma, self._binary_smooth_slider, self._binary_smooth_edit = vwg.create_slider_row(
            "Z Smooth (sigma)", 0.0, 5.0, 1.5, None
        )
        self._binary_params.add_child(row_bsigma)
        
        # Ramp slices
        row_bramp, self._binary_ramp_slider, self._binary_ramp_edit = vwg.create_slider_row(
            "Ramp Slices", 1, 10, 3, None, is_int=True
        )
        self._binary_params.add_child(row_bramp)

        # Initial visibility
        self._static_params.visible = True
        self._keyblending_params.visible = False
        self._adaptive_params.visible = False
        self._binary_params.visible = False
        
        self._gen_options_container.add_child(self._static_params)
        self._gen_options_container.add_child(self._keyblending_params)
        self._gen_options_container.add_child(self._adaptive_params)
        self._gen_options_container.add_child(self._binary_params)
        # ----------------------------------

        self._btn_compute = gui.Button("Load / Re-generate Bracing")

        # --- Postprocess Bracing ---
        self._postprocess_panel = gui.Vert(0, gui.Margins(0, 5, 0, 5))
        self._postprocess_panel.add_child(gui.Label("--- Postprocess Bracing ---"))
        
        self._chk_enable_postprocess = gui.Checkbox("Enable Postprocess")
        self._chk_enable_postprocess.checked = False
        self._chk_enable_postprocess.set_on_checked(self._on_postprocess_param_changed)
        self._postprocess_panel.add_child(self._chk_enable_postprocess)

        # Close Radius (0–6)
        row_close, self._pp_close_slider, self._pp_close_edit = vwg.create_slider_row(
            "Close Radius", 0.0, 6.0, 0.0, self._on_postprocess_param_changed
        )
        self._postprocess_panel.add_child(row_close)

        # Min Area (0–2000)
        row_area, self._pp_area_slider, self._pp_area_edit = vwg.create_slider_row(
            "Min Area", 0.0, 2000.0, 0.0, self._on_postprocess_param_changed
        )
        self._postprocess_panel.add_child(row_area)

        # Temporal Window (1, 3, 5)
        row_temp, self._pp_temp_slider, self._pp_temp_edit = vwg.create_slider_row(
            "Temporal Window", 1, 5, 1, self._on_postprocess_param_changed, is_int=True
        )
        self._postprocess_panel.add_child(row_temp)


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
        self._compute_panel.add_child(self._gen_options_container)
        self._compute_panel.add_fixed(10)
        self._compute_panel.add_child(self._btn_compute)
        self._compute_panel.add_fixed(10)
        self._compute_panel.add_child(self._postprocess_panel)
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

        # --- TAB 3: MESH ---
        self._mesh_panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        self._mesh_panel.add_child(gui.Label("Source"))
        self._mesh_source_combo = gui.Combobox()
        self._mesh_source_combo.add_item("Compute")
        self._mesh_source_combo.add_item("NPZ Viewer")
        self._mesh_source_combo.add_item("Custom")
        self._mesh_source_combo.selected_index = 1
        self._mesh_panel.add_child(self._mesh_source_combo)
        self._mesh_panel.add_fixed(6)
        
        # Load Custom NPZ button
        self._btn_load_custom_npz = gui.Button("Load Custom NPZ...")
        self._mesh_panel.add_child(self._btn_load_custom_npz)
        self._mesh_panel.add_fixed(10)

        self._mesh_panel.add_child(gui.Label("Include"))
        h_mesh_opts = gui.Horiz(10)
        self._mesh_chk_result = gui.Checkbox("Result Mesh")
        self._mesh_chk_result.checked = False
        self._mesh_chk_profile = gui.Checkbox("Profile Mesh")
        self._mesh_chk_profile.checked = True
        self._mesh_chk_bracing = gui.Checkbox("Bracing Mesh")
        self._mesh_chk_bracing.checked = True
        h_mesh_opts.add_child(self._mesh_chk_result)
        h_mesh_opts.add_child(self._mesh_chk_profile)
        h_mesh_opts.add_child(self._mesh_chk_bracing)
        self._mesh_panel.add_child(h_mesh_opts)
        self._mesh_panel.add_fixed(10)

        # Marching Cubes Parameters
        self._mesh_panel.add_child(gui.Label("--- Marching Cubes Parameters ---"))
        
        row_m_height, self._mesh_height_slider, self._mesh_height_edit = vwg.create_slider_row(
            "Total Height", 0.1, 100.0, 10.0, None
        )
        row_m_interp, self._mesh_interp_slider, self._mesh_interp_edit = vwg.create_slider_row(
            "Z Interpolation", 0, 5, 2, None, is_int=True
        )
        
        self._mesh_panel.add_child(row_m_height)
        self._mesh_panel.add_child(row_m_interp)
        
        # Slice count info
        self._mesh_slice_info = gui.Label("")
        self._mesh_panel.add_child(self._mesh_slice_info)
        self._mesh_panel.add_fixed(10)
        
        # Individual Iso Overrides
        self._mesh_panel.add_child(gui.Label("--- Iso Overrides ---"))
        
        row_m_iso_r, self._mesh_iso_result_slider, self._mesh_iso_result_edit = vwg.create_slider_row(
            "Result Iso Override", -2.0, 2.0, 0.0, None
        )
        row_m_iso_p, self._mesh_iso_profile_slider, self._mesh_iso_profile_edit = vwg.create_slider_row(
            "Profile Iso Override", -2.0, 2.0, 0.0, None
        )
        row_m_iso_b, self._mesh_iso_bracing_slider, self._mesh_iso_bracing_edit = vwg.create_slider_row(
            "Bracing Iso Override", -2.0, 2.0, 0.0, None
        )
        
        self._mesh_panel.add_child(row_m_iso_r)
        self._mesh_panel.add_child(row_m_iso_p)
        self._mesh_panel.add_child(row_m_iso_b)
        self._mesh_panel.add_fixed(10)
        
        # Smoothing Parameters
        self._mesh_panel.add_child(gui.Label("--- Smoothing ---"))
        
        self._mesh_panel.add_child(gui.Label("Method"))
        self._mesh_smooth_method_combo = gui.Combobox()
        self._mesh_smooth_method_combo.add_item("None")
        self._mesh_smooth_method_combo.add_item("Laplacian")
        self._mesh_smooth_method_combo.add_item("Taubin")
        self._mesh_smooth_method_combo.add_item("Laplacian + Taubin")
        self._mesh_smooth_method_combo.selected_index = 1  # Default to Laplacian
        self._mesh_panel.add_child(self._mesh_smooth_method_combo)
        self._mesh_panel.add_fixed(5)
        
        row_m_smooth, self._mesh_smooth_slider, self._mesh_smooth_edit = vwg.create_slider_row(
            "Iterations", 0, 10, 2, None, is_int=True
        )
        self._mesh_panel.add_child(row_m_smooth)
        self._mesh_panel.add_fixed(10)

        self._btn_fit_mesh = gui.Button("Fit Camera")
        self._mesh_panel.add_child(self._btn_fit_mesh)
        self._mesh_panel.add_fixed(10)

        self._btn_generate_mesh = gui.Button("Update Mesh")
        self._mesh_panel.add_child(self._btn_generate_mesh)
        self._mesh_panel.add_fixed(10)
        
        # Export Location
        self._mesh_panel.add_child(gui.Label("--- Export ---"))
        row_export_loc, self._mesh_export_location = vwg.create_file_input_row(
            "Export Location",
            "./output/mesh",
            self._on_select_mesh_export_location
        )
        self._mesh_panel.add_child(row_export_loc)
        self._mesh_panel.add_fixed(6)
        
        self._btn_export_mesh = gui.Button("Export Mesh (.obj)")
        self._mesh_panel.add_child(self._btn_export_mesh)
        self._mesh_panel.add_fixed(10)
        self._mesh_panel.add_child(gui.Label("Note: Uses marching cubes."))

        self._tabs.add_tab("Mesh", self._mesh_panel)
        self._mesh_tab_index = 2

        # --- SHARED CONTROLS (Bottom) ---
        self._panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self._panel.preferred_width = 360
        self._panel.add_child(self._tabs)
        self._panel.add_fixed(16)
        
        # Visualization panel (hide in mesh tab)
        self._viz_label = gui.Label("--- Visualization ---")
        self._panel.add_child(self._viz_label)

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

        # Store slice label for visibility control
        self._slice_label = gui.Label("Slice")
        self._panel.add_child(self._slice_label)
        self._panel.add_child(self._slice_slider)
        self._panel.add_fixed(10)
        self._panel.add_child(self._btn_fit)
        self._panel.add_fixed(16)
        
        # Export panel (hide in mesh tab)
        self._export_label = gui.Label("--- Export ---")
        self._panel.add_child(self._export_label)
        self._panel.add_child(gui.Label("Output Directory"))
        
        h = gui.Horiz(4)
        h.add_child(self._output_dir)
        h.add_child(self._btn_select_output)
        self._panel.add_child(h)
        
        self._export_opts_horiz = gui.Horiz(10)
        self._export_opts_horiz.add_child(self._chk_export_profile)
        self._export_opts_horiz.add_child(self._chk_export_bracing)
        self._panel.add_child(self._export_opts_horiz)
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
        self._btn_fit_mesh.set_on_clicked(self._on_fit)
        self._btn_select_output.set_on_clicked(self._on_select_output)
        self._btn_export.set_on_clicked(self._on_export)
        # self._btn_select_npz is now handled in _create_file_input_row callback
        self._btn_load_npz.set_on_clicked(self._on_load_npz)
        self._btn_generate_mesh.set_on_clicked(self._on_generate_mesh)
        self._btn_export_mesh.set_on_clicked(self._on_export_mesh)
        self._btn_load_custom_npz.set_on_clicked(self._on_load_custom_npz)
        self._slice_slider.set_on_value_changed(self._on_slice_changed)
        self._tabs.set_on_selected_tab_changed(self._on_tab_changed)
        
        # Wire Z interpolation slider for live slice count update
        self._mesh_interp_slider.set_on_value_changed(self._on_z_interp_changed)

        # Sync axis inset as the user navigates the main viewport.
        self._scene_widget.set_on_mouse(self._on_mouse)

        # --- State ---
        self.compute_state = vut.ViewState()
        self.viewer_state = vut.ViewState()
        self.custom_state = vut.ViewState()  # For custom NPZ loaded from Mesh tab
        self.custom_npz_path: Optional[str] = None  # Track custom NPZ filename

        self._curves_geom: Optional[o3d.geometry.LineSet] = None
        self._profile_geom: Optional[o3d.geometry.LineSet] = None
        self._bracing_geom: Optional[o3d.geometry.LineSet] = None
        self._overlay_geom: Optional[o3d.geometry.Geometry] = None
        self._current_bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = None
        self._result_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self._profile_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self._bracing_mesh: Optional[o3d.geometry.TriangleMesh] = None

    def _on_gen_method_changed(self, name, index):
        is_static = (index == 0)
        is_keyblend = (index == 1)
        is_adaptive = (index == 2)
        is_binary = (index == 3)
        
        self._static_params.visible = is_static
        self._keyblending_params.visible = is_keyblend
        self._adaptive_params.visible = is_adaptive
        self._binary_params.visible = is_binary
        self._window.set_needs_layout()

    def _on_view_channel_changed(self, name, index):
        self._update_scene(fit_camera=False)
        self._update_slider_ranges()

    def _on_tab_changed(self, index):
        # Hide/show visualization and export panels based on tab
        is_mesh_tab = (index == self._mesh_tab_index)
        
        # Hide slice controls and export panel in mesh tab
        self._viz_label.visible = not is_mesh_tab
        self._slice_label.visible = not is_mesh_tab
        self._slice_slider.visible = not is_mesh_tab
        self._btn_fit.visible = not is_mesh_tab
        self._export_label.visible = not is_mesh_tab
        self._output_dir.visible = not is_mesh_tab
        self._btn_select_output.visible = not is_mesh_tab
        self._export_opts_horiz.visible = not is_mesh_tab
        self._btn_export.visible = not is_mesh_tab
        
        self._window.set_needs_layout()
        
        if index == self._mesh_tab_index:
            self._clear_curve_geometries()
            # Try to show cached meshes from current source
            self._refresh_mesh_from_cache()
            # Update slice count if data available
            self._on_z_interp_changed(self._mesh_interp_slider.int_value)
            if self._profile_mesh is None and self._bracing_mesh is None and self._result_mesh is None:
                self._status.text = "Mesh view. Select source and generate meshes."
            return

        self._clear_mesh_geometries()
        # Refresh scene when switching tabs to show the correct data
        self._update_scene(fit_camera=False)
        self._update_slider_ranges()

    def _update_slider_ranges(self):
        idx_tab = self._tabs.selected_tab_index
        if idx_tab == self._mesh_tab_index:
            return
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
        self._gen_options_container.visible = checked
        self._window.set_needs_layout()
    
    def _invalidate_mesh_cache(self, state: vut.ViewState):
        """Clear mesh cache when parameters change."""
        state.profile_mesh_cache = None
        state.bracing_mesh_cache = None
        state.result_mesh_cache = None
        state.mesh_params_cache = None

    def _on_boolean_param_changed(self, *args):
        if self.compute_state.profile is None or self.compute_state.bracing is None:
            return
        # If user is tweaking compute params, ensure we are on the compute tab
        if self._tabs.selected_tab_index != 0:
            self._tabs.selected_tab_index = 0
        
        # Invalidate mesh cache when boolean params change
        self._invalidate_mesh_cache(self.compute_state)
        
        self._update_boolean_result()
        self._update_scene(fit_camera=False)

    def _on_postprocess_param_changed(self, *args):
        if self.compute_state.bracing is None or self.compute_state.profile is None:
            return
        
        # Invalidate mesh cache when postprocess params change
        self._invalidate_mesh_cache(self.compute_state)
        
        if self._chk_enable_postprocess.checked:
            state = self.compute_state
            # Re-run cleaning logic on current raw bracing
            state.bracing_clean = core.postprocess_bracing_fields(
                state.bracing,
                state.profile,
                iso_profile=state.iso_p_base + self._profile_offset_slider.double_value,
                iso_brace=state.iso_b_base + self._bracing_offset_slider.double_value,
                close_radius=self._pp_close_slider.double_value,
                min_area=self._pp_area_slider.double_value,
                temporal_window=int(self._pp_temp_slider.double_value)
            )
        else:
            self.compute_state.bracing_clean = None
            
        self._update_boolean_result()
        self._update_scene(fit_camera=False)


    def _update_boolean_result(self):
        mode = self._op_mode_combo.get_item(self._op_mode_combo.selected_index)
        off_p = self._profile_offset_slider.double_value
        off_b = self._bracing_offset_slider.double_value

        # Use bracing_clean if available, else fallback to bracing
        brac = self.compute_state.bracing_clean if self.compute_state.bracing_clean is not None else self.compute_state.bracing

        self.compute_state.result = core.compute_sf_operation(
            self.compute_state.profile,
            brac,
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
            
            # Invalidate mesh cache after recomputing
            self._invalidate_mesh_cache(self.compute_state)
            
            self._status.text = "Computed. Use slice/iso controls."
            self._update_scene(fit_camera=True)
            self._mesh_source_combo.selected_index = 0
        except Exception as e:
            self._status.text = f"Compute failed: {e}"

    def _on_fit(self):
        idx_tab = self._tabs.selected_tab_index
        if idx_tab == self._mesh_tab_index:
            if self._current_bbox is None:
                self._status.text = "No mesh to fit camera."
                return
            self._fit_camera_to_current()
            return
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
    
    def _on_select_mesh_export_location(self):
        """Browse button callback for mesh export location."""
        vwg.show_file_dialog(
            self._window,
            "Select Mesh Export Folder",
            self._on_mesh_export_location_done,
            mode=gui.FileDialog.OPEN_DIR
        )
    
    def _on_mesh_export_location_done(self, path):
        """Callback when mesh export folder is selected."""
        self._mesh_export_location.text_value = path

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
    
    def _on_load_custom_npz(self):
        """Load a custom NPZ file from the Mesh tab, auto-detect available fields."""
        vwg.show_file_dialog(
            self._window,
            "Select Custom NPZ File",
            self._on_custom_npz_file_done,
            filters=[(".npz", "NPZ files (.npz)")]
        )
    
    def _on_custom_npz_file_done(self, path):
        """Callback when custom NPZ is selected - load and auto-populate mesh tab."""
        self._status.text = "Loading custom NPZ file..."
        self._window.set_needs_layout()
        
        npz_path = Path(path)
        if not npz_path.exists():
            self._status.text = f"File not found: {npz_path}"
            return
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Auto-detect available fields
            res_fields = data.get("result_fields", None)
            if res_fields is not None and (res_fields.ndim == 0 or res_fields.size == 0):
                res_fields = None
            
            prof_fields = data.get("profile_fields", None)
            if prof_fields is not None and (prof_fields.ndim == 0 or prof_fields.size == 0):
                prof_fields = None
            
            brac_fields = data.get("bracing_fields", None)
            if brac_fields is not None and (brac_fields.ndim == 0 or brac_fields.size == 0):
                brac_fields = None
            
            brac_clean = data.get("bracing_clean_fields", None)
            if brac_clean is not None and (brac_clean.ndim == 0 or brac_clean.size == 0):
                brac_clean = None
            
            # Check if at least one field exists
            if res_fields is None and prof_fields is None and brac_fields is None:
                self._status.text = "No valid fields found in NPZ file."
                return
            
            # Load bounds
            current_bmin = self.custom_state.bounds_min if self.custom_state.bounds_min is not None else (
                self.viewer_state.bounds_min if self.viewer_state.bounds_min is not None else self.compute_state.bounds_min
            )
            current_bmax = self.custom_state.bounds_max if self.custom_state.bounds_max is not None else (
                self.viewer_state.bounds_max if self.viewer_state.bounds_max is not None else self.compute_state.bounds_max
            )
            
            bmin = data.get("bounds_min", current_bmin)
            bmax = data.get("bounds_max", current_bmax)
            
            if bmin is None or bmax is None:
                bmin = np.array([0.0, 0.0, 0.0])
                bmax = np.array([100.0, 100.0, 100.0])
            
            # Load iso level
            iso = float(data.get("iso_level", 0.0))
            
            # Load total_height if available
            total_height = data.get("total_height", None)
            if total_height is not None:
                try:
                    total_height = float(total_height) if np.isscalar(total_height) or total_height.size == 1 else 10.0
                except:
                    total_height = 10.0
            else:
                total_height = 10.0
            
            # Update custom state
            self.custom_state.result = res_fields
            self.custom_state.profile = prof_fields
            self.custom_state.bracing = brac_fields
            self.custom_state.bracing_clean = brac_clean
            self.custom_state.bounds_min = bmin
            self.custom_state.bounds_max = bmax
            self.custom_state.iso_p_base = iso
            self.custom_state.iso_b_base = iso
            
            # Infer grid from first available field
            first_field = res_fields if res_fields is not None else (prof_fields if prof_fields is not None else brac_fields)
            num_fields, nx, ny = core.infer_grid_from_scalar_fields(first_field)
            x = np.linspace(bmin[0], bmax[0], nx)
            y = np.linspace(bmin[1], bmax[1], ny)
            X, Y = np.meshgrid(x, y, indexing="xy")
            self.custom_state.grid = (nx, ny, X, Y)
            
            # Store custom NPZ filename
            self.custom_npz_path = npz_path.name
            
            # Update mesh source dropdown to show "Custom: filename"
            self._mesh_source_combo.remove_item(2)  # Remove old "Custom" entry
            self._mesh_source_combo.add_item(f"Custom: {npz_path.name}")
            self._mesh_source_combo.selected_index = 2  # Select the custom source
            
            # Update mesh tab UI
            self._mesh_height_slider.double_value = total_height
            self._mesh_height_edit.double_value = total_height
            
            # Auto-check mesh checkboxes based on available fields
            self._mesh_chk_result.checked = (res_fields is not None)
            self._mesh_chk_profile.checked = (prof_fields is not None)
            self._mesh_chk_bracing.checked = (brac_fields is not None or brac_clean is not None)
            
            # Clear existing meshes (custom state has fresh cache)
            self._clear_mesh_geometries()
            
            self._status.text = f"Loaded custom NPZ: {npz_path.name} - select source and click Update Mesh"
            self._window.set_needs_layout()
            
        except Exception as e:
            self._status.text = f"Load custom NPZ failed: {e}"

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

            brac_clean = data.get("bracing_clean_fields", None)
            if brac_clean is not None and brac_clean.ndim == 0:
                brac_clean = None

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
            self.viewer_state.bracing_clean = brac_clean
            
            self.viewer_state.bounds_min = bmin
            self.viewer_state.bounds_max = bmax
            self.viewer_state.iso_p_base = iso
            
            # Load total_height if available, otherwise default to 10.0
            total_height = data.get("total_height", None)
            if total_height is not None:
                try:
                    total_height = float(total_height) if np.isscalar(total_height) or total_height.size == 1 else 10.0
                except:
                    total_height = 10.0
            else:
                total_height = 10.0
            self._mesh_height_slider.double_value = total_height
            self._mesh_height_edit.double_value = total_height
            
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
            self._mesh_source_combo.selected_index = 1
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
            
            # Get total height from mesh tab slider (for future mesh generation)
            total_height = float(self._mesh_height_slider.double_value)
            
            save_dict = {
                "result_fields": res,
                "iso_level": iso_level,
                "bounds_min": bmin,
                "bounds_max": bmax,
                "total_height": total_height,
            }
            
            # Add optional fields if requested
            if self._chk_export_profile.checked:
                prof = state.profile
                if prof is not None:
                    save_dict["profile_fields"] = prof
            
            if self._chk_export_bracing.checked:
                brac = state.bracing_clean if state.bracing_clean is not None else state.bracing
                if brac is not None:
                    save_dict["bracing_fields"] = brac

            np.savez(output_path, **save_dict)
            self._status.text = f"Exported: {filename}"
            print(f"Successfully exported results to {output_path}")
        except Exception as e:
            self._status.text = f"Export failed: {e}"

    def _on_slice_changed(self, _):
        if self._tabs.selected_tab_index == self._mesh_tab_index:
            return
        idx_tab = self._tabs.selected_tab_index
        state = self.compute_state if idx_tab == 0 else self.viewer_state
        if state.result is None:
            return
        self._update_scene(fit_camera=False)

    def _on_iso_changed(self, _):
        if self._tabs.selected_tab_index == self._mesh_tab_index:
            return
        idx_tab = self._tabs.selected_tab_index
        state = self.compute_state if idx_tab == 0 else self.viewer_state
        if state.result is None:
            return
        self._update_scene(fit_camera=False)
    
    def _on_z_interp_changed(self, value):
        """Update slice count info when Z interpolation changes."""
        if self._tabs.selected_tab_index != self._mesh_tab_index:
            return
        
        source_text = self._mesh_source_combo.get_item(self._mesh_source_combo.selected_index)
        if source_text == "Compute":
            state = self.compute_state
        elif source_text == "NPZ Viewer":
            state = self.viewer_state
        elif source_text.startswith("Custom"):
            state = self.custom_state
        else:
            return
        
        if state.result is None:
            self._mesh_slice_info.text = ""
            return
        
        original_slices = state.result.shape[0]
        z_interp = int(value)
        final_slices = original_slices + (original_slices - 1) * z_interp
        
        if z_interp > 0:
            self._mesh_slice_info.text = f"{final_slices} slices (×{z_interp + 1} interp)"
        else:
            self._mesh_slice_info.text = f"{final_slices} slices (no interp)"

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
            
            method_idx = self._gen_method_combo.selected_index
            if method_idx == 1: # Key-Field Blend
                smooth_val = self._kb_slider.double_value
                sigma_val = self._sigma_slider.double_value
                tau_val = self._tau_slider.double_value
                beta_val = self._beta_slider.double_value
                keys_str = self._kb_keys_edit.text_value
                
                # Parse keys
                keys_config = []
                try:
                    for part in keys_str.split(','):
                        s_idx, s_k = part.strip().split(':')
                        keys_config.append((int(s_idx), int(s_k)))
                except Exception as e:
                    self._status.text = f"Invalid Key Config: {e}"
                    print(f"Error parsing keys: {e}")
                    return

                print(f"Key-Field Blend: keys={keys_config}, smooth={smooth_val}, sigma={sigma_val}, tau={tau_val}, beta={beta_val}")
                self.compute_state.bracing = core.generate_bracing_keyfield_blend(
                    self.compute_state.profile,
                    self.compute_state.iso_p_base or 0.0,
                    nx, ny, 
                    keys_config=keys_config,
                    smooth=smooth_val,
                    sigma=sigma_val,
                    tau=tau_val,
                    beta=beta_val
                )
            elif method_idx == 2: # Shape-Adaptive
                area_val = self._area_slider.double_value
                kmin_val = int(self._kmin_slider.int_value)
                kmax_val = int(self._kmax_slider.int_value)
                smooth_sigma_val = self._adaptive_smooth_slider.double_value
                ramp_val = int(self._ramp_slider.int_value)
                
                print(f"Shape-Adaptive: area={area_val}, k=[{kmin_val},{kmax_val}], smooth={smooth_sigma_val}, ramp={ramp_val}")
                self.compute_state.bracing = core.generate_bracing_adaptive(
                    self.compute_state.profile,
                    self.compute_state.iso_p_base or 0.0,
                    nx, ny,
                    area_per_seed=area_val,
                    k_min=kmin_val,
                    k_max=kmax_val,
                    seed=42,
                    smooth_sigma=smooth_sigma_val,
                    ramp_slices=ramp_val
                )
            elif method_idx == 3: # Binary Splitting
                k_start_powers = [1, 2, 4, 8, 16]
                k_max_powers = [1, 2, 4, 8, 16, 32]
                k_start_val = k_start_powers[self._binary_k_start_combo.selected_index]
                k_max_val = k_max_powers[self._binary_k_max_combo.selected_index]
                offset_val = self._split_offset_slider.double_value
                binary_smooth_val = self._binary_smooth_slider.double_value
                binary_ramp_val = int(self._binary_ramp_slider.int_value)
                
                print(f"Binary Splitting: k={k_start_val}→{k_max_val} (Z-based), offset={offset_val}, smooth={binary_smooth_val}, ramp={binary_ramp_val}")
                self.compute_state.bracing = core.generate_bracing_adaptive_binary(
                    self.compute_state.profile,
                    self.compute_state.iso_p_base or 0.0,
                    nx, ny,
                    area_per_cell=700.0,  # Unused but required by API
                    k_start=k_start_val,
                    k_max=k_max_val,
                    split_offset=offset_val,
                    seed=42,
                    smooth_sigma=binary_smooth_val,
                    ramp_slices=binary_ramp_val
                )
            else: # static-bracing (method_idx == 0)
                self.compute_state.bracing = core.generate_bracing_static(
                    self.compute_state.profile,
                    self.compute_state.iso_p_base or 0.0,
                    nx, ny, k
                )
            
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

        # Update grid info first so postprocess/boolean can use it
        self.compute_state.grid = (nx, ny, None, None) # X,Y handled below
        x = np.linspace(self.compute_state.bounds_min[0], self.compute_state.bounds_max[0], nx)
        y = np.linspace(self.compute_state.bounds_min[1], self.compute_state.bounds_max[1], ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        self.compute_state.grid = (nx, ny, X, Y)

        # Trigger postprocess logic (which also updates boolean and scene)
        self._on_postprocess_param_changed()

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
        self._mesh_source_combo.selected_index = 0
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
            inputs["brac"] = state.bracing_clean if state.bracing_clean is not None else state.bracing
            
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
                inputs["res"] = state.bracing_clean if state.bracing_clean is not None else state.bracing
                inputs["curve_color"] = [0.1, 0.7, 0.95, 1.0]
            else: # Result
                inputs["res"] = state.result
                inputs["prof"] = state.profile
                inputs["brac"] = state.bracing_clean if state.bracing_clean is not None else state.bracing
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

        # Grid spacing (world units per pixel)
        x_vec = X[0, :] if np.asarray(X).ndim == 2 else np.asarray(X)
        y_vec = Y[:, 0] if np.asarray(Y).ndim == 2 else np.asarray(Y)
        dx = float(abs(x_vec[1] - x_vec[0])) if x_vec.shape[0] > 1 else 1.0
        dy = float(abs(y_vec[1] - y_vec[0])) if y_vec.shape[0] > 1 else 1.0
        
        geoms = {
            "curves": None, "profile": None, "bracing": None,
            "curve_count": 0
        }

        # 1. Main Curves
        slice_2d = res[idx].reshape((ny, nx))
        curves = core.iso_curves_for_slice_2d(slice_2d, inputs["iso"], X, Y)
        
        # No filtering/simplification

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
            # No filtering/simplification
            b_ls = vut.curves_to_lineset(b_curves, z)
            if b_ls is not None:
                b_mat = rendering.MaterialRecord()
                b_mat.shader = "unlitLine"
                b_mat.line_width = 1.0
                b_mat.base_color = [0.1, 0.7, 0.95, 0.6]
                geoms["bracing"] = (b_ls, b_mat)
                
        return geoms

    def _clear_curve_geometries(self):
        for name in ["curves", "profile", "bracing", "overlay"]:
            try:
                self._scene_widget.scene.remove_geometry(name)
            except Exception:
                pass
        self._curves_geom = None
        self._profile_geom = None
        self._bracing_geom = None
        self._overlay_geom = None

    def _clear_mesh_geometries(self):
        for name in ["result_mesh", "profile_mesh", "bracing_mesh"]:
            try:
                self._scene_widget.scene.remove_geometry(name)
            except Exception:
                pass
        self._result_mesh = None
        self._profile_mesh = None
        self._bracing_mesh = None

    def _apply_geometries(self, geoms, inputs, fit_camera):
        if geoms is None:
            return

        self._clear_curve_geometries()
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
        if self._tabs.selected_tab_index == self._mesh_tab_index:
            return
        inputs = self._collect_scene_inputs()
        geoms = self._build_geometries(inputs)
        self._apply_geometries(geoms, inputs, fit_camera)

    def _ensure_state_grid(self, state):
        nx, ny, X, Y = state.grid
        if nx is not None and ny is not None and X is not None and Y is not None:
            return nx, ny, X, Y

        fields = state.result
        if fields is None:
            fields = state.profile
        if fields is None:
            fields = state.bracing
        if fields is None:
            return None

        if state.bounds_min is None or state.bounds_max is None:
            return None

        num_fields, nx, ny = core.infer_grid_from_scalar_fields(fields)
        x = np.linspace(state.bounds_min[0], state.bounds_max[0], nx)
        y = np.linspace(state.bounds_min[1], state.bounds_max[1], ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        state.grid = (nx, ny, X, Y)
        return state.grid

    def _build_mesh_from_fields(
        self,
        fields,
        iso_level,
        bounds_min,
        bounds_max,
        nx,
        ny,
        total_height,
        z_interp_steps,
        smooth_method,
        smooth_iterations,
    ):
        """
        Generate mesh using marching cubes algorithm.
        Delegates to gui.mesh_builders.build_mesh_from_fields().
        """
        return mesh_build.build_mesh_from_fields(
            fields, iso_level, bounds_min, bounds_max, nx, ny,
            total_height, z_interp_steps, smooth_method, smooth_iterations
        )

    def _apply_mesh_geometries(self, result_mesh, profile_mesh, bracing_mesh, fit_camera: bool):
        """Apply mesh geometries to the scene with proper materials."""
        self._clear_curve_geometries()
        self._clear_mesh_geometries()
        self._current_bbox = None

        # Store meshes in instance variables
        self._result_mesh = result_mesh
        self._profile_mesh = profile_mesh
        self._bracing_mesh = bracing_mesh

        # Use helper to add geometries and get bbox
        bbox = mesh_build.apply_mesh_geometries(
            self._scene_widget, result_mesh, profile_mesh, bracing_mesh
        )

        self._current_bbox = bbox
        if fit_camera and bbox is not None:
            self._fit_camera_to_current()

    def _on_generate_mesh(self):
        """Generate mesh using marching cubes with caching support."""
        # Run heavy computation in background to avoid freezing UI
        self._status.text = "Generating mesh (this may take a moment)..."
        self._window.set_needs_layout()
        
        # Disable button during generation
        self._btn_generate_mesh.enabled = False
        
        def compute_meshes():
            """Background thread computation."""
            source_text = self._mesh_source_combo.get_item(self._mesh_source_combo.selected_index)
            
            # Determine which state to use based on source
            if source_text == "Compute":
                state = self.compute_state
            elif source_text == "NPZ Viewer":
                state = self.viewer_state
            elif source_text.startswith("Custom"):
                state = self.custom_state
            else:
                # Fallback to viewer state
                state = self.viewer_state

            if state.bounds_min is None or state.bounds_max is None:
                return None, f"No bounds available for {source_text} data."

            if state.result is None and state.profile is None and state.bracing is None:
                return None, f"No {source_text} data loaded."

            grid = self._ensure_state_grid(state)
            if grid is None:
                return None, f"Missing grid data for {source_text}."

            nx, ny, X, Y = grid
            
            # Get mesh parameters
            total_height = float(self._mesh_height_slider.double_value)
            z_interp = int(self._mesh_interp_slider.int_value)
            
            # Get individual iso overrides
            iso_override_result = float(self._mesh_iso_result_slider.double_value)
            iso_override_profile = float(self._mesh_iso_profile_slider.double_value)
            iso_override_bracing = float(self._mesh_iso_bracing_slider.double_value)
            
            # Get smoothing parameters
            smooth_method_idx = self._mesh_smooth_method_combo.selected_index
            smooth_methods = ["None", "Laplacian", "Taubin", "Laplacian + Taubin"]
            smooth_method = smooth_methods[smooth_method_idx]
            smooth_iterations = int(self._mesh_smooth_slider.int_value)
            
            # Build parameter hash for cache checking
            params = {
                "total_height": total_height,
                "z_interp": z_interp,
                "iso_override_result": iso_override_result,
                "iso_override_profile": iso_override_profile,
                "iso_override_bracing": iso_override_bracing,
                "smooth_method": smooth_method,
                "smooth_iterations": smooth_iterations,
            }
            
            # Check if we can use cached meshes
            use_cache = (state.mesh_params_cache == params)
            
            # Calculate iso levels with individual overrides
            iso_r = state.iso_p_base + iso_override_result
            iso_p = state.iso_p_base + iso_override_profile
            iso_b = state.iso_b_base + iso_override_bracing
            
            if source_text == "Compute":
                iso_p += self._profile_offset_slider.double_value
                iso_b += self._bracing_offset_slider.double_value

            result_mesh = None
            profile_mesh = None
            bracing_mesh = None
            details = []

            # Generate Result Mesh
            if self._mesh_chk_result.checked:
                if state.result is None:
                    details.append("result: missing")
                elif use_cache and state.result_mesh_cache is not None:
                    result_mesh = state.result_mesh_cache
                    details.append("result: cached")
                else:
                    try:
                        result_mesh = self._build_mesh_from_fields(
                            state.result,
                            iso_r,
                            state.bounds_min,
                            state.bounds_max,
                            nx,
                            ny,
                            total_height,
                            z_interp,
                            smooth_method,
                            smooth_iterations,
                        )
                        state.result_mesh_cache = result_mesh
                        if result_mesh:
                            details.append(f"result: {len(result_mesh.vertices)} verts")
                    except Exception as e:
                        details.append(f"result: {e}")

            # Generate Profile Mesh
            if self._mesh_chk_profile.checked:
                if state.profile is None:
                    details.append("profile: missing")
                elif use_cache and state.profile_mesh_cache is not None:
                    profile_mesh = state.profile_mesh_cache
                    details.append("profile: cached")
                else:
                    try:
                        profile_mesh = self._build_mesh_from_fields(
                            state.profile,
                            iso_p,
                            state.bounds_min,
                            state.bounds_max,
                            nx,
                            ny,
                            total_height,
                            z_interp,
                            smooth_method,
                            smooth_iterations,
                        )
                        state.profile_mesh_cache = profile_mesh
                        if profile_mesh:
                            details.append(f"profile: {len(profile_mesh.vertices)} verts")
                        else:
                            details.append(f"profile: returned None (iso={iso_p:.3f}, slices={state.profile.shape[0] if state.profile is not None else 'N/A'})")
                    except Exception as e:
                        details.append(f"profile: ERROR {e}")

            # Generate Bracing Mesh
            if self._mesh_chk_bracing.checked:
                bracing_fields = state.bracing_clean if state.bracing_clean is not None else state.bracing
                if bracing_fields is None:
                    details.append("bracing: missing")
                elif use_cache and state.bracing_mesh_cache is not None:
                    bracing_mesh = state.bracing_mesh_cache
                    details.append("bracing: cached")
                else:
                    try:
                        bracing_mesh = self._build_mesh_from_fields(
                            bracing_fields,
                            iso_b,
                            state.bounds_min,
                            state.bounds_max,
                            nx,
                            ny,
                            total_height,
                            z_interp,
                            smooth_method,
                            smooth_iterations,
                        )
                        state.bracing_mesh_cache = bracing_mesh
                        if bracing_mesh:
                            details.append(f"bracing: {len(bracing_mesh.vertices)} verts")
                        else:
                            details.append(f"bracing: returned None (iso={iso_b:.3f}, slices={bracing_fields.shape[0] if bracing_fields is not None else 'N/A'})")
                    except Exception as e:
                        details.append(f"bracing: ERROR {e}")

            # Update cache params
            state.mesh_params_cache = params

            return (result_mesh, profile_mesh, bracing_mesh, details), None
        
        def on_done(result_tuple):
            """Main thread callback after background computation."""
            self._btn_generate_mesh.enabled = True
            
            if result_tuple is None or len(result_tuple) != 2:
                self._status.text = "Mesh generation failed unexpectedly."
                return
            
            result, error = result_tuple
            
            if error:
                self._status.text = error
                return
            
            result_mesh, profile_mesh, bracing_mesh, details = result
            
            if result_mesh is None and profile_mesh is None and bracing_mesh is None:
                self._clear_mesh_geometries()
                if details:
                    self._status.text = "Mesh generation: " + "; ".join(details)
                else:
                    self._status.text = "Mesh generation produced no geometry."
                return

            self._apply_mesh_geometries(result_mesh, profile_mesh, bracing_mesh, fit_camera=False)
            self._status.text = "Mesh generated: " + " | ".join(details)
        
        # Run in background thread
        import threading
        def run_and_post():
            result = compute_meshes()
            gui.Application.instance.post_to_main_thread(self._window, lambda: on_done(result))
        
        thread = threading.Thread(target=run_and_post, daemon=True)
        thread.start()
    
    def _on_export_mesh(self):
        """Export generated meshes to OBJ files."""
        # Get export location from mesh tab (custom location or default)
        export_location = self._mesh_export_location.text_value.strip()
        if not export_location:
            export_location = "./output/mesh"  # Default fallback
        
        output_dir = self._resolve_path(export_location)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self._status.text = f"Cannot create export dir: {e}"
                return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported = []
        errors = []
        
        def save_mesh_obj(mesh, name):
            if mesh is None or len(mesh.vertices) == 0:
                return
            filename = output_dir / f"{name}_mesh_{timestamp}.obj"
            try:
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                with open(filename, "w") as f:
                    f.write(f"# Narnia {name.title()} Mesh\n")
                    for v in verts:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    for face in faces:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                exported.append(f"{name}: {filename.name}")
            except Exception as e:
                errors.append(f"{name}: {e}")
        
        # Export each mesh that exists
        save_mesh_obj(self._result_mesh, "result")
        save_mesh_obj(self._profile_mesh, "profile")
        save_mesh_obj(self._bracing_mesh, "bracing")
        
        if not exported and not errors:
            self._status.text = "No meshes to export. Generate meshes first."
            return
        
        msg_parts = []
        if exported:
            msg_parts.append("Exported: " + ", ".join(exported))
        if errors:
            msg_parts.append("Errors: " + ", ".join(errors))
        self._status.text = " | ".join(msg_parts)

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
    
    def _refresh_mesh_from_cache(self):
        """When switching to mesh tab, display cached meshes from current source if available."""
        source_text = self._mesh_source_combo.get_item(self._mesh_source_combo.selected_index)
        
        # Determine which state to use
        if source_text == "Compute":
            state = self.compute_state
        elif source_text == "NPZ Viewer":
            state = self.viewer_state
        elif source_text.startswith("Custom"):
            state = self.custom_state
        else:
            return
        
        # Display cached meshes if available
        if state.result_mesh_cache is not None or state.profile_mesh_cache is not None or state.bracing_mesh_cache is not None:
            self._apply_mesh_geometries(
                state.result_mesh_cache,
                state.profile_mesh_cache,
                state.bracing_mesh_cache,
                fit_camera=False
            )
            self._status.text = f"Showing cached meshes from {source_text}"


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


