import json
import os
from pathlib import Path
import numpy as np
import core
import vis_utils as vut

# --- CONFIGURATION ---
GENERATE_BRACING = False
SAVE_RESULTS = True
EXTRACT_CURVES = False # Set to True if you need curves exported
LAUNCH_VIEWER = True

# Bracing Parameters
NUM_CENTROIDS_START = 3
NUM_CENTROIDS_END = 4  # Interpolate from 3 to 8
OP_MODE = "difference" # difference, union, intersection

# Bracing Method
# - "voronoi": static ridge-style bracing (simple)
# - "keyfield_blend": manual keyframes with interpolation
# - "adaptive": shape-adaptive (auto k based on area) - RECOMMENDED
BRACING_METHOD = "adaptive"

# Adaptive Bracing Parameters (for BRACING_METHOD="adaptive")
ADAPTIVE_AREA_PER_SEED = 1500.0  # Target pixels per Voronoi cell
ADAPTIVE_K_MIN = 2               # Minimum centroids per slice
ADAPTIVE_K_MAX = 12              # Maximum centroids per slice
ADAPTIVE_SMOOTH_SIGMA = 1.5      # Z-axis smoothing (slices)
ADAPTIVE_RAMP_SLICES = 3         # Weight ramp for new centroids

# Paths
PROFILE_JSON_PATH = Path("./alice_result/251120/ext/waveStackFields.json")
BRACING_JSON_PATH = Path("./alice_result/251120/bracing/waveStackFields.json") # Used if GENERATE_BRACING=False
OUTPUT_DIR = Path("./output")
OUTPUT_FILENAME = "processed_sdf_results.npz"

# Threading context (from previous main.py)
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(limits=None, user_api=None):
        yield

def main():
    vut.print_system_diagnostics()
    
    # 1. Load Profile Data
    print(f"Loading profile from: {PROFILE_JSON_PATH}")
    if not PROFILE_JSON_PATH.exists():
        print(f"Error: Profile file not found at {PROFILE_JSON_PATH}")
        return

    with open(PROFILE_JSON_PATH, "r") as f:
        data_profile = json.load(f)

    iso_level_profile, slice_count_profile, bounds_max, bounds_min = core.meta_data_info(data_profile)
    
    print("Stacking profile fields...")
    with threadpool_limits(limits=1):
        profile_fields_2d, _ = core.stack_scalar_fields(data_profile)

    # Prefer explicit nx/ny if present; otherwise fall back (legacy square-grid inference).
    _, nx, ny = core.infer_grid_from_scalar_fields(profile_fields_2d)
    
    # 2. Generate or Load Bracing
    if GENERATE_BRACING:
        print(f"Generating bracing (Method: {BRACING_METHOD})...")
        
        with threadpool_limits(limits=1):
            if BRACING_METHOD == "adaptive":
                # Shape-adaptive: auto-determines k per slice based on area
                print(f"  Area per seed: {ADAPTIVE_AREA_PER_SEED} px²")
                print(f"  K range: [{ADAPTIVE_K_MIN}, {ADAPTIVE_K_MAX}]")
                bracing_fields_2d = core.generate_bracing_adaptive(
                    profile_fields_2d,
                    iso_level=iso_level_profile,
                    nx=nx,
                    ny=ny,
                    area_per_seed=ADAPTIVE_AREA_PER_SEED,
                    k_min=ADAPTIVE_K_MIN,
                    k_max=ADAPTIVE_K_MAX,
                    seed=42,
                    smooth_sigma=ADAPTIVE_SMOOTH_SIGMA,
                    ramp_slices=ADAPTIVE_RAMP_SLICES
                )
            
            elif BRACING_METHOD == "keyfield_blend":
                # Manual keyframes (original implementation)
                print(f"  Centroids: {NUM_CENTROIDS_START} -> {NUM_CENTROIDS_END}")
                bracing_fields_2d = core.generate_bracing_keyfield_blend(
                    profile_fields_2d,
                    iso_level=iso_level_profile,
                    nx=nx,
                    ny=ny,
                    keys_config=[(0, NUM_CENTROIDS_START), (profile_fields_2d.shape[0]-1, NUM_CENTROIDS_END)],
                    smooth=0.5,
                    seed=42
                )
            
            else:  # "voronoi" - simple static
                print(f"  Static k={NUM_CENTROIDS_START} centroids")
                bracing_fields_2d = core.generate_bracing_static(
                    profile_fields_2d,
                    iso_level=iso_level_profile,
                    nx=nx,
                    ny=ny,
                    k=NUM_CENTROIDS_START,
                    seed=42
                )
        
        iso_level_bracing = 0.0
    else:
        print(f"Loading bracing from: {BRACING_JSON_PATH}")
        if not BRACING_JSON_PATH.exists():
            print("Error: Bracing file not found.")
            return
        with open(BRACING_JSON_PATH, "r") as f:
            data_bracing = json.load(f)
        iso_level_bracing, _, _, _ = core.meta_data_info(data_bracing)
        bracing_fields_2d, _ = core.stack_scalar_fields(data_bracing)

    # 3. Boolean Operation
    print(f"Performing boolean operation ({OP_MODE})...")
    result_fields = core.compute_sf_operation(
        profile_fields_2d,
        bracing_fields_2d,
        iso_level_A=iso_level_profile,
        iso_level_B=iso_level_bracing,
        mode=OP_MODE
    )
    
    # 4. Save Results
    if SAVE_RESULTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / OUTPUT_FILENAME
        print(f"Saving results to {out_path}...")
        save_dict = {
            "result_fields": result_fields,
            "iso_level": iso_level_profile,  # Result usually takes profile iso level if difference
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "nx": nx,
            "ny": ny,
            "slice_count": slice_count_profile,
            "profile_fields": profile_fields_2d,
            "bracing_fields": bracing_fields_2d,
        }
        if GENERATE_BRACING and BRACING_METHOD.lower() == "cell_wall":
            pass # Removed legacy save fields

        np.savez(out_path, **save_dict)

    # 5. Visualization
    if LAUNCH_VIEWER and not vut.HEADLESS_MODE:
        print(f"Launching viewer on monitor [{vut.PREFERRED_MONITOR_INDEX}]...")
        import narnia_vis
        narnia_vis.run_app_from_data(
            result_fields, 
            bounds_min, 
            bounds_max, 
            iso_level=iso_level_profile, 
            profile_fields=profile_fields_2d, 
            iso_p=iso_level_profile,
            monitor_index=vut.PREFERRED_MONITOR_INDEX
        )
    elif LAUNCH_VIEWER and vut.HEADLESS_MODE:
        print("Headless mode enabled. Skipping viewer.")

if __name__ == "__main__":
    main()
