import json
import os
from pathlib import Path
import numpy as np
import core
import vis_utils as vut

# --- CONFIGURATION ---
GENERATE_BRACING = True
SAVE_RESULTS = True
EXTRACT_CURVES = False # Set to True if you need curves exported
LAUNCH_VIEWER = True

# Bracing Parameters
NUM_CENTROIDS_START = 3
NUM_CENTROIDS_END = 8  # Interpolate from 3 to 8
OP_MODE = "difference" # difference, union, intersection

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
    
    # 2. Generate or Load Bracing
    if GENERATE_BRACING:
        print(f"Generating bracing (Centroids: {NUM_CENTROIDS_START} -> {NUM_CENTROIDS_END})...")
        
        # Use the new interpolated pipeline
        with threadpool_limits(limits=1):
            bracing_fields_2d, tracks, weights = core.generate_interpolated_bracing_fields(
                profile_fields_2d, 
                k_min=NUM_CENTROIDS_START, 
                k_max=NUM_CENTROIDS_END,
                iso_level=iso_level_profile,
                ramp=5, # Ramp over 5 slices
                smooth_sigma=2.0
            )
        
        iso_level_bracing = 0.0 # Voronoi SDF is distance based, 0 is the boundary
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
        np.savez(
            out_path, 
            result_fields=result_fields, 
            iso_level=iso_level_profile, # Result usually takes profile iso level if difference
            bounds_min=bounds_min,
            bounds_max=bounds_max
        )

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
