import json
from pathlib import Path
import numpy as np
import core

def main(
    bracing_json_path="./alice_result/251120/bracing/waveStackFields.json",
    profile_json_path="./alice_result/251120/ext/waveStackFields.json",
    data_bracing=None,
    data_profile=None,
    output_dir="./output",
    op_mode="difference",
    iso_level_result=0.0,
    iso_offset_profile=0.0,
    iso_offset_bracing=0.0,
    save_results=True,
    extract_curves=True,
    generate_bracing=False,
    num_centroids=3
):
    # 1. Load Data (Optional)
    if data_profile is None:
        profile_json_path = Path(profile_json_path)
        print(f"Loading profile from: {profile_json_path}")
        if not profile_json_path.exists():
            print(f"Error: Profile file not found at {profile_json_path}")
            return None
        with open(profile_json_path, "r") as f:
            data_profile = json.load(f)

    if not generate_bracing and data_bracing is None:
        bracing_json_path = Path(bracing_json_path)
        print(f"Loading bracing from: {bracing_json_path}")
        if not bracing_json_path.exists():
            print(f"Error: Bracing file not found at {bracing_json_path}")
            return None
        with open(bracing_json_path, "r") as f:
            data_bracing = json.load(f)

    # 2. Parse Profile
    iso_level_profile, slice_count_profile, bounds_max_profile, bounds_min_profile = core.meta_data_info(data_profile)
    profile_fields_2d, _ = core.stack_scalar_fields(data_profile)
    iso_level_profile += iso_offset_profile

    # 3. Bracing Logic
    if generate_bracing:
        print(f"Generating bracing from centroids (k={num_centroids})...")
        num_fields, nx, ny = core.infer_grid_from_scalar_fields(profile_fields_2d)
        bracing_fields_2d = np.zeros_like(profile_fields_2d)
        prev_centroids = None
        
        for i in range(num_fields):
            slice_2d = profile_fields_2d[i].reshape((ny, nx))
            mask = core.get_profile_mask(slice_2d, iso_level=iso_level_profile)
            
            centroids = core.generate_centroids(mask, k=num_centroids, prev_centroids=prev_centroids)
            centroids = core.constrain_centroids_to_mask(centroids, mask)
            
            voronoi_sdf_flat = core.compute_voronoi_sdf((ny, nx), centroids)
            bracing_fields_2d[i] = voronoi_sdf_flat.ravel()
            prev_centroids = centroids
            
            if i % 10 == 0:
                print(f"Generated bracing for slice {i}/{num_fields}")
        
        iso_level_bracing = 0.0
    else:
        iso_level_bracing, slice_count_bracing, bounds_max_bracing, bounds_min_bracing = core.meta_data_info(data_bracing)
        bracing_fields_2d, _ = core.stack_scalar_fields(data_bracing)

    # Apply offsets
    iso_level_bracing += iso_offset_bracing

    # 4. Validation
    assert bracing_fields_2d.shape == profile_fields_2d.shape, "Shape mismatch between bracing and profile fields!"
    
    # 5. Boolean Operation
    print(f"Performing boolean operation ({op_mode})...")
    result_fields_2d = core.compute_sf_operation(
        profile_fields_2d,
        bracing_fields_2d,
        iso_level_A=iso_level_profile,
        iso_level_B=iso_level_bracing,
        mode=op_mode
    )

    all_curves = None
    if extract_curves:
        # 6. Iso-curve Extraction
        print("Extracting iso-curves...")
        num_fields, nx, ny = core.infer_grid_from_scalar_fields(result_fields_2d)
        
        # Create coordinate grid
        x = np.linspace(bounds_min_profile[0], bounds_max_profile[0], nx)
        y = np.linspace(bounds_min_profile[1], bounds_max_profile[1], ny)
        X, Y = np.meshgrid(x, y, indexing="xy")

        all_curves = []

        for i in range(num_fields):
            slice_2d = result_fields_2d[i].reshape((ny, nx))
            curves = core.iso_curves_for_slice_2d(slice_2d, iso_level_result, X, Y)
            all_curves.append(curves)
            if i % 10 == 0:
                print(f"Processed slice {i}/{num_fields}")

    # 7. Save Results (Optional)
    if save_results:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "processed_sdf_results.npz"
        np.savez(output_path, 
                 result_fields=result_fields_2d, 
                 iso_level=iso_level_result)
        print(f"Successfully saved results to {output_path}")

    if all_curves is not None:
        print(f"Total curves extracted: {sum(len(c) for c in all_curves)}")
    
    return result_fields_2d, all_curves, profile_fields_2d, iso_level_profile

if __name__ == "__main__":
    # Define variables here to modify before running
    BRACING_PATH = "./alice_result/251120/bracing/waveStackFields.json"
    PROFILE_PATH = "./alice_result/251120/ext/waveStackFields.json"
    OUTPUT = "./output"
    MODE = "difference"
    ISO_LEVEL = 0.0
    OFFSET_PROFILE = 0.0
    OFFSET_BRACING = 0.0
    SAVE_RESULTS = False
    EXTRACT_CURVES = False
    LAUNCH_VIEWER = True
    
    # New parameters for centroid-based bracing
    GENERATE_BRACING = True
    NUM_CENTROIDS = 5

    res = main(
        bracing_json_path=BRACING_PATH,
        profile_json_path=PROFILE_PATH,
        output_dir=OUTPUT,
        op_mode=MODE,
        iso_level_result=ISO_LEVEL,
        iso_offset_profile=OFFSET_PROFILE,
        iso_offset_bracing=OFFSET_BRACING,
        save_results=SAVE_RESULTS,
        extract_curves=EXTRACT_CURVES,
        generate_bracing=GENERATE_BRACING,
        num_centroids=NUM_CENTROIDS
    )

    fields_2d_result, curves, profile_fields, iso_p = res

    if LAUNCH_VIEWER:
        import narnia_vis
        with open(PROFILE_PATH, "r") as f:
            meta = json.load(f)
        _, _, b_max, b_min = core.meta_data_info(meta)
        narnia_vis.run_app_from_data(
            fields_2d_result,
            b_min,
            b_max,
            iso_level=ISO_LEVEL,
            profile_fields=profile_fields,
            iso_p=iso_p,
            output_dir=OUTPUT,
        )


   
