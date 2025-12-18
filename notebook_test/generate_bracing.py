import os
# Fix for potential OpenMP runtime conflict/crash on Windows
os.environ["OMP_NUM_THREADS"] = "1"

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sf_tools as sft
import sf_bracing as sfb

def main():
    # Paths
    base_path = Path("c:/Users/Daxler.Zou/ZahaWork/Project/4359_exh_mocaup/251013_LatentSDF/Model/SDF_Stack")
    profile_json_path = base_path / "alice_result/251120/ext/waveStackFields.json"
    
    print(f"Loading profile from: {profile_json_path}")
    
    with open(profile_json_path, "r") as f:
        data_profile = json.load(f)
        
    # Parse metadata
    iso_level_profile, slice_count_profile, _, _ = sft.meta_data_info(data_profile)
    
    # Stack fields
    profile_scalar_fields_2d, _ = sft.stack_scalar_fields(data_profile)
    
    # Parameters
    k_centroids = 3
    
    # Storage
    centroids_list = []
    bracing_fields_list = []
    prev_centroids = None
    
    print("Generating bracing fields...")
    
    _, nx, ny = sft.infer_grid_from_scalar_fields(profile_scalar_fields_2d)
    
    for i in range(slice_count_profile):
        field = profile_scalar_fields_2d[i].reshape((ny, nx))
        
        # 1. Mask
        mask = sfb.get_profile_mask(field, iso_level=iso_level_profile)
        
        # 2. Centroids
        centroids = sfb.generate_centroids(mask, k=k_centroids, prev_centroids=prev_centroids)
        centroids_list.append(centroids)
        prev_centroids = centroids
        
        # 3. Voronoi SDF
        voronoi_field = sfb.compute_voronoi_sdf((ny, nx), centroids)
        bracing_fields_list.append(voronoi_field.ravel())
        
        if i % 10 == 0:
            print(f"Processed slice {i}/{slice_count_profile}")
            
    bracing_scalar_fields_generated = np.array(bracing_fields_list)
    centroids_array = np.array(centroids_list)
    
    print("Generation complete.")
    
    # Save results
    output_file = base_path / "generated_bracing_fields.npz"
    np.savez(output_file, bracing=bracing_scalar_fields_generated, centroids=centroids_array)
    print(f"Saved results to {output_file}")
    
    # Generate Preview Plot (Middle Slice)
    idx = slice_count_profile // 2
    profile_field = profile_scalar_fields_2d[idx].reshape((ny, nx))
    bracing_field = bracing_scalar_fields_generated[idx].reshape((ny, nx))
    centroids = centroids_list[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Profile + Centroids
    ax = axes[0]
    ax.imshow(profile_field, origin="lower", cmap="Greys")
    ax.contour(profile_field, levels=[iso_level_profile], colors="blue", linewidths=1)
    ax.scatter(centroids[:, 1], centroids[:, 0], c="red", s=50, marker="x", label="Centroids")
    ax.set_title(f"Slice {idx}: Profile & Centroids")
    ax.legend()
    
    # Plot 2: Combined Curves
    ax = axes[1]
    # Background: Profile Field
    ax.imshow(profile_field, origin="lower", cmap="Greys", alpha=0.5)
    
    # Profile Curve (Blue)
    ax.contour(profile_field, levels=[iso_level_profile], colors="blue", linewidths=2, label="Profile")
    
    # Bracing Curve (Red) - Voronoi Boundaries
    # We mask the bracing field to only show inside the profile for cleaner visualization
    mask = sfb.get_profile_mask(profile_field, iso_level=iso_level_profile)
    masked_bracing = np.where(mask, bracing_field, np.nan)
    
    ax.contour(masked_bracing, levels=[0], colors="red", linewidths=2, label="Bracing")
    
    ax.set_title(f"Slice {idx}: Combined Curves")
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="blue", lw=2),
                    Line2D([0], [0], color="red", lw=2)]
    ax.legend(custom_lines, ['Profile', 'Bracing'])
    
    preview_path = base_path / "bracing_preview_combined.png"
    plt.savefig(preview_path)
    print(f"Saved preview to {preview_path}")

if __name__ == "__main__":
    main()
