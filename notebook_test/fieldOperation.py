import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sf_tools as sft




# set json file path
bracing_json_path = Path("./alice_result/251120/bracing/waveStackFields.json")  
profile_json_path = Path("./alice_result/251120/ext/waveStackFields.json")



# SDF Parse
with open(bracing_json_path, "r") as f:
    data_bracing = json.load(f)
with open(profile_json_path, "r") as f:
    data_profile = json.load(f)
# Inspect the top-level structure
print("data_bracing Top-level type:", type(data_bracing))
if isinstance(data_bracing, dict):
    print("Top-level keys:", list(data_bracing.keys()))
else:
    print("Non-dict JSON root (e.g. list)\n", data_bracing)

print("\n")

# inspect profiles structure
print("data_profile Top-level type:", type(data_profile))
if isinstance(data_profile, dict):
    print("Top-level keys:", list(data_profile.keys()))
else:
    print("Non-dict JSON root (e.g. list)\n", data_profile)




# parse scalar field of bracing

print("\n---Parsing bracing scalar fields---\n")

iso_level_bracing, slice_count_bracing, bounds_max_bracing, bounds_min_bracing = sft.meta_data_info(data_bracing)
sft.stack_scalar_fields(data_bracing)
bracing_scalar_fields_2d, bracing_scalar_field_keys = sft.stack_scalar_fields(data_bracing)


# parse scalar field of profiles

print("\n---Parsing bracing scalar fields---\n")

iso_level_profile, slice_count_profile, bounds_max_profile, bounds_min_profile = sft.meta_data_info(data_profile)


sft.stack_scalar_fields(data_profile)
profile_scalar_fields_2d, profile_scalar_field_keys = sft.stack_scalar_fields(data_profile)

# match the shape of two fields
assert bracing_scalar_fields_2d.shape == profile_scalar_fields_2d.shape, "Scalar fields from bracing and profile do not match in shape!"
assert slice_count_bracing == slice_count_profile, "Slice counts do not match!"
assert bounds_max_bracing == bounds_max_profile, "Bounds max do not match!"
assert bounds_min_bracing == bounds_min_profile, "Bounds min do not match!"

# warn if iso-levels do not match
if iso_level_bracing != iso_level_profile:
    print("\nWARNING: Iso levels from bracing and profile do not match! use customized iso-level.")





# Visualize the scalar fields as an image
sft.plot_scalar_overview(bracing_scalar_fields_2d, field_name="Bracing")
sft.plot_scalar_overview(profile_scalar_fields_2d, field_name="Profile")

# infer_grid_from_scalar_fields using profile fields
number_of_fields, nx, ny = sft.infer_grid_from_scalar_fields(profile_scalar_fields_2d)

# bounds of the grid using profile fields
bounds_min = bounds_min_profile
bounds_max = bounds_max_profile

# iso level using profile fields
iso_level_result = 0.0

# world xy grid for one slice
x = np.linspace(bounds_min[0], bounds_max[0], nx)
y = np.linspace(bounds_min[1], bounds_max[1], ny)
X, Y = np.meshgrid(x, y, indexing="xy")  # X,Y both (ny, nx)


# run for all slices
iso_curves_per_slice_2d = []
for i in range(number_of_fields):
    slice_2d = sft.result_scalar_fields_2d[i].reshape((ny, nx))
    
    curves_2d = sft.iso_curves_for_slice_2d(slice_2d, iso_level_result, X, Y)
    
    iso_curves_per_slice_2d.append(curves_2d)

print(f"Computed iso-curves for {number_of_fields} slices.")
print(f"Slice 0 has {len(iso_curves_per_slice_2d[0])} curves.")