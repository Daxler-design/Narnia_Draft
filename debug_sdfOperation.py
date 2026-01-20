


import numpy as np
import core
import debug_utils as debug
import json
from pathlib import Path







# read profile fields
PROFILE_JSON_PATH = Path("./alice_result/251120/ext/waveStackFields.json")
with open(PROFILE_JSON_PATH, "r") as f:
    data_profile = json.load(f)

iso_level_profile, slice_count_profile, bounds_max, bounds_min = core.meta_data_info(data_profile)
print(f"Profile iso level: {iso_level_profile}, slice count: {slice_count_profile}")

profile_fields_2d, _ = core.stack_scalar_fields(data_profile)
_, nx, ny = core.infer_grid_from_scalar_fields(profile_fields_2d)


# bracing field generation just using static methods
bracing_fields_2d =core.generate_bracing_static(
    profile_fields_2d,
    0.1,
    nx=nx,
    ny=ny,
    k=5,
    seed=42,
    sigma=1.0,
    normalize_range=True
)

result_fields = []


# check fileds info, hologram
for i in range(len(profile_fields_2d)):
    pf = profile_fields_2d[i]
    bf = bracing_fields_2d[i]
    rf = core.compute_sf_operation(
        pf,
        bf,
        iso_level_A=iso_level_profile,
        iso_level_B=0.0,
        mode="difference"
    )
    result_fields.append(rf)


    if i == 10:
        debug._save_debug_boolean_show(pf, bf, rf, nx,ny, slice_idx=i, tag="boolean_debug")