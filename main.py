
import json
from pathlib import Path
# vis_utils sets up environment variables (threading, etc) on import
import vis_utils as vut 
import numpy as np
import core

# Try to import threadpoolctl, otherwise use a dummy context manager
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(limits=None, user_api=None):
        yield

if __name__ == "__main__":
    vut.print_system_diagnostics()
    
    # Import narnia_vis here to avoid early GL context creation
    import narnia_vis

    output_path = Path("./output/processed_sdf_results.npz")
    profile_json_path = Path("./alice_result/251120/ext/waveStackFields.json")

    if output_path.exists() and profile_json_path.exists():
        print("Loading data...")
        data = np.load(output_path, allow_pickle=True)
        result_fields = data["result_fields"]
        iso_level = float(data["iso_level"])
        
        with open(profile_json_path, "r") as f:
            meta = json.load(f)
        
        iso_p, _, bounds_max, bounds_min = core.meta_data_info(meta)
        
        print("Stacking scalar fields...")
        with threadpool_limits(limits=1):
            profile_fields, _ = core.stack_scalar_fields(meta)
        del meta
        
        print(f"Loaded fields: {result_fields.shape}")
        
        if vut.HEADLESS_MODE:
            print("Headless Mode: Skipping 3D Viewer. Results are ready in output folder.")
        else:
            print(f"Launching viewer on monitor [{vut.PREFERRED_MONITOR_INDEX}]...")
            narnia_vis.run_app_from_data(
                result_fields, 
                bounds_min, 
                bounds_max, 
                iso_level=iso_level, 
                profile_fields=profile_fields, 
                iso_p=iso_p,
                monitor_index=vut.PREFERRED_MONITOR_INDEX
            )
    else:
        if vut.HEADLESS_MODE:
            print("Headless Mode: Data not found. Cannot compute without GUI.")
        else:
            narnia_vis.run_app(monitor_index=vut.PREFERRED_MONITOR_INDEX)
   
