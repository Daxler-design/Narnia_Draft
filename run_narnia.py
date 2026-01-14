"""
Narnia Viewer - EXE Entry Point

This is a clean entry point for PyInstaller executable compilation.
It wraps main.py without modification, keeping original code untouched.

Usage:
    python run_narnia.py              # Run GUI normally
    pyinstaller narnia.spec           # Build exe
"""
import os
import sys

# Set thread limiting BEFORE importing numpy/scipy to prevent GUI freezes
# These must be set at startup before any scientific library imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

# Optional: Force software rendering if GPU compatibility issues
# Uncomment if exe crashes on certain GPUs:
# os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

def fix_open3d_resources():
    """Fix Open3D resource path for PyInstaller bundled exe."""
    # Check if running as PyInstaller bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as exe - sys._MEIPASS is PyInstaller's temp extraction folder
        import open3d as o3d
        resource_path = os.path.join(sys._MEIPASS, 'open3d', 'resources')
        
        # Check if resources exist
        if os.path.exists(resource_path):
            # Set Open3D resource path
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
            print(f"[PyInstaller] Setting Open3D resource path: {resource_path}")
            # Open3D will use this path automatically via sys._MEIPASS
        else:
            print(f"[PyInstaller Warning] Open3D resources not found at: {resource_path}")
            print("[PyInstaller Warning] GUI rendering may fail")

def main():
    """Launch the Narnia Viewer application."""
    # Fix Open3D resource paths for exe mode
    fix_open3d_resources()
    
    # Import main application (after env vars are set)
    import main as narnia_main
    
    # Run the application
    narnia_main.main()


if __name__ == "__main__":
    main()
