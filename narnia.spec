# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec File for Narnia Viewer

This spec file packages the Narnia SDF/Curve Viewer into a portable Windows executable.

Build:
    pyinstaller narnia.spec

Output:
    dist/NarniaViewer/NarniaViewer.exe  (one-folder mode)
    
To switch to one-file mode:
    Change EXE() section to include all data in the exe argument list
"""

import os
from pathlib import Path

block_cipher = None

# ============================
# Analysis - Collect all code, imports, and data
# ============================
import os
import open3d

# Get Open3D installation path for bundling resources
open3d_path = os.path.dirname(open3d.__file__)
open3d_resources = os.path.join(open3d_path, 'resources')

a = Analysis(
    ['run_narnia.py'],  # Clean entry point (not main.py)
    pathex=[],
    binaries=[],
    datas=[
        # Bundle sample data files
        ('alice_result', 'alice_result'),
        
        # Include core and gui packages explicitly
        ('core', 'core'),
        ('gui', 'gui'),
        
        # Include other module files
        ('narnia_vis.py', '.'),
        ('vis_utils.py', '.'),
        ('MeshFromNPZ.py', '.'),  # Optional: Include CLI tool
        
        # CRITICAL: Bundle Open3D resources (shaders, materials, fonts)
        (open3d_resources, 'open3d/resources'),
    ],
    hiddenimports=[
        # Core scientific stack
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.lib',
        'numpy.lib.format',
        
        # SciPy and submodules
        'scipy',
        'scipy.ndimage',
        'scipy.spatial',
        'scipy.spatial.distance',
        'scipy.spatial.cKDTree',
        'scipy.ndimage.morphology',
        'scipy.ndimage.gaussian_filter1d',
        'scipy.ndimage._nd_image',
        
        # Scikit-learn
        'sklearn',
        'sklearn.cluster',
        'sklearn.cluster._kmeans',
        'sklearn.neighbors',
        'sklearn.neighbors._kd_tree',
        'sklearn.utils._cython_blas',
        
        # Scikit-image (marching cubes)
        'skimage',
        'skimage.measure',
        'skimage.measure._marching_cubes_lewiner',
        
        # Contour extraction
        'contourpy',
        'contourpy._contourpy',
        
        # Open3D and visualization
        'open3d',
        'open3d.visualization',
        'open3d.visualization.gui',
        'open3d.visualization.rendering',
        'open3d.cpu',
        'open3d.cpu.pybind',
        
        # Threading control (optional but recommended)
        'threadpoolctl',
        
        # Windows-specific
        'ctypes',
        'ctypes.wintypes',
    ],
    hookspath=['hooks'],  # Use our custom runtime hook
    hooksconfig={},
    runtime_hooks=['hooks/hook-thread-env.py'],  # Set env vars before imports
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',  # Only used in notebook_test/
        'notebook_test',
        'examples',
        'tests',
        'tkinter',
        'PyQt5',
        'PySide2',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ============================
# PYZ - Python Archive
# ============================
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# ============================
# EXE - Executable Configuration
# ============================
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # One-folder mode (faster startup)
    name='NarniaViewer',
    debug=False,  # Set True for debugging
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress with UPX (reduces size ~30%)
    console=True,  # Show console window (useful for debugging; set False for release)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',  # Optional: Add custom icon
)

# ============================
# COLLECT - Bundle into dist folder
# ============================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NarniaViewer'
)

# ============================
# Notes for Future Modifications
# ============================
# 1. One-file mode: Move a.binaries, a.zipfiles, a.datas into EXE() and remove COLLECT()
# 2. Add icon: Place icon.ico in root and uncomment icon= line
# 3. Hide console: Set console=False in EXE() for windowed app
# 4. Reduce size: Remove alice_result from datas if sample data not needed
# 5. Debug imports: Run with `pyinstaller --log-level DEBUG narnia.spec`
