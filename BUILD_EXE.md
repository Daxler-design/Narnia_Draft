# Building Narnia Viewer as Portable EXE

Complete guide for compiling the Narnia SDF/Curve Viewer into a Windows portable executable.

## Quick Start

```bash
# 1. Install PyInstaller
pip install pyinstaller

# 2. Build the exe
build_exe.bat

# 3. Test the exe
cd dist\NarniaViewer
NarniaViewer.exe
```

---

## Prerequisites

### 1. Install PyInstaller

```bash
conda activate narnia
pip install pyinstaller
```

### 2. Verify Dependencies

Ensure all required packages are installed:

```bash
pip install -r requirements.txt
```

**Note**: The missing `scikit-image` dependency has been added to `requirements.txt`.

---

## Build Methods

### Method 1: Automated Build Script (Recommended)

```bash
# Build exe
build_exe.bat

# Clean build artifacts
build_exe.bat clean
```

**Output**: `dist\NarniaViewer\NarniaViewer.exe` + all dependencies (~600-800 MB folder)

### Method 2: Manual PyInstaller

```bash
# One-folder mode (recommended - faster startup)
pyinstaller narnia.spec

# Debug mode (shows import errors)
pyinstaller --log-level DEBUG narnia.spec

# Clean rebuild
pyinstaller --clean narnia.spec
```

---

## File Structure (Build System)

```
Narnia_Draft/
├── run_narnia.py          # EXE entry point (wraps main.py)
├── narnia.spec            # PyInstaller configuration
├── build_exe.bat          # Automated build script
├── hooks/
│   └── hook-thread-env.py # Runtime hook for thread limiting
├── main.py                # Original dev entry point (unchanged)
└── dist/                  # Build output (created by PyInstaller)
    └── NarniaViewer/
        ├── NarniaViewer.exe
        ├── alice_result/  # Bundled sample data
        └── [many DLLs]
```

**Key Design**: `run_narnia.py` is a clean wrapper around `main.py` — your original code stays untouched!

---

## Output Structure

After building, you'll get:

```
dist/NarniaViewer/
├── NarniaViewer.exe       # Main executable (~2 MB)
├── python311.dll          # Python runtime
├── _internal/             # Python libraries and dependencies
│   ├── open3d/           # Open3D with GPU rendering DLLs
│   ├── numpy/            # NumPy
│   ├── scipy/            # SciPy
│   ├── sklearn/          # Scikit-learn
│   └── [many .pyd files]
├── alice_result/          # Sample data (bundled from repo)
│   └── 251120/
│       ├── ext/
│       └── bracing/
└── output/                # Created at runtime for exports
```

**Total Size**: 600-800 MB (Open3D GPU libraries are large)

---

## Testing Checklist

Test on a **clean Windows VM** (recommended) or separate machine:

### ✅ Basic Launch
- [ ] Double-click `NarniaViewer.exe` - GUI window appears
- [ ] No missing DLL errors
- [ ] Window appears on correct monitor

### ✅ Compute Tab (Generate)
- [ ] Load sample JSON: `alice_result/251120/ext/waveStackFields.json`
- [ ] Generate bracing (all 4 methods):
  - [ ] Static bracing
  - [ ] KeyField bracing
  - [ ] Adaptive bracing
  - [ ] Binary bracing
- [ ] Boolean operations work:
  - [ ] Union
  - [ ] Difference
  - [ ] Intersection
- [ ] Slice and iso-level controls update mesh in real-time

### ✅ Export/Import
- [ ] Export NPZ to `output/` folder
- [ ] Switch to "NPZ Viewer" tab
- [ ] Load exported NPZ file
- [ ] Export OBJ mesh

### ✅ GPU Compatibility
- [ ] Test on NVIDIA GPU system
- [ ] Test on AMD GPU system
- [ ] Test on Intel integrated GPU

### ✅ Performance
- [ ] GUI remains responsive during mesh generation
- [ ] No system freezes (thread limiting working)
- [ ] Startup time < 10 seconds

---

## Troubleshooting

### Build Errors

#### ImportError during build
```
Problem: Module not found during PyInstaller analysis
Solution: Add to hiddenimports in narnia.spec
```

#### Missing data files
```
Problem: alice_result/ not found in exe
Solution: Check datas section in narnia.spec
```

### Runtime Errors

#### DLL Load Failed
```
Problem: Missing Visual C++ Redistributables
Solution: Install VC++ Redist 2015-2022 (x64)
Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### Open3D GUI Crashes
```
Problem: GPU driver incompatibility
Solution 1: Update GPU drivers (NVIDIA/AMD/Intel)
Solution 2: Enable software rendering in run_narnia.py:
    Uncomment: os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
```

#### System Freezes
```
Problem: Thread limiting not working
Solution: Check hooks/hook-thread-env.py is loaded
Verify: Console shows "[PyInstaller Hook] Thread limiting environment variables set"
```

#### FileNotFoundError at runtime
```
Problem: Paths hardcoded to development environment
Solution: Use relative paths or sys._MEIPASS in bundled mode
```

---

## Advanced Configuration

### Switch to One-File Mode

Edit `narnia.spec`:

```python
# Change EXE section:
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,      # Add these
    a.zipfiles,      # Add these
    a.datas,         # Add these
    [],
    exclude_binaries=False,  # Change to False
    name='NarniaViewer',
    # ... rest unchanged
)

# Remove COLLECT section entirely
```

**Trade-off**: One-file is simpler (single .exe) but slower startup (extracts to temp folder each run).

### Reduce EXE Size

1. **Exclude sample data**: Remove `alice_result` from `datas` in `narnia.spec`
2. **Disable UPX**: Set `upx=False` if causing issues (increases size ~30%)
3. **Exclude matplotlib**: Already excluded in spec file

### Hide Console Window

Edit `narnia.spec`:

```python
exe = EXE(
    # ...
    console=False,  # Change from True
    # ...
)
```

**Warning**: Harder to debug runtime errors without console output.

### Add Custom Icon

1. Create or download `icon.ico` (256x256 recommended)
2. Place in repo root
3. Edit `narnia.spec`:

```python
exe = EXE(
    # ...
    icon='icon.ico',  # Uncomment this line
    # ...
)
```

---

## Distribution

### Package for Users

1. **Zip the output folder**:
   ```bash
   cd dist
   tar -a -c -f NarniaViewer_v1.0.zip NarniaViewer
   ```

2. **Include README.txt** with:
   - System requirements (Windows 10+, OpenGL 3.3+)
   - GPU driver recommendations
   - VC++ Redistributable link if needed

### System Requirements (for users)

- **OS**: Windows 10/11 (64-bit)
- **GPU**: OpenGL 3.3+ or DirectX 11
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 1 GB free space
- **Graphics Drivers**: Latest NVIDIA/AMD/Intel drivers

---

## Known Limitations

1. **Large file size**: Open3D + scipy = ~600-800 MB (unavoidable)
2. **Windows-only**: Monitor detection uses Windows API ([vis_utils.py](vis_utils.py#L39-L64))
3. **GPU required**: Software rendering is very slow
4. **First launch slow**: PyInstaller extracts to temp (one-file mode only)
5. **Antivirus false positives**: Some AVs flag PyInstaller exes (sign code to avoid)

---

## Development Workflow

```bash
# Make code changes to main.py, core/, gui/ as usual
python main.py  # Test normally in dev

# When ready to test exe:
build_exe.bat   # Rebuild exe
cd dist\NarniaViewer
NarniaViewer.exe  # Test exe

# Original files unchanged!
```

**Benefit**: `run_narnia.py` is just a thin wrapper — all your original code stays clean and testable.

---

## Support & Debugging

### Enable Debug Mode

Edit `narnia.spec`:

```python
exe = EXE(
    # ...
    debug=True,      # Enable debug logging
    console=True,    # Keep console visible
    # ...
)
```

Rebuild and check console output for import errors.

### Check What's Bundled

```bash
# List all files in exe
pyi-archive_viewer dist\NarniaViewer\NarniaViewer.exe

# Extract specific file for inspection
pyi-archive_viewer dist\NarniaViewer\NarniaViewer.exe
> X <filename>  # Extract to disk
```

### PyInstaller Logs

Build logs saved to:
- `build/NarniaViewer/warn-NarniaViewer.txt` - Warnings and missing imports
- `build/NarniaViewer/xref-NarniaViewer.html` - Cross-reference of all imports

---

## FAQ

**Q: Can I distribute just the .exe without the folder?**  
A: No (one-folder mode). Use one-file mode for single exe, but startup is slower.

**Q: Why is it so large?**  
A: Open3D bundles GPU rendering engines, scipy has BLAS/LAPACK libraries. Scientific apps are typically 500MB+.

**Q: Will it work on Windows 7?**  
A: Unlikely. Python 3.11+ and Open3D require Windows 10+.

**Q: Can I build on Mac/Linux?**  
A: No. PyInstaller builds are platform-specific. Use separate VM for each OS.

**Q: How to update the exe after code changes?**  
A: Just run `build_exe.bat` again. PyInstaller rebuilds from source.

**Q: Can I include MeshFromNPZ.py as separate exe?**  
A: Yes! Create `meshfromnpz.spec` pointing to MeshFromNPZ.py and build separately for Grasshopper subprocess use.

---

## Next Steps

1. **Test build**: Run `build_exe.bat` and verify all features work
2. **Test on clean VM**: Ensure no missing dependencies
3. **Test GPU compatibility**: Try on NVIDIA, AMD, and Intel graphics
4. **Create installer** (optional): Use Inno Setup or NSIS for professional installer
5. **Code signing** (optional): Sign exe to avoid antivirus false positives

---

**Build successfully? Star this repo and share your builds! 🚀**
