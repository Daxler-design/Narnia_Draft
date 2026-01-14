"""
PyInstaller Runtime Hook - Thread Environment Variables

This hook runs BEFORE any user code or imports.
It sets environment variables to limit threading and prevent GUI freezes.

These settings match the approach in vis_utils.py but are applied
earlier in the PyInstaller boot sequence.
"""
import os

# Limit threading for all math libraries to prevent system freezes
# Must be set before numpy/scipy/sklearn are imported
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

print("[PyInstaller Hook] Thread limiting environment variables set")
