"""
GUI package for Narnia visualization.

This package contains GUI-related helper modules:
- widgets: Reusable widget builders (sliders, file pickers, etc.)
- mesh_builders: Functions for building meshes from scalar fields

All tab-building logic remains in NarniaCurveViewer for now due to tight coupling
with instance variables and callbacks.
"""

from . import widgets, mesh_builders

__all__ = ["widgets", "mesh_builders"]
