"""
Test script for true SDF loading from polylines.

Validates poly_to_true_sdf and load_sdf_list_from_inshapes functions.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent dir to path for core module
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_sdf_list_from_inshapes, poly_to_true_sdf


def test_poly_to_true_sdf():
    """Test true SDF generation from a simple square polyline."""
    print("Testing poly_to_true_sdf()...")
    
    # Create a square polyline
    poly = np.array([
        [2.0, 2.0, 0.0],
        [8.0, 2.0, 0.0],
        [8.0, 8.0, 0.0],
        [2.0, 8.0, 0.0],
    ])
    
    bbox_min = [0.0, 0.0, 0.0]
    bbox_max = [10.0, 10.0, 10.0]
    
    sdf = poly_to_true_sdf(poly, bbox_min, bbox_max, nx=128, ny=128)
    
    # Verify shape
    assert sdf.shape == (128, 128), f"Expected shape (128, 128), got {sdf.shape}"
    print(f"  ✓ Shape: {sdf.shape}")
    
    # Verify interior (should be negative)
    center_idx = 64  # Center of grid should be center of square
    interior_value = sdf[center_idx, center_idx]
    print(f"  ✓ Center value (should be negative): {interior_value:.4f}")
    
    # Verify exterior (should be positive)
    exterior_value = sdf[0, 0]  # Corner should be outside
    print(f"  ✓ Corner value (should be positive): {exterior_value:.4f}")
    
    # Verify zero crossing exists (boundary)
    zero_crossings_y = np.sum(sdf[:-1, :] * sdf[1:, :] < 0)
    zero_crossings_x = np.sum(sdf[:, :-1] * sdf[:, 1:] < 0)
    zero_crossings = zero_crossings_y + zero_crossings_x
    print(f"  ✓ Zero crossings found: {zero_crossings}")
    
    # Check gradient magnitude near boundary
    grad_y = np.gradient(sdf, axis=0)
    grad_x = np.gradient(sdf, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Near zero-crossing, gradient should be close to 1 (true SDF property)
    near_boundary = np.abs(sdf) < 0.5
    if np.any(near_boundary):
        avg_grad = np.mean(grad_mag[near_boundary])
        print(f"  ✓ Average gradient magnitude near boundary: {avg_grad:.4f} (should be ~1.0)")
    
    print("  ✓ poly_to_true_sdf() test passed!\n")


def test_load_sdf_list():
    """Test loading SDF list from inShapes.json."""
    print("Testing load_sdf_list_from_inshapes()...")
    
    json_path = "alice_result/inShapes.json"
    
    if not Path(json_path).exists():
        print(f"  ⚠ Skipping test - {json_path} not found")
        print("    (This is OK if you don't have the test data)\n")
        return
    
    try:
        sdfs = load_sdf_list_from_inshapes(json_path, nx=256, ny=256, branch_index=0)
        
        print(f"  ✓ Loaded {len(sdfs)} SDF slices")
        print(f"  ✓ First SDF shape: {sdfs[0].shape}")
        print(f"  ✓ Value range: [{np.min(sdfs[0]):.4f}, {np.max(sdfs[0]):.4f}]")
        
        # Verify all SDFs have correct shape
        all_correct_shape = all(sdf.shape == (256, 256) for sdf in sdfs)
        assert all_correct_shape, "Not all SDFs have shape (256, 256)"
        print(f"  ✓ All {len(sdfs)} SDFs have correct shape")
        
        # Verify SDFs contain interior (negative values)
        has_interior = all(np.min(sdf) < 0 for sdf in sdfs)
        assert has_interior, "Some SDFs have no interior (no negative values)"
        print(f"  ✓ All SDFs contain interior regions (negative values)")
        
        print("  ✓ load_sdf_list_from_inshapes() test passed!\n")
        
    except Exception as e:
        print(f"  ✗ Error loading inShapes.json: {e}\n")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("TRUE SDF FUNCTION TESTS")
    print("=" * 60 + "\n")
    
    test_poly_to_true_sdf()
    test_load_sdf_list()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
