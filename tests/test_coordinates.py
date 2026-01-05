import unittest

import numpy as np

import core


class TestCoordinateConventions(unittest.TestCase):
    def test_center_rc_world_roundtrip(self):
        # Use odd dims so the center is an integer pixel.
        nx, ny = 5, 7
        bounds_min = np.array([0.0, 0.0, 0.0])
        bounds_max = np.array([10.0, 20.0, 30.0])

        rc_center = np.array([ny // 2, nx // 2])  # (row, col)

        xy = core.rc_to_world_xy(rc_center, bounds_min, bounds_max, nx=nx, ny=ny)
        rc_back = core.world_xy_to_rc(xy, bounds_min, bounds_max, nx=nx, ny=ny)

        np.testing.assert_array_equal(rc_back, rc_center)


if __name__ == "__main__":
    unittest.main()
