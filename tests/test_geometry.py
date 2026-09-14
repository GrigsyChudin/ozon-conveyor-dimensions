import math
import unittest

import numpy as np

from ozon_dimension_demo.geometry import constrained_obb, convex_hull_2d, radius_outlier_mask


def rectangle_points(length: float, width: float, height: float, yaw_deg: float) -> np.ndarray:
    xs = np.linspace(-length / 2, length / 2, 21)
    ys = np.linspace(-width / 2, width / 2, 13)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    top = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, height)])
    angle = math.radians(yaw_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    top[:, :2] = top[:, :2] @ rotation.T
    return top


class GeometryTests(unittest.TestCase):
    def test_convex_hull_discards_inner_points(self):
        points = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [1, 0.5], [1.5, 0.4]])
        hull = convex_hull_2d(points)
        self.assertEqual(len(hull), 4)

    def test_rotated_rectangle_dimensions(self):
        points = rectangle_points(200.0, 100.0, 40.0, 27.0)
        box = constrained_obb(points, belt_z=0.0)
        np.testing.assert_allclose(box.dimensions, [200.0, 100.0, 40.0], atol=1e-8)
        self.assertAlmostEqual(abs(math.degrees(box.yaw_rad)), 27.0, places=6)

    def test_axes_are_sorted_length_first(self):
        points = rectangle_points(70.0, 160.0, 25.0, -18.0)
        box = constrained_obb(points)
        np.testing.assert_allclose(box.dimensions, [160.0, 70.0, 25.0], atol=1e-8)

    def test_radius_filter_removes_isolated_point(self):
        cluster = np.array([[0, 0, 10], [1, 0, 10], [0, 1, 10], [1, 1, 10]], dtype=float)
        points = np.vstack([cluster, [100, 100, 100]])
        mask = radius_outlier_mask(points, radius_mm=2.0, min_neighbors=2)
        self.assertEqual(mask.tolist(), [True, True, True, True, False])

    def test_non_rectangular_u_shape_footprint(self):
        footprint = np.array(
            [
                [-50.0, -30.0],
                [50.0, -30.0],
                [50.0, 30.0],
                [20.0, 30.0],
                [20.0, -5.0],
                [-20.0, -5.0],
                [-20.0, 30.0],
                [-50.0, 30.0],
            ]
        )
        angle = math.radians(23.0)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        rotated = footprint @ rotation.T
        points = np.column_stack([rotated, np.full(len(rotated), 25.0)])
        box = constrained_obb(points)
        np.testing.assert_allclose(box.dimensions, [100.0, 60.0, 25.0], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
