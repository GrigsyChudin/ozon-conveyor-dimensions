import unittest

import numpy as np

from ozon_dimension_demo.pipeline import MeasurementConfig, measure_object
from ozon_dimension_demo.synthetic import generate_two_sensor_scene


class PipelineTests(unittest.TestCase):
    def test_demo_scene_is_measured_within_assignment_tolerance(self):
        scene = generate_two_sensor_scene(seed=7)
        result = measure_object(scene.sensor_clouds)
        self.assertEqual(result.quality.status, "OK")
        self.assertIsNotNone(result.box)
        errors = np.abs(result.box.dimensions - scene.truth_dimensions)
        tolerances = np.maximum(scene.truth_dimensions * 0.05, 5.0)
        self.assertTrue(np.all(errors <= tolerances), (errors, tolerances))

    def test_minimum_10_mm_object_is_not_lost(self):
        scene = generate_two_sensor_scene(10.0, 10.0, 10.0, yaw_deg=31.0, seed=9)
        result = measure_object(scene.sensor_clouds)
        self.assertEqual(result.quality.status, "OK")
        errors = np.abs(result.box.dimensions - scene.truth_dimensions)
        self.assertTrue(np.all(errors <= 5.0), errors)

    def test_belt_only_is_rejected(self):
        x = np.linspace(-100, 100, 30)
        y = np.linspace(-100, 100, 30)
        xx, yy = np.meshgrid(x, y)
        belt = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        result = measure_object({"sensor_1": belt, "sensor_2": belt})
        self.assertEqual(result.quality.status, "REJECT")
        self.assertIn("TOO_FEW_POINTS", result.quality.reasons)

    def test_one_sensor_with_too_small_share_is_rejected(self):
        scene = generate_two_sensor_scene(seed=4)
        tiny_second = scene.sensor_clouds["sensor_2"][:5]
        result = measure_object(
            {"sensor_1": scene.sensor_clouds["sensor_1"], "sensor_2": tiny_second},
            MeasurementConfig(min_sensor_share=0.10),
        )
        self.assertEqual(result.quality.status, "REJECT")
        self.assertIn("LOW_SHARE_sensor_2", result.quality.reasons)

    def test_missing_second_sensor_is_rejected(self):
        scene = generate_two_sensor_scene(seed=7)
        result = measure_object({"sensor_1": scene.sensor_clouds["sensor_1"]})
        self.assertEqual(result.quality.status, "REJECT")
        self.assertIsNone(result.box)
        self.assertIn("MISSING_SENSOR_sensor_2", result.quality.reasons)


if __name__ == "__main__":
    unittest.main()
