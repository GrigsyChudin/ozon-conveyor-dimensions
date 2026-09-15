import json
import tempfile
import unittest
from pathlib import Path

from ozon_dimension_demo.benchmark import load_catalog
from ozon_dimension_demo.dataset_generator import generate_datasets


class DatasetGeneratorTests(unittest.TestCase):
    def test_csv_clouds_and_metadata_are_exported(self):
        specs = load_catalog()
        selected = [specs[0], specs[-1]]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = generate_datasets(selected, output)

            self.assertEqual(len(manifest["datasets"]), 2)
            self.assertTrue((output / "minimum-cube" / "sensor_1.csv").exists())
            self.assertTrue((output / "minimum-cube" / "sensor_2.csv").exists())
            self.assertTrue((output / "missing-second-sensor" / "sensor_1.csv").exists())
            self.assertFalse((output / "missing-second-sensor" / "sensor_2.csv").exists())

            metadata = json.loads(
                (output / "missing-second-sensor" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["expected_status"], "REJECT")
            self.assertEqual(metadata["sensor_files"], {"sensor_1": "sensor_1.csv"})
            self.assertEqual(
                (output / "minimum-cube" / "sensor_1.csv").read_text(encoding="utf-8").splitlines()[0],
                "x_mm,y_mm,z_mm",
            )


if __name__ == "__main__":
    unittest.main()
