import json
import tempfile
import unittest
from pathlib import Path

from ozon_dimension_demo.benchmark import build_report, load_catalog, write_outputs


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = load_catalog()
        cls.report = build_report(cls.specs)

    def test_catalog_has_unique_scenarios(self):
        ids = [spec.id for spec in self.specs]
        self.assertEqual(len(ids), 9)
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_scenarios_match_expected_outcome(self):
        summary = self.report["summary"]
        self.assertEqual(summary["passed"], summary["total"])
        self.assertEqual(summary["geometry_passed"], summary["geometry_total"])
        self.assertEqual(summary["quality_passed"], summary["quality_total"])

    def test_dashboard_and_machine_readable_results_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            write_outputs(directory, self.report)
            output = Path(directory)
            self.assertTrue((output / "dashboard.html").exists())
            self.assertTrue((output / "overview.svg").exists())
            saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"]["total"], 9)
            dashboard = (output / "dashboard.html").read_text(encoding="utf-8")
            self.assertIn("Как алгоритм ведёт себя", dashboard)


if __name__ == "__main__":
    unittest.main()
