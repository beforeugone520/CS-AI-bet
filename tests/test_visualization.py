import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def training_report():
    return {
        "feature_importance": {
            "rank_diff": 0.82,
            "map_winrate_diff": 0.64,
            "star_rating_diff": 0.51,
        },
        "probabilities": [
            {"date": "2026-05-20", "winner_probability_team1": 0.62},
            {"date": "2026-05-21", "winner_probability_team1": 0.48},
            {"date": "2026-05-22", "winner_probability_team1": 0.81},
        ],
    }


class VisualizationTests(unittest.TestCase):
    def test_write_training_visualizations_outputs_importance_and_probability_charts(self):
        from cs2pickem.visualization import write_training_visualizations

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_training_visualizations(training_report(), tmpdir, prefix="demo")

            self.assertTrue(os.path.exists(manifest["feature_importance_path"]))
            self.assertTrue(os.path.exists(manifest["probability_distribution_path"]))
            self.assertGreater(os.path.getsize(manifest["feature_importance_path"]), 0)
            self.assertGreater(os.path.getsize(manifest["probability_distribution_path"]), 0)

        self.assertEqual(manifest["feature_count"], 3)
        self.assertEqual(manifest["probability_count"], 3)
        self.assertIn(manifest["renderer"], {"matplotlib", "svg"})

    def test_cli_visualize_writes_manifest_from_training_report(self):
        from cs2pickem.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "train_report.json")
            output_dir = os.path.join(tmpdir, "viz")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(training_report(), handle)

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "visualize",
                "--training-report",
                report_path,
                "--output-dir",
                output_dir,
                "--prefix",
                "demo",
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            with open(os.path.join(output_dir, "demo_visualization_manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)

            chart_exists = os.path.exists(manifest["feature_importance_path"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["feature_count"], 3)
        self.assertTrue(chart_exists)


if __name__ == "__main__":
    unittest.main()
