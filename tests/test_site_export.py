import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SiteExportTests(unittest.TestCase):
    def test_export_site_data_writes_static_contract(self):
        from scripts.export_site_data import export_site_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = export_site_data(ROOT, Path(tmpdir))

            self.assertEqual(report["event_id"], "iem-cologne-2026")
            self.assertEqual(report["current_stage"], "stage-1")
            self.assertEqual(report["files_written"] >= 7, True)

            latest = json.loads((Path(tmpdir) / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["event_id"], "iem-cologne-2026")
            self.assertEqual(latest["current_view"], "swiss")
            self.assertIn(latest["source_status"], {"primary_success", "fallback_success", "cached"})

            stage1 = json.loads((Path(tmpdir) / "stages" / "stage-1.json").read_text(encoding="utf-8"))
            self.assertEqual(stage1["format"], "swiss")
            self.assertEqual(stage1["status"], "live")
            self.assertGreaterEqual(len(stage1["standings"]), 16)
            self.assertGreaterEqual(len(stage1["fixtures"]), 1)

            stage2 = json.loads((Path(tmpdir) / "stages" / "stage-2.json").read_text(encoding="utf-8"))
            self.assertEqual(stage2["format"], "swiss")
            self.assertEqual(stage2["status"], "upcoming")
            self.assertIn("empty_state", stage2)

            stage3 = json.loads((Path(tmpdir) / "stages" / "stage-3.json").read_text(encoding="utf-8"))
            self.assertEqual(stage3["format"], "playoff")
            self.assertEqual(stage3["status"], "upcoming")
            self.assertIn("bracket", stage3)

    def test_export_site_data_preserves_pickem_summary(self):
        from scripts.export_site_data import export_site_data

        with tempfile.TemporaryDirectory() as tmpdir:
            export_site_data(ROOT, Path(tmpdir))
            pickem = json.loads((Path(tmpdir) / "pickem" / "current.json").read_text(encoding="utf-8"))

            self.assertEqual(pickem["summary"], {"alive": 2, "broken": 4, "locked": 4, "missing": 0})
            self.assertEqual(sorted(pickem["alive_picks"]), ["BIG", "TYLOO"])
            self.assertIn("Gaimin Gladiators", pickem["locked_picks"])


if __name__ == "__main__":
    unittest.main()
