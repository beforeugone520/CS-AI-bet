import contextlib
import io
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def fixture_rows():
    return [
        {"date": "2026-06-01", "team1": "Alpha", "team2": "Bravo", "best_of": 1, "map": "unknown"},
        {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "best_of": 3, "map": "unknown"},
    ]


def bp_rows():
    return [
        {
            "date": "2026-06-01",
            "source": "analyst-note",
            "team1": "Bravo",
            "team2": "Alpha",
            "map": "de_inferno",
            "confidence": 0.82,
            "team1_bans": "nuke|ancient",
            "team2_bans": "anubis",
        }
    ]


class BpIntelTests(unittest.TestCase):
    def test_bp_module_is_part_of_public_package_exports(self):
        from cs2pickem import __all__

        self.assertIn("bp", __all__)

    def test_merge_bp_intel_overrides_confirmed_map_and_swaps_team_fields(self):
        from cs2pickem.bp import merge_bp_into_fixtures

        merged, report = merge_bp_into_fixtures(fixture_rows(), bp_rows())

        self.assertEqual(report, {"fixtures": 2, "matched": 1, "unmatched": 1, "map_overrides": 1})
        self.assertEqual(merged[0]["map"], "inferno")
        self.assertEqual(merged[0]["bp_source"], "analyst-note")
        self.assertEqual(merged[0]["bp_confidence"], 0.82)
        self.assertEqual(merged[0]["team1_bans"], "anubis")
        self.assertEqual(merged[0]["team2_bans"], "nuke|ancient")
        self.assertEqual(merged[1]["map"], "unknown")

    def test_cli_merge_bp_writes_augmented_fixtures(self):
        from cs2pickem.cli import main
        from cs2pickem.data import read_matches_csv, write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_path = os.path.join(tmpdir, "fixtures.csv")
            bp_path = os.path.join(tmpdir, "bp.csv")
            output_path = os.path.join(tmpdir, "fixtures_with_bp.csv")
            write_matches_csv(fixtures_path, fixture_rows())
            write_matches_csv(bp_path, bp_rows())

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "merge-bp",
                "--fixtures",
                fixtures_path,
                "--bp",
                bp_path,
                "--output",
                output_path,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            merged = read_matches_csv(output_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(merged[0]["map"], "inferno")
        self.assertEqual(merged[0]["bp_applied"], 1)


if __name__ == "__main__":
    unittest.main()
