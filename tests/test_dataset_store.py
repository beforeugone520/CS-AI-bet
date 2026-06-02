import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def existing_rows():
    return [
        {
            "date": "2026-05-20",
            "event": "IEM Cologne",
            "event_tier": "S",
            "status": "completed",
            "team1": "Alpha",
            "team2": "Bravo",
            "winner": "Alpha",
            "best_of": 1,
            "map": "mirage",
            "source": "hltv",
            "source_match_url": "https://www.hltv.org/matches/1/alpha-vs-bravo",
        }
    ]


def incoming_rows():
    return [
        {
            "date": "2026-05-20",
            "event": "IEM Cologne",
            "event_tier": "S",
            "status": "completed",
            "team1": "Alpha",
            "team2": "Bravo",
            "winner": "Alpha",
            "best_of": 1,
            "map": "mirage",
            "source": "hltv",
            "source_match_url": "https://www.hltv.org/matches/1/alpha-vs-bravo",
        },
        {
            "date": "2026-05-21",
            "event": "RMR Europe",
            "event_tier": "S",
            "status": "completed",
            "team1": "Charlie",
            "team2": "Delta",
            "winner": "Delta",
            "best_of": 3,
            "map": "inferno",
            "source": "hltv",
            "source_match_url": "https://www.hltv.org/matches/2/charlie-vs-delta",
        },
    ]


class DatasetStoreTests(unittest.TestCase):
    def test_append_rows_deduplicates_and_writes_manifest(self):
        from cs2pickem.data import read_matches_csv, write_matches_csv
        from cs2pickem.dataset_store import append_matches_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = os.path.join(tmpdir, "matches.csv")
            manifest_path = os.path.join(tmpdir, "manifest.json")
            write_matches_csv(dataset_path, existing_rows())

            report = append_matches_dataset(dataset_path, incoming_rows(), manifest_path=manifest_path, source_name="hltv")
            merged = read_matches_csv(dataset_path)
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(report["existing_rows"], 1)
        self.assertEqual(report["incoming_rows"], 2)
        self.assertEqual(report["added_rows"], 1)
        self.assertEqual(report["total_rows"], 2)
        self.assertEqual(len(merged), 2)
        self.assertEqual(manifest["date_min"], "2026-05-20")
        self.assertEqual(manifest["date_max"], "2026-05-21")
        self.assertEqual(manifest["teams"], 4)
        self.assertEqual(manifest["sources"], ["hltv"])

    def test_match_identity_falls_back_to_date_teams_map_when_source_url_missing(self):
        from cs2pickem.dataset_store import match_identity

        left = {"date": "2026-05-20", "team1": "Alpha", "team2": "Bravo", "map": "mirage"}
        right = {"date": "2026-05-20", "team1": "Bravo", "team2": "Alpha", "map": "mirage"}

        self.assertEqual(match_identity(left), match_identity(right))

    def test_dataset_coverage_report_summarizes_rows_teams_and_missing_lists(self):
        from cs2pickem.dataset_store import dataset_coverage_report

        report = dataset_coverage_report(
            incoming_rows(),
            minimum_rows=5,
            required_teams=5,
            participant_teams=["Alpha", "Delta", "MissingParticipant"],
            top_teams=["Alpha", "Bravo", "MissingTop"],
        )

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["rows_remaining"], 3)
        self.assertEqual(report["teams"], 4)
        self.assertEqual(report["teams_remaining"], 1)
        self.assertEqual(report["participant_coverage"]["missing"], ["MissingParticipant"])
        self.assertEqual(report["top_team_coverage"]["missing"], ["MissingTop"])


if __name__ == "__main__":
    unittest.main()
