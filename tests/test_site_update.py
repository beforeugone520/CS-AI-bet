import json
import tempfile
import unittest
from pathlib import Path


class SiteUpdateTests(unittest.TestCase):
    def test_update_uses_primary_source_and_writes_manifest(self):
        from scripts.update_site_data import SourceCandidate, update_site_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            output_dir = repo_root / "data" / "cologne2026" / "site_updates"

            def fetcher(url, headers):
                self.assertEqual(url, "https://primary.example/results")
                return json.dumps(
                    {
                        "results": [
                            {
                                "date": "2026-06-05",
                                "event": "IEM Cologne 2026",
                                "status": "completed",
                                "team1": "BIG",
                                "team2": "NRG",
                                "winner": "BIG",
                                "best_of": 3,
                                "source_match_url": url,
                            },
                            {
                                "date": "2026-06-05",
                                "event": "IEM Cologne 2026",
                                "status": "completed",
                                "team1": "TYLOO",
                                "team2": "Lynn Vision",
                                "winner": "TYLOO",
                                "best_of": 3,
                                "source_match_url": url,
                            },
                        ]
                    }
                )

            report = update_site_sources(
                repo_root=repo_root,
                output_dir=output_dir,
                source_candidates=[SourceCandidate("primary-test", "https://primary.example/results")],
                fivee_candidate=None,
                fetcher=fetcher,
            )

            self.assertEqual(report["status"], "primary_success")
            self.assertEqual(report["selected_source"], "primary-test")
            self.assertEqual(report["completed_results"], 2)
            self.assertTrue((output_dir / "latest.json").exists())
            self.assertTrue((output_dir / "auto_results.csv").exists())
            self.assertTrue((output_dir / "auto_standings.csv").exists())

    def test_update_returns_cached_manifest_when_sources_fail(self):
        from scripts.update_site_data import SourceCandidate, update_site_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            output_dir = repo_root / "data" / "cologne2026" / "site_updates"
            output_dir.mkdir(parents=True)
            cached = {
                "status": "primary_success",
                "selected_source": "previous-valid",
                "completed_results": 1,
                "results_path": "data/cologne2026/site_updates/auto_results.csv",
                "standings_path": "data/cologne2026/site_updates/auto_standings.csv",
            }
            (output_dir / "latest.json").write_text(json.dumps(cached), encoding="utf-8")

            def failing_fetcher(url, headers):
                raise RuntimeError("source down")

            report = update_site_sources(
                repo_root=repo_root,
                output_dir=output_dir,
                source_candidates=[SourceCandidate("primary-test", "https://primary.example/results")],
                fivee_candidate=None,
                fetcher=failing_fetcher,
            )

            self.assertEqual(report["status"], "cached")
            self.assertEqual(report["selected_source"], "previous-valid")
            self.assertEqual(report["completed_results"], 1)
            self.assertEqual(report["attempts"][0]["status"], "error")

    def test_exporter_prefers_auto_update_manifest_paths(self):
        from scripts.export_site_data import _site_input_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            update_dir = repo_root / "data" / "cologne2026" / "site_updates"
            update_dir.mkdir(parents=True)
            (update_dir / "auto_results.csv").write_text("team1,team2,winner\nBIG,NRG,BIG\n", encoding="utf-8")
            (update_dir / "auto_standings.csv").write_text("team,wins,losses,status\nBIG,3,2,advanced\n", encoding="utf-8")
            (update_dir / "latest.json").write_text(
                json.dumps(
                    {
                        "results_path": "data/cologne2026/site_updates/auto_results.csv",
                        "standings_path": "data/cologne2026/site_updates/auto_standings.csv",
                    }
                ),
                encoding="utf-8",
            )

            paths = _site_input_paths(repo_root)

            self.assertEqual(paths["results"].name, "auto_results.csv")
            self.assertEqual(paths["standings"].name, "auto_standings.csv")


if __name__ == "__main__":
    unittest.main()
