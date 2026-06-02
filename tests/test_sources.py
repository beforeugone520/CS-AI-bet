import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


HLTV_FIXTURE = """
<html>
  <body>
    <div class="result-con">
      <a class="a-reset" href="/matches/1/alpha-vs-bravo">
        <span class="team1">Alpha</span>
        <span class="team2">Bravo</span>
        <span class="score-won">13</span><span class="score-lost">9</span>
        <span class="event-name">IEM Cologne</span>
        <span class="map-text">Mirage</span>
        <span class="date">2026-05-20</span>
      </a>
    </div>
    <div class="result-con">
      <a href="/matches/2/charlie-vs-delta">
        <span class="team1">Charlie</span>
        <span class="team2">Delta</span>
        <span class="score-lost">8</span><span class="score-won">13</span>
        <span class="event-name">RMR Europe</span>
        <span class="map-text">Inferno</span>
        <span class="date">2026-05-21</span>
      </a>
    </div>
  </body>
</html>
"""


HLTV_EVENT_FIXTURE = """
<html>
  <body>
    <section class="teams-attending">
      <div class="team-card">
        <a href="/team/1/alpha"><span class="team-name">Alpha</span></a>
        <span class="event-seed">#1</span>
        <span class="world-rank">#4</span>
        <span class="qualification">VRS (Europe)</span>
      </div>
      <div class="team-card">
        <a href="/team/2/bravo"><span class="team-name">Bravo</span></a>
        <span class="event-seed">#2</span>
        <span class="world-rank">#22</span>
        <span class="qualification">VRS (Americas)</span>
      </div>
    </section>
  </body>
</html>
"""


HLTV_RANKING_FIXTURE = """
<html>
  <body>
    <div class="ranked-team standard-box">
      <span class="position">#1</span>
      <a href="/team/1/alpha"><span class="name">Alpha</span></a>
      <span class="points">927 points</span>
      <span class="country">Europe</span>
    </div>
    <div class="ranked-team standard-box">
      <span class="position">#2</span>
      <a href="/team/2/bravo"><span class="name">Bravo</span></a>
      <span class="points">855 points</span>
      <span class="country">Americas</span>
    </div>
    <div class="ranked-team standard-box">
      <span class="position">#3</span>
      <a href="/team/3/academy"><span class="name">Academy Young</span></a>
      <span class="points">600 points</span>
      <span class="country">Europe</span>
    </div>
  </body>
</html>
"""


HLTV_PLAYER_STATS_FIXTURE = """
<html>
  <body>
    <table class="stats-table">
      <tr class="player-row">
        <td class="player"><a href="/player/11/alpha-awp">alpha_awp</a></td>
        <td class="team">Alpha</td>
        <td class="rating">1.34</td>
        <td class="kd">1.28</td>
        <td class="opening-success">61%</td>
        <td class="clutch-winrate">64%</td>
        <td class="role">starter</td>
      </tr>
      <tr class="player-row">
        <td class="player"><a href="/player/12/bravo-stand-in">bravo_stand_in</a></td>
        <td class="team">Bravo</td>
        <td class="rating">0.88</td>
        <td class="kd">0.87</td>
        <td class="opening-success">43%</td>
        <td class="clutch-winrate">44%</td>
        <td class="role">substitute</td>
      </tr>
    </table>
  </body>
</html>
"""


class SourceWorkflowTests(unittest.TestCase):
    def test_hltv_fixture_parser_returns_match_rows(self):
        from cs2pickem.sources import HltvResultParser

        rows = HltvResultParser().parse_results_html(HLTV_FIXTURE)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["team1"], "Alpha")
        self.assertEqual(rows[0]["winner"], "Alpha")
        self.assertEqual(rows[0]["map"], "mirage")
        self.assertEqual(rows[0]["event_tier"], "S")
        self.assertEqual(rows[1]["winner"], "Delta")

    def test_hltv_event_parser_returns_seeded_team_rows(self):
        from cs2pickem.sources import HltvEventParser

        rows = HltvEventParser().parse_teams_html(HLTV_EVENT_FIXTURE)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["team"], "Alpha")
        self.assertEqual(rows[0]["seed"], 1)
        self.assertEqual(rows[0]["world_rank"], 4)
        self.assertEqual(rows[0]["qualification"], "VRS (Europe)")
        self.assertEqual(rows[0]["source_team_url"], "https://www.hltv.org/team/1/alpha")
        self.assertEqual(rows[1]["team"], "Bravo")
        self.assertEqual(rows[1]["seed"], 2)
        self.assertEqual(rows[1]["world_rank"], 22)
        self.assertEqual(rows[1]["qualification"], "VRS (Americas)")

    def test_hltv_ranking_parser_returns_top_professional_team_rows(self):
        from cs2pickem.sources import HltvRankingParser

        rows = HltvRankingParser().parse_rankings_html(HLTV_RANKING_FIXTURE, limit=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["team"], "Alpha")
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["points"], 927)
        self.assertEqual(rows[0]["region"], "Europe")
        self.assertEqual(rows[0]["source_team_url"], "https://www.hltv.org/team/1/alpha")
        self.assertEqual(rows[1]["team"], "Bravo")
        self.assertEqual(rows[1]["rank"], 2)

    def test_hltv_player_stats_parser_returns_merge_ready_rows(self):
        from cs2pickem.sources import HltvPlayerStatsParser

        rows = HltvPlayerStatsParser().parse_player_stats_html(HLTV_PLAYER_STATS_FIXTURE, default_date="2026-05-31")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["date"], "2026-05-31")
        self.assertEqual(rows[0]["team"], "Alpha")
        self.assertEqual(rows[0]["player"], "alpha_awp")
        self.assertEqual(rows[0]["rating"], 1.34)
        self.assertEqual(rows[0]["kd"], 1.28)
        self.assertEqual(rows[0]["opening_success"], 0.61)
        self.assertEqual(rows[0]["clutch_winrate"], 0.64)
        self.assertEqual(rows[0]["is_substitute"], 0)
        self.assertEqual(rows[0]["source_player_url"], "https://www.hltv.org/player/11/alpha-awp")
        self.assertEqual(rows[1]["is_substitute"], 1)

    def test_http_cache_uses_disk_before_fetching_again(self):
        from cs2pickem.sources import HttpCache

        calls = []

        def fetcher(url, headers):
            calls.append((url, headers))
            return "payload"

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = HttpCache(tmpdir, fetcher=fetcher)
            first = cache.get("https://example.test/results")
            second = cache.get("https://example.test/results")

        self.assertEqual(first, "payload")
        self.assertEqual(second, "payload")
        self.assertEqual(len(calls), 1)
        self.assertIn("User-Agent", calls[0][1])

    def test_http_cache_default_fetcher_falls_back_after_tls_failure(self):
        from cs2pickem import sources
        from cs2pickem.sources import HttpCache

        calls = []
        old_urllib = sources._urllib_fetch
        old_requests = getattr(sources, "_requests_fetch", None)
        old_curl = getattr(sources, "_curl_fetch", None)

        def failing_urllib(url, headers):
            calls.append("urllib")
            raise OSError("TLS EOF")

        def missing_requests(url, headers):
            calls.append("requests")
            raise RuntimeError("requests unavailable")

        def working_curl(url, headers):
            calls.append("curl")
            return HLTV_FIXTURE

        sources._urllib_fetch = failing_urllib
        sources._requests_fetch = missing_requests
        sources._curl_fetch = working_curl
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache = HttpCache(tmpdir)
                first = cache.get("https://www.hltv.org/major/cologne", refresh=True)
                second = cache.get("https://www.hltv.org/major/cologne")
        finally:
            sources._urllib_fetch = old_urllib
            if old_requests is None:
                delattr(sources, "_requests_fetch")
            else:
                sources._requests_fetch = old_requests
            if old_curl is None:
                delattr(sources, "_curl_fetch")
            else:
                sources._curl_fetch = old_curl

        self.assertEqual(first, HLTV_FIXTURE)
        self.assertEqual(second, HLTV_FIXTURE)
        self.assertEqual(calls, ["urllib", "requests", "curl"])

    def test_version_tags_apply_latest_patch_at_match_date(self):
        from cs2pickem.sources import annotate_version_tags, parse_version_log

        log = parse_version_log(
            "2026-04-10,pre-cologne-map-pool\n"
            "2026-05-15,incendiary-economy-update\n"
        )
        rows = [{"date": "2026-05-10"}, {"date": "2026-05-20"}]
        annotated = annotate_version_tags(rows, log)

        self.assertEqual(annotated[0]["version_tag"], "pre-cologne-map-pool")
        self.assertEqual(annotated[1]["version_tag"], "incendiary-economy-update")

    def test_update_pipeline_writes_dataset_manifest(self):
        from cs2pickem.update import update_dataset_from_html

        with tempfile.TemporaryDirectory() as tmpdir:
            version_path = os.path.join(tmpdir, "versions.csv")
            html_path = os.path.join(tmpdir, "hltv.html")
            output_path = os.path.join(tmpdir, "matches.json")
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(version_path, "w", encoding="utf-8") as handle:
                handle.write("2026-05-01,cologne-practice-patch\n")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(HLTV_FIXTURE)

            manifest = update_dataset_from_html(
                html_path=html_path,
                output_path=output_path,
                manifest_path=manifest_path,
                version_log_path=version_path,
                source_name="hltv-fixture",
            )

            with open(output_path, encoding="utf-8") as handle:
                rows = json.load(handle)
            with open(manifest_path, encoding="utf-8") as handle:
                manifest_from_disk = json.load(handle)

        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(manifest_from_disk["source"], "hltv-fixture")
        self.assertEqual(rows[0]["version_tag"], "cologne-practice-patch")

    def test_update_pipeline_can_append_to_long_lived_csv_dataset(self):
        from cs2pickem.data import read_matches_csv
        from cs2pickem.update import update_dataset_from_html

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "hltv.html")
            output_path = os.path.join(tmpdir, "matches.json")
            manifest_path = os.path.join(tmpdir, "manifest.json")
            dataset_path = os.path.join(tmpdir, "training_matches.csv")
            dataset_manifest_path = os.path.join(tmpdir, "training_manifest.json")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(HLTV_FIXTURE)

            first = update_dataset_from_html(
                html_path=html_path,
                output_path=output_path,
                manifest_path=manifest_path,
                dataset_path=dataset_path,
                dataset_manifest_path=dataset_manifest_path,
            )
            second = update_dataset_from_html(
                html_path=html_path,
                output_path=output_path,
                manifest_path=manifest_path,
                dataset_path=dataset_path,
                dataset_manifest_path=dataset_manifest_path,
            )
            merged = read_matches_csv(dataset_path)
            with open(dataset_manifest_path, encoding="utf-8") as handle:
                dataset_manifest = json.load(handle)

        self.assertEqual(first["dataset"]["added_rows"], 2)
        self.assertEqual(second["dataset"]["added_rows"], 0)
        self.assertEqual(len(merged), 2)
        self.assertEqual(dataset_manifest["rows"], 2)

    def test_update_pipeline_can_fetch_url_through_cache(self):
        from cs2pickem.update import update_dataset_from_url

        calls = []

        def fetcher(url, headers):
            calls.append(url)
            return HLTV_FIXTURE

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "matches.json")
            manifest_path = os.path.join(tmpdir, "manifest.json")
            manifest = update_dataset_from_url(
                url="https://www.hltv.org/results",
                cache_dir=os.path.join(tmpdir, "cache"),
                output_path=output_path,
                manifest_path=manifest_path,
                fetcher=fetcher,
            )
            second_manifest = update_dataset_from_url(
                url="https://www.hltv.org/results",
                cache_dir=os.path.join(tmpdir, "cache"),
                output_path=output_path,
                manifest_path=manifest_path,
                fetcher=fetcher,
            )

        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(second_manifest["rows"], 2)
        self.assertEqual(calls, ["https://www.hltv.org/results"])

    def test_daily_update_config_runs_multiple_result_jobs_and_deduplicates_dataset(self):
        from cs2pickem.data import read_matches_csv
        from cs2pickem.update import daily_update_from_config

        second_fixture = (
            HLTV_FIXTURE.replace("/matches/1/alpha-vs-bravo", "/matches/3/echo-vs-foxtrot")
            .replace("/matches/2/charlie-vs-delta", "/matches/4/golf-vs-hotel")
            .replace("Alpha", "Echo")
            .replace("Bravo", "Foxtrot")
            .replace("Charlie", "Golf")
            .replace("Delta", "Hotel")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            first_html = os.path.join(tmpdir, "day1.html")
            second_html = os.path.join(tmpdir, "day2.html")
            version_path = os.path.join(tmpdir, "versions.csv")
            config_path = os.path.join(tmpdir, "daily.json")
            output_dir = os.path.join(tmpdir, "daily-output")
            dataset_path = os.path.join(tmpdir, "training.csv")
            dataset_manifest_path = os.path.join(tmpdir, "training_manifest.json")
            with open(first_html, "w", encoding="utf-8") as handle:
                handle.write(HLTV_FIXTURE)
            with open(second_html, "w", encoding="utf-8") as handle:
                handle.write(second_fixture)
            with open(version_path, "w", encoding="utf-8") as handle:
                handle.write("2026-05-01,cologne-practice-patch\n")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": dataset_path,
                        "dataset_manifest": dataset_manifest_path,
                        "version_log": version_path,
                        "jobs": [
                            {"name": "hltv-day1", "kind": "results", "html": first_html, "source_name": "hltv-day1"},
                            {"name": "hltv-day2", "kind": "results", "html": second_html, "source_name": "hltv-day2"},
                        ],
                    },
                    handle,
                )

            first = daily_update_from_config(config_path, output_dir=output_dir)
            second = daily_update_from_config(config_path, output_dir=output_dir)
            merged = read_matches_csv(dataset_path)
            with open(dataset_manifest_path, encoding="utf-8") as handle:
                dataset_manifest = json.load(handle)
            output_paths_exist = all(os.path.exists(job["output_path"]) for job in first["job_reports"])
            manifest_paths_exist = all(os.path.exists(job["manifest_path"]) for job in first["job_reports"])

        self.assertEqual(first["jobs"], 2)
        self.assertEqual(first["total_rows"], 4)
        self.assertEqual(first["total_added_rows"], 4)
        self.assertEqual(second["total_added_rows"], 0)
        self.assertEqual(len(merged), 4)
        self.assertEqual(dataset_manifest["rows"], 4)
        self.assertTrue(output_paths_exist)
        self.assertTrue(manifest_paths_exist)

    def test_daily_update_url_job_parses_string_false_refresh_as_disabled(self):
        from cs2pickem.update import daily_update_from_config

        calls = []

        def fetcher(url, headers):
            calls.append(url)
            return HLTV_FIXTURE

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "daily.json")
            output_dir = os.path.join(tmpdir, "daily-output")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "jobs": [
                            {
                                "name": "hltv-results",
                                "kind": "results",
                                "url": "https://www.hltv.org/results",
                                "refresh": "false",
                            }
                        ],
                    },
                    handle,
                )

            daily_update_from_config(config_path, output_dir=output_dir, fetcher=fetcher)
            daily_update_from_config(config_path, output_dir=output_dir, fetcher=fetcher)

        self.assertEqual(calls, ["https://www.hltv.org/results"])

    def test_daily_update_can_augment_results_before_dataset_append(self):
        from cs2pickem.data import read_matches_csv, write_matches_csv
        from cs2pickem.update import daily_update_from_config

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "hltv.html")
            teams_path = os.path.join(tmpdir, "teams.csv")
            players_path = os.path.join(tmpdir, "players.csv")
            version_path = os.path.join(tmpdir, "versions.csv")
            config_path = os.path.join(tmpdir, "daily.json")
            output_dir = os.path.join(tmpdir, "daily-output")
            dataset_path = os.path.join(tmpdir, "training.csv")
            dataset_manifest_path = os.path.join(tmpdir, "training_manifest.json")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(HLTV_FIXTURE)
            with open(version_path, "w", encoding="utf-8") as handle:
                handle.write("2026-05-01,cologne-practice-patch\n")
            write_matches_csv(
                teams_path,
                [
                    {"team": "Alpha", "rank": 4, "rmr_points": 930, "major_best_placement": 2, "recent_winrate_10": 0.75, "bo1_winrate_6m": 0.71, "bo3_winrate_6m": 0.74},
                    {"team": "Bravo", "rank": 22, "rmr_points": 510, "major_best_placement": 16, "recent_winrate_10": 0.42, "bo1_winrate_6m": 0.46, "bo3_winrate_6m": 0.49},
                    {"team": "Charlie", "rank": 12, "rmr_points": 700, "major_best_placement": 8, "recent_winrate_10": 0.58, "bo1_winrate_6m": 0.55, "bo3_winrate_6m": 0.57},
                    {"team": "Delta", "rank": 8, "rmr_points": 860, "major_best_placement": 4, "recent_winrate_10": 0.7, "bo1_winrate_6m": 0.62, "bo3_winrate_6m": 0.69},
                ],
            )
            write_matches_csv(
                players_path,
                [
                    {"date": "2026-05-10", "team": "Alpha", "player": "alpha_awp", "rating": 1.32, "kd": 1.24, "opening_success": 0.61, "clutch_winrate": 0.64, "is_substitute": 0},
                    {"date": "2026-05-11", "team": "Bravo", "player": "bravo_sub", "rating": 0.88, "kd": 0.86, "opening_success": 0.43, "clutch_winrate": 0.44, "is_substitute": 1},
                    {"date": "2026-05-12", "team": "Charlie", "player": "charlie_rifle", "rating": 1.08, "kd": 1.04, "opening_success": 0.53, "clutch_winrate": 0.55, "is_substitute": 0},
                    {"date": "2026-05-12", "team": "Delta", "player": "delta_star", "rating": 1.22, "kd": 1.18, "opening_success": 0.58, "clutch_winrate": 0.6, "is_substitute": 0},
                ],
            )
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": dataset_path,
                        "dataset_manifest": dataset_manifest_path,
                        "version_log": version_path,
                        "team_metadata": teams_path,
                        "player_stats": players_path,
                        "player_window_days": 15,
                        "default_swiss_state": True,
                        "jobs": [{"name": "hltv-results", "kind": "results", "html": html_path}],
                    },
                    handle,
                )

            manifest = daily_update_from_config(config_path, output_dir=output_dir)
            merged = read_matches_csv(dataset_path)

        self.assertEqual(manifest["total_added_rows"], 2)
        self.assertEqual(merged[0]["version_tag"], "cologne-practice-patch")
        self.assertEqual(merged[0]["team1_rank"], 4)
        self.assertEqual(merged[0]["team2_rmr_points"], 510)
        self.assertEqual(merged[0]["team1_major_best_placement"], 2)
        self.assertEqual(merged[0]["team1_player_sample"], 1)
        self.assertEqual(merged[0]["team2_substitute_flag"], 1)
        self.assertAlmostEqual(merged[0]["team1_rating"], 1.32)
        self.assertEqual(merged[0]["swiss_round"], 1)
        self.assertEqual(merged[0]["team1_wins"], 0)
        self.assertTrue(dict(manifest["job_reports"][0]["augmentation"])["team_metadata"]["applied"])

    def test_daily_update_manifest_reports_dataset_coverage_gaps(self):
        from cs2pickem.data import write_matches_csv
        from cs2pickem.update import daily_update_from_config

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "hltv.html")
            participants_path = os.path.join(tmpdir, "participants.csv")
            top_teams_path = os.path.join(tmpdir, "top80.csv")
            config_path = os.path.join(tmpdir, "daily.json")
            output_dir = os.path.join(tmpdir, "daily-output")
            dataset_path = os.path.join(tmpdir, "training.csv")
            dataset_manifest_path = os.path.join(tmpdir, "training_manifest.json")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(HLTV_FIXTURE)
            write_matches_csv(participants_path, [{"team": "Alpha"}, {"team": "Delta"}, {"team": "MissingParticipant"}])
            write_matches_csv(top_teams_path, [{"team": "Alpha"}, {"team": "Bravo"}, {"team": "MissingTop"}])
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": dataset_path,
                        "dataset_manifest": dataset_manifest_path,
                        "participants": participants_path,
                        "top_teams": top_teams_path,
                        "minimum_rows": 5,
                        "required_teams": 5,
                        "jobs": [{"name": "hltv-results", "kind": "results", "html": html_path}],
                    },
                    handle,
                )

            manifest = daily_update_from_config(config_path, output_dir=output_dir)
            with open(os.path.join(output_dir, "daily_update_manifest.json"), encoding="utf-8") as handle:
                disk_manifest = json.load(handle)

        self.assertEqual(manifest["coverage"]["rows"], 2)
        self.assertEqual(manifest["coverage"]["rows_remaining"], 3)
        self.assertEqual(manifest["coverage"]["teams"], 4)
        self.assertEqual(manifest["coverage"]["teams_remaining"], 1)
        self.assertEqual(manifest["coverage"]["participant_coverage"]["missing"], ["MissingParticipant"])
        self.assertEqual(manifest["coverage"]["top_team_coverage"]["missing"], ["MissingTop"])
        self.assertEqual(disk_manifest["coverage"], manifest["coverage"])

    def test_daily_update_cli_runs_configured_jobs(self):
        from cs2pickem.cli import main
        from cs2pickem.data import read_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "hltv.html")
            config_path = os.path.join(tmpdir, "daily.json")
            output_dir = os.path.join(tmpdir, "daily-output")
            dataset_path = os.path.join(tmpdir, "training.csv")
            dataset_manifest_path = os.path.join(tmpdir, "training_manifest.json")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(HLTV_FIXTURE)
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": dataset_path,
                        "dataset_manifest": dataset_manifest_path,
                        "jobs": [{"name": "hltv-results", "kind": "results", "html": html_path}],
                    },
                    handle,
                )

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "daily-update",
                "--config",
                config_path,
                "--output-dir",
                output_dir,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv
            merged = read_matches_csv(dataset_path)
            manifest_path = os.path.join(output_dir, "daily_update_manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(merged), 2)
        self.assertEqual(manifest["total_added_rows"], 2)

    def test_event_teams_cli_writes_csv_and_manifest(self):
        from cs2pickem.cli import main
        from cs2pickem.data import read_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "event.html")
            output_path = os.path.join(tmpdir, "participants.csv")
            manifest_path = os.path.join(tmpdir, "participants_manifest.json")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(HLTV_EVENT_FIXTURE)

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "event-teams",
                "--html",
                html_path,
                "--output",
                output_path,
                "--manifest",
                manifest_path,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            rows = read_matches_csv(output_path)
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertEqual([row["team"] for row in rows], ["Alpha", "Bravo"])
        self.assertEqual(rows[1]["seed"], 2)
        self.assertEqual(rows[1]["world_rank"], 22)
        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(manifest["teams"], ["Alpha", "Bravo"])

    def test_rankings_cli_writes_top_team_csv_and_manifest(self):
        from cs2pickem.cli import main
        from cs2pickem.data import read_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "rankings.html")
            output_path = os.path.join(tmpdir, "top80.csv")
            manifest_path = os.path.join(tmpdir, "top80_manifest.json")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(HLTV_RANKING_FIXTURE)

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "rankings",
                "--html",
                html_path,
                "--output",
                output_path,
                "--manifest",
                manifest_path,
                "--limit",
                "2",
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            rows = read_matches_csv(output_path)
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertEqual([row["team"] for row in rows], ["Alpha", "Bravo"])
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[1]["points"], 855)
        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(manifest["limit"], 2)

    def test_player_stats_cli_writes_merge_ready_csv_and_manifest(self):
        from cs2pickem.cli import main
        from cs2pickem.data import read_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "players.html")
            output_path = os.path.join(tmpdir, "player_stats.csv")
            manifest_path = os.path.join(tmpdir, "player_stats_manifest.json")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(HLTV_PLAYER_STATS_FIXTURE)

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "player-stats",
                "--html",
                html_path,
                "--date",
                "2026-05-31",
                "--output",
                output_path,
                "--manifest",
                manifest_path,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            rows = read_matches_csv(output_path)
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertEqual([row["player"] for row in rows], ["alpha_awp", "bravo_stand_in"])
        self.assertEqual(rows[1]["is_substitute"], 1)
        self.assertEqual(manifest["rows"], 2)
        self.assertEqual(manifest["date"], "2026-05-31")


if __name__ == "__main__":
    unittest.main()
