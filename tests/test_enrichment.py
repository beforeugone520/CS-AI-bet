import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def raw_history():
    rows = []
    for index, date in enumerate(["2026-05-01", "2026-05-03", "2026-05-05", "2026-05-07", "2026-05-09"]):
        rows.append(
            {
                "date": date,
                "event": "IEM Cologne Qualifier",
                "event_tier": "S",
                "status": "completed",
                "team1": "Alpha",
                "team2": "Bravo" if index % 2 == 0 else "Charlie",
                "winner": "Alpha" if index != 3 else "Charlie",
                "best_of": 1 if index % 2 == 0 else 3,
                "map": "mirage" if index < 4 else "inferno",
                "team1_rank": 4,
                "team2_rank": 18 + index,
                "team1_rmr_points": 900,
                "team2_rmr_points": 520,
                "team1_rating": 1.18,
                "team2_rating": 1.02,
                "team1_kd": 1.15,
                "team2_kd": 0.99,
                "team1_opening_success": 0.56,
                "team2_opening_success": 0.48,
                "team1_clutch_winrate": 0.61,
                "team2_clutch_winrate": 0.44,
                "team1_star_rating": 1.31,
                "team2_star_rating": 1.08,
                "odds_team1": 1.55,
                "odds_team2": 2.35,
            }
        )
    rows.append(
        {
            "date": "2026-05-12",
            "event": "IEM Cologne Qualifier",
            "event_tier": "S",
            "status": "completed",
            "team1": "Bravo",
            "team2": "Alpha",
            "winner": "Alpha",
            "best_of": 1,
            "map": "mirage",
            "team1_rank": 22,
            "team2_rank": 4,
            "team1_rmr_points": 500,
            "team2_rmr_points": 900,
            "team1_rating": 1.01,
            "team2_rating": 1.2,
            "team1_kd": 0.98,
            "team2_kd": 1.17,
            "team1_opening_success": 0.47,
            "team2_opening_success": 0.57,
            "team1_clutch_winrate": 0.43,
            "team2_clutch_winrate": 0.62,
            "team1_star_rating": 1.04,
            "team2_star_rating": 1.34,
            "odds_team1": 2.8,
            "odds_team2": 1.42,
        }
    )
    return rows


class EnrichmentTests(unittest.TestCase):
    def test_enrich_matches_uses_only_prior_rows_for_rolling_features(self):
        from cs2pickem.enrichment import enrich_match_history

        enriched = enrich_match_history(raw_history())
        last = enriched[-1]

        self.assertEqual(last["team1"], "Bravo")
        self.assertEqual(last["team2"], "Alpha")
        self.assertAlmostEqual(last["team2_recent_winrate_5"], 0.8)
        self.assertAlmostEqual(last["team2_recent_winrate_10"], 0.8)
        self.assertAlmostEqual(last["team2_bo1_winrate_6m"], 1.0)
        self.assertAlmostEqual(last["team2_bo3_winrate_6m"], 1 / 2)
        self.assertAlmostEqual(last["team2_map_winrate"], 3 / 4)
        self.assertAlmostEqual(last["h2h_team1_winrate"], 0.0)
        self.assertEqual(last["team2_current_streak"], 1)
        self.assertEqual(last["team2_matches_30d"], 5)

    def test_bo_winrate_6m_excludes_matches_older_than_window(self):
        from cs2pickem.enrichment import enrich_match_history

        # Alpha plays five bo1 matches: four very old wins (well outside the 6-month
        # window) and one recent loss inside the window. The "_6m" winrate must only
        # count the recent match, so it should be 0.0, not 4/5 (all-time).
        rows = [
            {"date": "2024-01-01", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2024-01-08", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2024-01-15", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2024-01-22", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2025-06-01", "team1": "Alpha", "team2": "Charlie", "winner": "Charlie", "best_of": 1, "map": "mirage"},
            {"date": "2025-06-10", "team1": "Alpha", "team2": "Delta", "winner": "Delta", "best_of": 1, "map": "mirage"},
        ]

        enriched = enrich_match_history(rows)
        last = enriched[-1]
        self.assertEqual(last["team1"], "Alpha")
        # Prior to the last match Alpha has 4 old wins (outside window) + 1 recent loss
        # inside the window. The 6-month bo1 winrate must reflect only the recent loss.
        self.assertAlmostEqual(last["team1_bo1_winrate_6m"], 0.0)

    def test_bo_winrate_6m_counts_all_in_window_matches(self):
        from cs2pickem.enrichment import enrich_match_history

        # Two recent bo1 matches within the window (one win, one loss) -> 0.5,
        # plus an in-window bo3 win -> 1.0.
        rows = [
            {"date": "2026-01-05", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2026-02-05", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo", "best_of": 1, "map": "mirage"},
            {"date": "2026-03-05", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 3, "map": "inferno"},
            {"date": "2026-04-05", "team1": "Alpha", "team2": "Charlie", "winner": "Charlie", "best_of": 1, "map": "nuke"},
        ]

        enriched = enrich_match_history(rows)
        last = enriched[-1]
        self.assertEqual(last["team1"], "Alpha")
        # bo1 prior to last match: win (1/5) + loss (2/5) -> 1/2
        self.assertAlmostEqual(last["team1_bo1_winrate_6m"], 1 / 2)
        # bo3 prior to last match: single win -> 1.0
        self.assertAlmostEqual(last["team1_bo3_winrate_6m"], 1.0)

    def test_enrich_adds_leakage_free_pre_match_elo(self):
        from cs2pickem.enrichment import enrich_match_history

        enriched = enrich_match_history(raw_history())

        # First match: both teams enter at the Elo base (no prior games, no leakage).
        self.assertEqual(enriched[0]["team1_elo"], 1500.0)
        self.assertEqual(enriched[0]["team2_elo"], 1500.0)
        # By the last match Alpha (the dominant winner) has a higher rating than its opponent.
        last = enriched[-1]
        self.assertEqual(last["team2"], "Alpha")
        self.assertGreater(last["team2_elo"], 1500.0)
        self.assertGreater(last["team2_elo"], last["team1_elo"])

    def test_h2h_uses_only_last_three_prior_meetings(self):
        from cs2pickem.enrichment import enrich_match_history

        rows = [
            {"date": "2026-05-01", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2026-05-02", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2026-05-03", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo", "best_of": 1, "map": "inferno"},
            {"date": "2026-05-04", "team1": "Bravo", "team2": "Alpha", "winner": "Bravo", "best_of": 3, "map": "inferno"},
            {"date": "2026-05-05", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "nuke"},
            {"date": "2026-05-06", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "nuke"},
        ]

        enriched = enrich_match_history(rows)

        self.assertAlmostEqual(enriched[-1]["h2h_team1_winrate"], 1 / 3)
        self.assertAlmostEqual(enriched[-1]["team1_h2h_winrate_vs_opponent"], 1 / 3)

    def test_enrich_matches_fills_missing_player_stats_from_prior_team_average(self):
        from cs2pickem.enrichment import enrich_match_history

        rows = raw_history()
        rows[-1]["team2_rating"] = None
        rows[-1]["team2_kd"] = None
        enriched = enrich_match_history(rows)
        last = enriched[-1]

        self.assertAlmostEqual(last["team2_rating"], 1.18)
        self.assertAlmostEqual(last["team2_kd"], 1.15)

    def test_team_profiles_capture_map_preferences_and_bans(self):
        from cs2pickem.enrichment import build_team_profiles

        profiles = build_team_profiles(raw_history())

        self.assertIn("Alpha", profiles)
        self.assertEqual(profiles["Alpha"]["prefer_top3"][0], "mirage")
        self.assertIn("map_winrates", profiles["Alpha"])
        self.assertLessEqual(len(profiles["Alpha"]["ban_top3"]), 3)

    def test_enriched_rolling_fields_are_consumed_by_feature_builder(self):
        from cs2pickem.enrichment import enrich_match_history
        from cs2pickem.features import FeatureBuilder

        enriched = enrich_match_history(raw_history())
        dataset = FeatureBuilder().fit_transform(enriched)

        self.assertIn("recent_winrate_5_diff", dataset.feature_names)
        self.assertIn("matches_30d_diff", dataset.feature_names)
        self.assertIn("current_streak_diff", dataset.feature_names)

    def test_enrich_csv_workflow_writes_training_rows_and_profiles(self):
        from cs2pickem.data import read_matches_csv
        from cs2pickem.pipeline import enrich_matches_file

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "raw.csv")
            output_path = os.path.join(tmpdir, "enriched.csv")
            profiles_path = os.path.join(tmpdir, "profiles.json")
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "date,event,event_tier,status,team1,team2,winner,best_of,map,team1_rank,team2_rank,"
                    "team1_rmr_points,team2_rmr_points,team1_rating,team2_rating,team1_kd,team2_kd\n"
                )
                for row in raw_history():
                    handle.write(
                        f"{row['date']},{row['event']},{row['event_tier']},{row['status']},{row['team1']},{row['team2']},"
                        f"{row['winner']},{row['best_of']},{row['map']},{row['team1_rank']},{row['team2_rank']},"
                        f"{row['team1_rmr_points']},{row['team2_rmr_points']},{row['team1_rating']},{row['team2_rating']},"
                        f"{row['team1_kd']},{row['team2_kd']}\n"
                    )

            manifest = enrich_matches_file(input_path, output_path, profiles_path)
            enriched = read_matches_csv(output_path)

        self.assertEqual(manifest["rows"], 6)
        self.assertEqual(manifest["profiles"], 3)
        self.assertIn("team1_recent_winrate_5", enriched[0])
        self.assertIn("team2_current_streak", enriched[-1])


class WinrateKnobTests(unittest.TestCase):
    """Three independent, default-off winrate refinements: time decay, strength-of-schedule,
    and bayesian shrinkage. Each must be separately switchable and leakage-free."""

    def _decay_history(self):
        # Alpha: 3 ancient wins, then 1 recent loss. Recency weighting should pull the
        # recent-5 winrate below the unweighted 3/4 = 0.75 toward the recent loss.
        return [
            {"date": "2025-01-01", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2025-01-08", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2025-01-15", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2026-06-01", "team1": "Alpha", "team2": "Charlie", "winner": "Charlie", "best_of": 1, "map": "mirage"},
            {"date": "2026-06-05", "team1": "Alpha", "team2": "Delta", "winner": "Delta", "best_of": 1, "map": "mirage"},
        ]

    def test_default_call_unchanged_when_all_knobs_off(self):
        from cs2pickem.enrichment import enrich_match_history

        rows = self._decay_history()
        baseline = enrich_match_history(rows)
        explicit_off = enrich_match_history(
            rows,
            winrate_half_life_days=None,
            enable_sos=False,
            winrate_shrinkage_pseudocount=0.0,
        )
        for a, b in zip(baseline, explicit_off):
            self.assertAlmostEqual(a["team1_recent_winrate_5"], b["team1_recent_winrate_5"])

    def test_time_decay_downweights_old_results_monotonically(self):
        from cs2pickem.enrichment import enrich_match_history

        rows = self._decay_history()
        no_decay = enrich_match_history(rows)[-1]["team1_recent_winrate_5"]
        # Pre-match for the last row: Alpha has 3 old wins + 1 recent loss = 3/4 unweighted.
        self.assertAlmostEqual(no_decay, 0.75)

        # Shorter half-life weights the recent loss more -> winrate drops monotonically.
        long_hl = enrich_match_history(rows, winrate_half_life_days=400.0)[-1]["team1_recent_winrate_5"]
        short_hl = enrich_match_history(rows, winrate_half_life_days=30.0)[-1]["team1_recent_winrate_5"]
        self.assertLess(long_hl, no_decay)
        self.assertLess(short_hl, long_hl)

    def test_time_decay_does_not_read_future(self):
        from cs2pickem.enrichment import enrich_match_history

        rows = self._decay_history()
        baseline = enrich_match_history(rows, winrate_half_life_days=90.0)[-1]["team1_recent_winrate_5"]
        with_future = list(rows) + [
            {"date": "2026-07-01", "team1": "Alpha", "team2": "Echo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
        ]
        enriched = enrich_match_history(with_future, winrate_half_life_days=90.0)
        same = next(r for r in enriched if r["date"] == "2026-06-05")
        self.assertAlmostEqual(same["team1_recent_winrate_5"], baseline)

    def test_strength_of_schedule_rewards_beating_strong_opponents(self):
        from cs2pickem.enrichment import enrich_match_history

        # Strong opponent (Strong) builds a big Elo edge by beating filler teams; weak
        # opponent (Weak) loses to filler. Hero beats Strong once and Weak once.
        rows = []
        for i in range(8):
            rows.append({"date": f"2026-01-{i+1:02d}", "team1": "Strong", "team2": f"Filler{i}", "winner": "Strong", "best_of": 1, "map": "mirage"})
            rows.append({"date": f"2026-01-{i+1:02d}", "team1": f"Filler{i}", "team2": "Weak", "winner": f"Filler{i}", "best_of": 1, "map": "mirage"})
        rows.append({"date": "2026-02-01", "team1": "Hero", "team2": "Strong", "winner": "Hero", "best_of": 1, "map": "mirage"})
        rows.append({"date": "2026-02-02", "team1": "Hero", "team2": "Weak", "winner": "Hero", "best_of": 1, "map": "mirage"})
        # Final fixture to read Hero's pre-match SoS-weighted winrate.
        rows.append({"date": "2026-02-03", "team1": "Hero", "team2": "Foxtrot", "winner": "Hero", "best_of": 1, "map": "mirage"})

        plain = enrich_match_history(rows)[-1]["team1_recent_winrate_5"]
        weighted = enrich_match_history(rows, enable_sos=True, sos_elo_scale=400.0)[-1]["team1_recent_winrate_5"]
        # Hero won every game either way (both 1.0), but SoS does not exceed 1.0.
        self.assertLessEqual(weighted, 1.0)
        self.assertGreaterEqual(weighted, 0.0)
        # When mixing a win over a strong team with a loss, SoS weighting raises the
        # winrate relative to a loss against an equally strong team -> covered below.

    def test_strength_of_schedule_weights_quality_wins_above_quality_losses(self):
        from cs2pickem.enrichment import enrich_match_history

        # Hero beats a Strong team but loses to a Weak team. SoS should value the
        # quality win more than the cheap loss, pushing the winrate above the
        # unweighted 0.5.
        rows = []
        for i in range(8):
            rows.append({"date": f"2026-01-{i+1:02d}", "team1": "Strong", "team2": f"F{i}", "winner": "Strong", "best_of": 1, "map": "mirage"})
            rows.append({"date": f"2026-01-{i+1:02d}", "team1": f"F{i}", "team2": "Weak", "winner": f"F{i}", "best_of": 1, "map": "mirage"})
        rows.append({"date": "2026-02-01", "team1": "Hero", "team2": "Strong", "winner": "Hero", "best_of": 1, "map": "mirage"})
        rows.append({"date": "2026-02-02", "team1": "Hero", "team2": "Weak", "winner": "Weak", "best_of": 1, "map": "mirage"})
        rows.append({"date": "2026-02-03", "team1": "Hero", "team2": "Zulu", "winner": "Hero", "best_of": 1, "map": "mirage"})

        plain = enrich_match_history(rows)[-1]["team1_recent_winrate_5"]
        weighted = enrich_match_history(rows, enable_sos=True, sos_elo_scale=400.0)[-1]["team1_recent_winrate_5"]
        self.assertAlmostEqual(plain, 0.5)
        self.assertGreater(weighted, plain)

    def test_bayesian_shrinkage_pulls_small_samples_toward_half(self):
        from cs2pickem.enrichment import enrich_match_history

        # Alpha wins its only prior game -> raw recent winrate 1.0. Shrinkage toward
        # the 0.5 prior must pull a 1-game sample below 1.0.
        rows = [
            {"date": "2026-05-01", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2026-05-02", "team1": "Alpha", "team2": "Charlie", "winner": "Alpha", "best_of": 1, "map": "mirage"},
        ]
        raw = enrich_match_history(rows)[-1]["team1_recent_winrate_5"]
        shrunk = enrich_match_history(rows, winrate_shrinkage_pseudocount=4.0)[-1]["team1_recent_winrate_5"]
        self.assertAlmostEqual(raw, 1.0)
        self.assertLess(shrunk, 1.0)
        self.assertGreater(shrunk, 0.5)

    def test_bayesian_shrinkage_converges_to_raw_as_sample_grows(self):
        from cs2pickem.enrichment import enrich_match_history

        # Many wins -> shrunk winrate approaches the raw winrate (here ~1.0 limited to last 5).
        rows = [
            {"date": f"2026-05-{d:02d}", "team1": "Alpha", "team2": f"Opp{d}", "winner": "Alpha", "best_of": 1, "map": "mirage"}
            for d in range(1, 12)
        ]
        small = enrich_match_history(rows[:2], winrate_shrinkage_pseudocount=4.0)[-1]["team1_recent_winrate_5"]
        large = enrich_match_history(rows, winrate_shrinkage_pseudocount=4.0)[-1]["team1_recent_winrate_5"]
        # Larger samples are shrunk less, so the winrate sits closer to the raw 1.0.
        self.assertGreater(large, small)

    def test_knobs_are_independent(self):
        from cs2pickem.enrichment import enrich_match_history

        rows = self._decay_history()
        # Turning on only shrinkage leaves the decay path identical to no-decay.
        only_shrink = enrich_match_history(rows, winrate_shrinkage_pseudocount=2.0)
        decay_and_shrink = enrich_match_history(
            rows, winrate_shrinkage_pseudocount=2.0, winrate_half_life_days=30.0
        )
        # Decay changes the result on top of shrinkage -> the two differ.
        self.assertNotAlmostEqual(
            only_shrink[-1]["team1_recent_winrate_5"],
            decay_and_shrink[-1]["team1_recent_winrate_5"],
        )

    def test_sos_weight_is_mean_one_so_it_does_not_amplify_shrinkage(self):
        from cs2pickem.enrichment import enrich_match_history

        # Hero plays only average-Elo opponents (everyone starts at ELO_BASE and the
        # first game pre-match Elo is exactly the base), so each strength-of-schedule
        # weight is ~1.0. With the mean-1 normalisation, SoS must NOT shrink the
        # effective sample size: SoS+shrinkage should land very close to shrinkage
        # alone. The old 0.5-centred weight halved the effective N and would pull the
        # combined statistic noticeably further toward 0.5 -> the two knobs would
        # cross-talk. Locking near-equality keeps them orthogonal (red-line a).
        rows = [
            {"date": f"2026-05-{d:02d}", "team1": "Hero", "team2": f"Opp{d}", "winner": "Hero", "best_of": 1, "map": "mirage"}
            for d in range(1, 4)
        ]
        shrink_only = enrich_match_history(rows, winrate_shrinkage_pseudocount=4.0)[-1]["team1_recent_winrate_5"]
        shrink_and_sos = enrich_match_history(
            rows, winrate_shrinkage_pseudocount=4.0, enable_sos=True, sos_elo_scale=400.0
        )[-1]["team1_recent_winrate_5"]
        # Average opponents -> SoS weight ~1.0 each -> effective N unchanged -> the
        # shrunk statistic is essentially identical with or without SoS.
        self.assertAlmostEqual(shrink_only, shrink_and_sos, places=2)


if __name__ == "__main__":
    unittest.main()
