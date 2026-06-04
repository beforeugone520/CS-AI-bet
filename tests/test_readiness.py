import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def rows(count=12):
    output = []
    for index in range(count):
        row = _base_row(index)
        row.update(
            {
                "team1_rmr_points": 900,
                "team2_rmr_points": 520,
                "team1_major_best_placement": 4,
                "team2_major_best_placement": 16,
                "team1_matches_30d": 8,
                "team2_matches_30d": 6,
                "team1_recent_winrate_5": 0.8,
                "team2_recent_winrate_5": 0.4,
                "team1_bo1_winrate_6m": 0.74,
                "team2_bo1_winrate_6m": 0.45,
                "team1_bo3_winrate_6m": 0.76,
                "team2_bo3_winrate_6m": 0.44,
                "team1_map_winrate": 0.68,
                "team2_map_winrate": 0.42,
                "team1_kd": 1.08,
                "team2_kd": 0.98,
                "team1_opening_success": 0.55,
                "team2_opening_success": 0.48,
                "team1_clutch_winrate": 0.6,
                "team2_clutch_winrate": 0.47,
                "team1_star_rating": 1.25,
                "team2_star_rating": 1.05,
                "team1_substitute_flag": 0,
                "team2_substitute_flag": 0,
                "team1_player_sample": 5,
                "team2_player_sample": 5,
                "h2h_team1_winrate": 0.6,
                "swiss_round": 1 + (index % 5),
                "team1_wins": index % 3,
                "team1_losses": (index + 1) % 3,
                "team2_wins": (index + 2) % 3,
                "team2_losses": index % 3,
                "team1_current_streak": 1,
                "team2_current_streak": -1,
                "version_tag": "pre-cologne",
            }
        )
        output.append(row)
    return output


def minimal_rows_missing_objective_fields(count=12):
    return [_base_row(index) for index in range(count)]


def _base_row(index):
    return {
        "date": f"2026-05-{(index % 28) + 1:02d}",
        "event": "IEM Cologne Qualifier",
        "event_tier": "S",
        "status": "completed",
        "team1": f"Team{index % 6}",
        "team2": f"Team{(index + 1) % 6}",
        "winner": f"Team{index % 6}",
        "best_of": 1 if index % 2 == 0 else 3,
        "map": "mirage",
        "team1_rank": 5,
        "team2_rank": 20,
        "team1_recent_winrate_10": 0.7,
        "team2_recent_winrate_10": 0.4,
        "team1_rating": 1.1,
        "team2_rating": 1.0,
        "odds_team1": 1.7,
        "odds_team2": 2.1,
    }


def training_report():
    return {
        "cleaned_matches": 12,
        "segment_metrics": {
            "BO1": {"accuracy": 0.7, "auc": 0.74, "log_loss": 0.5, "profit_loss": 1.0},
            "BO3": {"accuracy": 0.8, "auc": 0.82, "log_loss": 0.45, "profit_loss": 1.2},
        },
        "model_comparison": {
            "logistic": {"accuracy": 0.65, "auc": 0.7, "log_loss": 0.6, "profit_loss": 0.1},
            "random_forest": {"accuracy": 0.66, "auc": 0.71, "log_loss": 0.58, "profit_loss": 0.2},
            "xgboost": {"accuracy": 0.67, "auc": 0.72, "log_loss": 0.57, "profit_loss": 0.3},
            "neural_network": {"accuracy": 0.64, "auc": 0.69, "log_loss": 0.62, "profit_loss": 0.0},
            "ensemble": {"accuracy": 0.72, "auc": 0.76, "log_loss": 0.5, "profit_loss": 0.5},
        },
    }


class ReadinessTests(unittest.TestCase):
    def test_audit_readiness_reports_failures_for_small_unproven_dataset(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(rows(12), training_report(), minimum_rows=8000, required_teams=80)

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["minimum_rows"]["passed"])
        self.assertFalse(report["checks"]["team_coverage"]["passed"])
        self.assertTrue(report["checks"]["bo1_performance"]["passed"])
        self.assertTrue(report["checks"]["bo3_performance"]["passed"])
        self.assertTrue(report["checks"]["ensemble_beats_single_models"]["passed"])
        self.assertIn("minimum_rows", report["failed_checks"])

    def test_audit_readiness_can_pass_with_relaxed_demo_thresholds(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(rows(12), training_report(), minimum_rows=10, required_teams=6)

        self.assertTrue(report["ready"])
        self.assertEqual(report["failed_checks"], [])

    def test_audit_readiness_requires_full_objective_modeling_fields(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(minimal_rows_missing_objective_fields(12), training_report(), minimum_rows=10, required_teams=6)

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["required_fields"]["passed"])
        missing = report["checks"]["required_fields"]["actual"]
        self.assertIn("team1_rmr_points", missing)
        self.assertIn("team2_rmr_points", missing)
        self.assertIn("team1_bo1_winrate_6m", missing)
        self.assertIn("team2_bo3_winrate_6m", missing)
        self.assertIn("team1_map_winrate", missing)
        self.assertIn("team2_kd", missing)
        self.assertIn("team1_opening_success", missing)
        self.assertIn("team2_clutch_winrate", missing)
        self.assertIn("team1_star_rating", missing)
        self.assertIn("h2h_team1_winrate", missing)
        self.assertIn("version_tag", missing)

    def test_audit_readiness_requires_high_quality_collection_scope(self):
        from cs2pickem.readiness import audit_readiness

        contaminated = rows(12)
        contaminated[0]["event_tier"] = "B"
        contaminated[1]["date"] = "2025-01-01"
        contaminated[2]["status"] = "forfeit"
        contaminated[3]["team1"] = "Team Academy"
        contaminated[4]["team2"] = "Team Mix"

        report = audit_readiness(
            contaminated,
            training_report(),
            minimum_rows=10,
            required_teams=6,
            sample_reference_date="2026-05-31",
            maximum_sample_age_days=180,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["data_quality_scope"]["passed"])
        actual = report["checks"]["data_quality_scope"]["actual"]
        self.assertEqual(actual["invalid_tier_rows"], 1)
        self.assertEqual(actual["stale_rows"], 1)
        self.assertEqual(actual["invalid_status_rows"], 1)
        self.assertEqual(actual["secondary_team_rows"], 1)
        self.assertEqual(actual["temporary_team_rows"], 1)
        self.assertIn("data_quality_scope", report["failed_checks"])

    def test_audit_readiness_accepts_whitespace_padded_event_tiers(self):
        from cs2pickem.readiness import audit_readiness

        padded = rows(12)
        padded[0]["event_tier"] = " s "
        padded[1]["event_tier"] = "\tA\n"

        report = audit_readiness(padded, training_report(), minimum_rows=10, required_teams=6)

        self.assertTrue(report["checks"]["data_quality_scope"]["passed"])
        actual = report["checks"]["data_quality_scope"]["actual"]
        self.assertEqual(actual["invalid_tier_rows"], 0)

    def test_audit_readiness_can_require_cologne_calendar_split_and_window(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            expected_train_end_date="2026-04-30",
            expected_validation_end_date="2026-05-15",
            minimum_max_age_days=180,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["calendar_split"]["passed"])
        self.assertFalse(report["checks"]["freshness_window"]["passed"])
        self.assertIn("calendar_split", report["failed_checks"])
        self.assertIn("freshness_window", report["failed_checks"])

        compliant_report = training_report()
        compliant_report["max_age_days"] = 180
        compliant_report["split_strategy"] = "date_boundaries"
        compliant_report["split_boundaries"] = {
            "train_end_date": "2026-04-30",
            "validation_end_date": "2026-05-15",
        }
        compliant = audit_readiness(
            rows(12),
            compliant_report,
            minimum_rows=10,
            required_teams=6,
            expected_train_end_date="2026-04-30",
            expected_validation_end_date="2026-05-15",
            minimum_max_age_days=180,
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["calendar_split"]["passed"])
        self.assertTrue(compliant["checks"]["freshness_window"]["passed"])

    def test_audit_readiness_can_require_validation_tuned_ensemble_weights(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            require_validation_tuned_weights=True,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["validation_tuned_weights"]["passed"])
        self.assertEqual(report["checks"]["validation_tuned_weights"]["actual"]["basis"], None)
        self.assertIn("validation_tuned_weights", report["failed_checks"])

        tuned_report = training_report()
        tuned_report["validation_tuned_ensemble_weights"] = {
            "basis": "validation_log_loss",
            "validation_count": 6,
            "weights": {"logistic": 0.2, "random_forest": 0.3, "xgboost": 0.35, "neural_network": 0.15},
            "model_log_loss": {"logistic": 0.6, "random_forest": 0.55, "xgboost": 0.52, "neural_network": 0.62},
        }
        compliant = audit_readiness(
            rows(12),
            tuned_report,
            minimum_rows=10,
            required_teams=6,
            require_validation_tuned_weights=True,
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["validation_tuned_weights"]["passed"])
        self.assertAlmostEqual(compliant["checks"]["validation_tuned_weights"]["actual"]["weight_sum"], 1.0)

    def test_audit_readiness_can_require_player_status_features(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            minimum_player_status_features=2,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["player_status_features"]["passed"])
        self.assertEqual(report["checks"]["player_status_features"]["actual"]["selected_count"], 0)
        self.assertIn("player_status_features", report["failed_checks"])

        status_ready = training_report()
        status_ready["feature_selection"] = {
            "required_features": {
                "requested": ["rating_diff", "player_form_score_diff", "player_sample_confidence_diff"],
                "available": ["rating_diff", "player_form_score_diff"],
                "selected": ["rating_diff", "player_form_score_diff"],
                "unavailable": ["player_sample_confidence_diff"],
            }
        }
        compliant = audit_readiness(
            rows(12),
            status_ready,
            minimum_rows=10,
            required_teams=6,
            minimum_player_status_features=2,
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["player_status_features"]["passed"])
        self.assertEqual(compliant["checks"]["player_status_features"]["actual"]["selected_count"], 2)

    def test_readiness_file_workflow_reads_csv_and_report_json(self):
        from cs2pickem.data import write_matches_csv
        from cs2pickem.readiness import audit_readiness_file

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "matches.csv")
            report_path = os.path.join(tmpdir, "train.json")
            pickem_path = os.path.join(tmpdir, "pickem.json")
            write_matches_csv(data_path, rows(12))
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(training_report(), handle)
            with open(pickem_path, "w", encoding="utf-8") as handle:
                json.dump({"simulations": 100000}, handle)

            report = audit_readiness_file(
                data_path,
                report_path,
                minimum_rows=10,
                required_teams=6,
                pickem_report_path=pickem_path,
                minimum_pickem_simulations=100000,
                required_pickem_slots={"3-0": 0, "advance": 0, "0-3": 0},
            )

        self.assertTrue(report["ready"])
        self.assertIn("required_fields", report["checks"])
        self.assertTrue(report["checks"]["pickem_simulations"]["passed"])
        self.assertTrue(report["checks"]["pickem_slots"]["passed"])

    def test_audit_readiness_checks_major_participant_and_top_team_lists(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            participant_teams=["Team0", "Team1", "Team9"],
            top_teams=["Team0", "Team1", "Team2", "Team7"],
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["participant_coverage"]["passed"])
        self.assertFalse(report["checks"]["top_team_coverage"]["passed"])
        self.assertEqual(report["checks"]["participant_coverage"]["actual"]["missing"], ["Team9"])
        self.assertEqual(report["checks"]["top_team_coverage"]["actual"]["covered"], 3)

    def test_audit_readiness_can_require_historical_pickem_backtest_pass_rate(self):
        from cs2pickem.readiness import audit_readiness

        weak_backtest = {
            "cases": 3,
            "passed_cases": 1,
            "pass_rate": 1 / 3,
            "pass_rate_target": 0.38,
            "meets_pass_rate_target": False,
        }
        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_backtest_report=weak_backtest,
            pickem_pass_rate_target=0.38,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["pickem_backtest_pass_rate"]["passed"])
        self.assertIn("pickem_backtest_pass_rate", report["failed_checks"])

        strong_backtest = dict(weak_backtest)
        strong_backtest.update({"passed_cases": 2, "pass_rate": 2 / 3, "meets_pass_rate_target": True})
        compliant = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_backtest_report=strong_backtest,
            pickem_pass_rate_target=0.38,
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["pickem_backtest_pass_rate"]["passed"])

    def test_audit_readiness_can_require_pickem_monte_carlo_simulation_count(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report={"simulations": 25000},
            minimum_pickem_simulations=100000,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["pickem_simulations"]["passed"])
        self.assertEqual(report["checks"]["pickem_simulations"]["actual"], 25000)
        self.assertIn("pickem_simulations", report["failed_checks"])

        compliant = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report={"simulations": 100000},
            minimum_pickem_simulations=100000,
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["pickem_simulations"]["passed"])

    def test_audit_readiness_can_require_complete_pickem_answer_slots(self):
        from cs2pickem.readiness import audit_readiness

        incomplete = {
            "simulations": 100000,
            "pickems": {
                "3-0": ["Alpha", "Bravo"],
                "advance": ["Charlie", "Delta", "Echo"],
                "0-3": ["Foxtrot", "Golf"],
            },
        }
        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report=incomplete,
            required_pickem_slots={"3-0": 2, "advance": 6, "0-3": 2},
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["pickem_slots"]["passed"])
        self.assertEqual(report["checks"]["pickem_slots"]["actual"]["advance"], 3)
        self.assertIn("pickem_slots", report["failed_checks"])

        complete = {
            "simulations": 100000,
            "pickems": {
                "3-0": ["Alpha", "Bravo"],
                "advance": ["Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"],
                "0-3": ["India", "Juliet"],
            },
        }
        compliant = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report=complete,
            required_pickem_slots={"3-0": 2, "advance": 6, "0-3": 2},
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["pickem_slots"]["passed"])

        unknown_team_report = {
            "simulations": 100000,
            "team_probabilities": {team: {} for team in ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet"]},
            "pickems": {
                "3-0": ["Alpha", "Bravo"],
                "advance": ["Charlie", "Delta", "Echo", "Foxtrot", "Golf", "GhostTeam"],
                "0-3": ["India", "Juliet"],
            },
        }
        unknown = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report=unknown_team_report,
            required_pickem_slots={"3-0": 2, "advance": 6, "0-3": 2},
        )

        self.assertFalse(unknown["ready"])
        self.assertFalse(unknown["checks"]["pickem_slots"]["passed"])
        self.assertEqual(unknown["checks"]["pickem_slots"]["actual"]["unknown_teams"], ["GhostTeam"])

    def test_audit_readiness_can_require_pickem_selection_margin(self):
        from cs2pickem.readiness import audit_readiness

        narrow_margin_report = {
            "pickem_details": {
                "3-0": [
                    {"team": "Alpha", "category": "3-0", "probability": 0.31, "selection_margin": 0.025},
                    {"team": "Bravo", "category": "3-0", "probability": 0.29, "selection_margin": 0.05},
                ],
                "advance": [
                    {"team": "Charlie", "category": "advance", "probability": 0.76, "selection_margin": 0.08},
                ],
                "0-3": [
                    {"team": "Delta", "category": "0-3", "probability": 0.43, "selection_margin": 0.06},
                ],
            }
        }

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report=narrow_margin_report,
            minimum_pickem_selection_margin=0.04,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["pickem_selection_margin"]["passed"])
        self.assertEqual(report["checks"]["pickem_selection_margin"]["actual"]["minimum_margin"], 0.025)
        self.assertEqual(report["checks"]["pickem_selection_margin"]["actual"]["low_margin_picks"][0]["team"], "Alpha")
        self.assertIn("pickem_selection_margin", report["failed_checks"])

        wide_margin_report = {
            "pickem_details": {
                "3-0": [{"team": "Alpha", "category": "3-0", "probability": 0.35, "selection_margin": 0.06}],
                "advance": [{"team": "Charlie", "category": "advance", "probability": 0.76, "selection_margin": 0.08}],
                "0-3": [{"team": "Delta", "category": "0-3", "probability": 0.43, "selection_margin": 0.05}],
            }
        }
        compliant = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report=wide_margin_report,
            minimum_pickem_selection_margin=0.04,
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["pickem_selection_margin"]["passed"])

    def test_audit_readiness_can_require_pickem_market_adjustment_coverage(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report={"market_adjustment_summary": {"cached_matchups": 3, "adjusted_matchups": 0}},
            minimum_pickem_market_adjusted_matchups=1,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["pickem_market_adjustment"]["passed"])
        self.assertEqual(report["checks"]["pickem_market_adjustment"]["actual"]["adjusted_matchups"], 0)
        self.assertIn("pickem_market_adjustment", report["failed_checks"])

        compliant = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            pickem_report={"market_adjustment_summary": {"cached_matchups": 3, "adjusted_matchups": 1}},
            minimum_pickem_market_adjusted_matchups=1,
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["pickem_market_adjustment"]["passed"])

    def test_audit_readiness_can_require_forecast_low_confidence_avoidance(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            forecast_report={
                "predictions": [
                    {"team1": "Alpha", "team2": "Bravo", "confidence_margin": 0.015, "low_confidence": True, "pick": "Alpha"},
                    {"team1": "Charlie", "team2": "Delta", "confidence_margin": 0.04, "low_confidence": False, "pick": "Charlie"},
                ]
            },
            require_forecast_low_confidence_avoidance=True,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["forecast_low_confidence_avoidance"]["passed"])
        self.assertEqual(report["checks"]["forecast_low_confidence_avoidance"]["actual"]["low_confidence_non_avoids"][0]["team1"], "Alpha")
        self.assertIn("forecast_low_confidence_avoidance", report["failed_checks"])

        compliant = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            forecast_report={
                "predictions": [
                    {"team1": "Alpha", "team2": "Bravo", "confidence_margin": 0.015, "low_confidence": True, "pick": "avoid"},
                    {"team1": "Charlie", "team2": "Delta", "confidence_margin": 0.04, "low_confidence": False, "pick": "Charlie"},
                ]
            },
            require_forecast_low_confidence_avoidance=True,
        )

        self.assertTrue(compliant["ready"])
        self.assertTrue(compliant["checks"]["forecast_low_confidence_avoidance"]["passed"])

    def test_audit_readiness_can_require_fresh_source_manifests(self):
        from cs2pickem.readiness import audit_readiness

        report = audit_readiness(
            rows(12),
            training_report(),
            minimum_rows=10,
            required_teams=6,
            source_manifests=[
                {"source": "hltv-results", "updated_at": "2026-05-31T22:00:00+00:00"},
                {"source": "player-stats", "updated_at": "2026-05-30T20:00:00+00:00"},
            ],
            required_sources=["hltv-results", "player-stats", "bp-intel"],
            source_reference_time="2026-06-01T00:00:00+00:00",
            maximum_source_age_hours=24,
        )

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["source_freshness"]["passed"])
        actual = report["checks"]["source_freshness"]["actual"]
        self.assertEqual(actual["fresh_sources"], ["hltv-results"])
        self.assertEqual(actual["stale_sources"], [{"source": "player-stats", "age_hours": 28.0}])
        self.assertEqual(actual["missing_sources"], ["bp-intel"])
        self.assertIn("source_freshness", report["failed_checks"])

    def test_readiness_cli_accepts_source_manifest_freshness_gate(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_json, write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            training_report_path = os.path.join(tmpdir, "training.json")
            manifest_path = os.path.join(tmpdir, "source_manifest.json")
            output_path = os.path.join(tmpdir, "readiness.json")
            write_matches_csv(matches_path, rows(12))
            write_json(training_report_path, training_report())
            write_json(manifest_path, {"source": "hltv-results", "updated_at": "2026-05-31T10:00:00+00:00"})

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "readiness",
                "--matches",
                matches_path,
                "--training-report",
                training_report_path,
                "--minimum-rows",
                "10",
                "--required-teams",
                "6",
                "--source-manifest",
                manifest_path,
                "--required-source",
                "hltv-results",
                "--source-reference-time",
                "2026-06-01T00:00:00+00:00",
                "--maximum-source-age-hours",
                "24",
                "--output",
                output_path,
            ]
            try:
                exit_code = main()
            finally:
                sys.argv = old_argv

            with open(output_path, encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["checks"]["source_freshness"]["passed"])

    def test_readiness_cli_accepts_player_status_feature_gate(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_json, write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            training_report_path = os.path.join(tmpdir, "training.json")
            output_path = os.path.join(tmpdir, "readiness.json")
            write_matches_csv(matches_path, rows(12))
            write_json(training_report_path, training_report())

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "readiness",
                "--matches",
                matches_path,
                "--training-report",
                training_report_path,
                "--minimum-rows",
                "10",
                "--required-teams",
                "6",
                "--minimum-player-status-features",
                "1",
                "--output",
                output_path,
            ]
            try:
                exit_code = main()
            finally:
                sys.argv = old_argv

            with open(output_path, encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertFalse(report["checks"]["player_status_features"]["passed"])
        self.assertIn("player_status_features", report["failed_checks"])

    def test_readiness_file_expands_daily_update_manifest_job_reports(self):
        from cs2pickem.data import write_json, write_matches_csv
        from cs2pickem.readiness import audit_readiness_file

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            training_report_path = os.path.join(tmpdir, "training.json")
            job_manifest_path = os.path.join(tmpdir, "01-hltv-manifest.json")
            daily_manifest_path = os.path.join(tmpdir, "daily_update_manifest.json")
            write_matches_csv(matches_path, rows(12))
            write_json(training_report_path, training_report())
            write_json(job_manifest_path, {"source": "hltv-results", "updated_at": "2026-05-31T22:00:00+00:00"})
            write_json(
                daily_manifest_path,
                {
                    "updated_at": "2026-05-31T23:00:00+00:00",
                    "job_reports": [{"name": "hltv", "manifest_path": job_manifest_path}],
                },
            )

            report = audit_readiness_file(
                matches_path,
                training_report_path,
                minimum_rows=10,
                required_teams=6,
                source_manifest_paths=[daily_manifest_path],
                required_sources=["hltv-results"],
                source_reference_time="2026-06-01T00:00:00+00:00",
                maximum_source_age_hours=24,
            )

        self.assertTrue(report["checks"]["source_freshness"]["passed"])
        self.assertEqual(report["checks"]["source_freshness"]["actual"]["fresh_sources"], ["hltv-results"])

    def test_readiness_cli_accepts_pickem_market_adjustment_gate(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_json, write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            training_report_path = os.path.join(tmpdir, "training.json")
            pickem_report_path = os.path.join(tmpdir, "pickem.json")
            output_path = os.path.join(tmpdir, "readiness.json")
            write_matches_csv(matches_path, rows(12))
            write_json(training_report_path, training_report())
            write_json(pickem_report_path, {"market_adjustment_summary": {"cached_matchups": 2, "adjusted_matchups": 0}})

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "readiness",
                "--matches",
                matches_path,
                "--training-report",
                training_report_path,
                "--minimum-rows",
                "10",
                "--required-teams",
                "6",
                "--pickem-report",
                pickem_report_path,
                "--minimum-pickem-market-adjusted-matchups",
                "1",
                "--output",
                output_path,
            ]
            try:
                self.assertEqual(main(), 0)
            finally:
                sys.argv = old_argv

            with open(output_path, encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertFalse(report["checks"]["pickem_market_adjustment"]["passed"])
        self.assertIn("pickem_market_adjustment", report["failed_checks"])

    def test_readiness_cli_accepts_forecast_low_confidence_gate(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_json, write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            training_report_path = os.path.join(tmpdir, "training.json")
            forecast_report_path = os.path.join(tmpdir, "forecast.json")
            output_path = os.path.join(tmpdir, "readiness.json")
            write_matches_csv(matches_path, rows(12))
            write_json(training_report_path, training_report())
            write_json(
                forecast_report_path,
                {"predictions": [{"team1": "Alpha", "team2": "Bravo", "confidence_margin": 0.01, "low_confidence": True, "pick": "Alpha"}]},
            )

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "readiness",
                "--matches",
                matches_path,
                "--training-report",
                training_report_path,
                "--minimum-rows",
                "10",
                "--required-teams",
                "6",
                "--forecast-report",
                forecast_report_path,
                "--require-forecast-low-confidence-avoidance",
                "--output",
                output_path,
            ]
            try:
                self.assertEqual(main(), 0)
            finally:
                sys.argv = old_argv

            with open(output_path, encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertFalse(report["checks"]["forecast_low_confidence_avoidance"]["passed"])
        self.assertIn("forecast_low_confidence_avoidance", report["failed_checks"])

    def test_readiness_file_workflow_accepts_team_list_csvs(self):
        from cs2pickem.data import write_matches_csv
        from cs2pickem.readiness import audit_readiness_file

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "matches.csv")
            report_path = os.path.join(tmpdir, "train.json")
            participants_path = os.path.join(tmpdir, "participants.csv")
            top_teams_path = os.path.join(tmpdir, "top.csv")
            write_matches_csv(data_path, rows(12))
            write_matches_csv(participants_path, [{"team": f"Team{index}"} for index in range(6)])
            write_matches_csv(top_teams_path, [{"team": f"Team{index}"} for index in range(6)])
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(training_report(), handle)

            report = audit_readiness_file(
                data_path,
                report_path,
                minimum_rows=10,
                required_teams=6,
                participants_path=participants_path,
                top_teams_path=top_teams_path,
            )

        self.assertTrue(report["ready"])
        self.assertTrue(report["checks"]["participant_coverage"]["passed"])
        self.assertTrue(report["checks"]["top_team_coverage"]["passed"])


if __name__ == "__main__":
    unittest.main()
