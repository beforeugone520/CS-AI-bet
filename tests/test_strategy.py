import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class StageStrategyTests(unittest.TestCase):
    def test_single_match_pick_requires_probability_strictly_above_52_percent(self):
        from cs2pickem.strategy import single_match_pick

        self.assertEqual(single_match_pick(0.52, "Alpha", "Bravo"), "avoid")
        self.assertEqual(single_match_pick(0.48, "Alpha", "Bravo"), "avoid")
        self.assertEqual(single_match_pick(0.5201, "Alpha", "Bravo"), "Alpha")
        self.assertEqual(single_match_pick(0.4799, "Alpha", "Bravo"), "Bravo")

    def test_challengers_stage_prefers_bo1_map_profile_when_probabilities_are_close(self):
        from cs2pickem.strategy import choose_pickems

        probabilities = {
            "RankedCore": {"3-0": 0.30, "advance": 0.76, "0-3": 0.02, "eliminate": 0.24},
            "MapSpecialist": {"3-0": 0.285, "advance": 0.74, "0-3": 0.03, "eliminate": 0.26},
            "LowSeed": {"3-0": 0.05, "advance": 0.35, "0-3": 0.42, "eliminate": 0.65},
        }
        rankings = {"RankedCore": 5, "MapSpecialist": 12, "LowSeed": 40}
        team_features = {
            "RankedCore": {"bo1_winrate_6m": 0.54, "map_depth": 0.50, "rating": 1.08},
            "MapSpecialist": {"bo1_winrate_6m": 0.78, "map_depth": 0.82, "rating": 1.03},
            "LowSeed": {"bo1_winrate_6m": 0.35, "map_depth": 0.30, "rating": 0.93},
        }

        pickems = choose_pickems(
            probabilities,
            rankings=rankings,
            slots={"3-0": 1, "advance": 1, "0-3": 1},
            stage="challengers",
            team_features=team_features,
        )

        self.assertEqual(pickems["3-0"], ["MapSpecialist"])

    def test_legends_stage_prefers_elite_strength_over_bo1_specialist(self):
        from cs2pickem.strategy import choose_pickems

        probabilities = {
            "EliteCore": {"3-0": 0.30, "advance": 0.82, "0-3": 0.01, "eliminate": 0.18},
            "MapSpecialist": {"3-0": 0.295, "advance": 0.74, "0-3": 0.03, "eliminate": 0.26},
            "LowSeed": {"3-0": 0.05, "advance": 0.35, "0-3": 0.42, "eliminate": 0.65},
        }
        rankings = {"EliteCore": 3, "MapSpecialist": 14, "LowSeed": 42}
        team_features = {
            "EliteCore": {"bo1_winrate_6m": 0.55, "map_depth": 0.52, "rating": 1.18},
            "MapSpecialist": {"bo1_winrate_6m": 0.79, "map_depth": 0.82, "rating": 1.02},
            "LowSeed": {"bo1_winrate_6m": 0.35, "map_depth": 0.30, "rating": 0.93},
        }

        pickems = choose_pickems(
            probabilities,
            rankings=rankings,
            slots={"3-0": 1, "advance": 1, "0-3": 1},
            stage="legends",
            team_features=team_features,
        )

        self.assertEqual(pickems["3-0"], ["EliteCore"])

    def test_upset_constraint_uses_rank_gap_not_absolute_rank_cutoff(self):
        from cs2pickem.strategy import choose_pickems

        probabilities = {
            "EliteLongshot": {"3-0": 0.02, "advance": 0.70, "0-3": 0.02},
            "BorderlineUpside": {"3-0": 0.30, "advance": 0.76, "0-3": 0.04},
            "SafeCore": {"3-0": 0.24, "advance": 0.72, "0-3": 0.05},
        }
        rankings = {"EliteLongshot": 1, "BorderlineUpside": 16, "SafeCore": 14}

        pickems = choose_pickems(
            probabilities,
            rankings=rankings,
            slots={"3-0": 1, "advance": 1, "0-3": 0},
            upset_rank_limit=15,
        )

        self.assertEqual(pickems["3-0"], ["BorderlineUpside"])

    def test_player_form_downweights_low_sample_substitute_pickem_candidates(self):
        from cs2pickem.strategy import choose_pickems, describe_pickem_risk

        probabilities = {
            "StableCore": {"3-0": 0.31, "advance": 0.74, "0-3": 0.03},
            "FragileFavorite": {"3-0": 0.32, "advance": 0.75, "0-3": 0.04},
        }
        rankings = {"StableCore": 8, "FragileFavorite": 9}
        team_features = {
            "StableCore": {
                "player_form_score": 0.10,
                "player_form_trend": 0.02,
                "player_sample_confidence": 1.0,
                "substitute_flag": 0,
            },
            "FragileFavorite": {
                "player_form_score": 0.01,
                "player_form_trend": -0.06,
                "player_sample_confidence": 0.2,
                "substitute_flag": 1,
            },
        }

        pickems = choose_pickems(
            probabilities,
            rankings=rankings,
            slots={"3-0": 1, "advance": 0, "0-3": 0},
            team_features=team_features,
        )
        risk_details = describe_pickem_risk(probabilities, rankings=rankings, team_features=team_features)
        fragile = next(entry for entry in risk_details["3-0"] if entry["team"] == "FragileFavorite")

        self.assertEqual(pickems["3-0"], ["StableCore"])
        self.assertLess(fragile["player_form_adjustment"], 0.0)
        self.assertLess(fragile["player_availability_multiplier"], 1.0)

    def test_describe_pickem_risk_reports_stage_and_upset_adjustments(self):
        from cs2pickem.strategy import describe_pickem_risk

        probabilities = {
            "Elite": {"3-0": 0.10, "advance": 0.70, "0-3": 0.02},
            "HighRiskUpset": {"3-0": 0.40, "advance": 0.62, "0-3": 0.08},
        }
        rankings = {"Elite": 1, "HighRiskUpset": 30}
        team_features = {
            "Elite": {"bo1_winrate_6m": 0.52, "map_depth": 0.50},
            "HighRiskUpset": {"bo1_winrate_6m": 0.80, "map_depth": 0.80},
        }

        details = describe_pickem_risk(
            probabilities,
            rankings=rankings,
            stage="challengers",
            team_features=team_features,
            upset_rank_limit=15,
        )

        upset = next(entry for entry in details["3-0"] if entry["team"] == "HighRiskUpset")
        self.assertEqual(upset["rank"], 30)
        self.assertEqual(upset["upset_rank_gap"], 29)
        self.assertEqual(upset["upset_penalty_multiplier"], 0.75)
        self.assertGreater(upset["stage_adjustment"], 0.0)
        self.assertAlmostEqual(upset["final_score"], (0.40 + upset["stage_adjustment"]) * 0.75)

    def test_describe_pickems_reports_probability_rank_and_next_candidate_margin(self):
        from cs2pickem.strategy import choose_pickems, describe_pickems

        probabilities = {
            "Alpha": {"3-0": 0.42, "advance": 0.91, "0-3": 0.01},
            "Bravo": {"3-0": 0.30, "advance": 0.82, "0-3": 0.04},
            "Charlie": {"3-0": 0.18, "advance": 0.62, "0-3": 0.12},
            "Delta": {"3-0": 0.08, "advance": 0.31, "0-3": 0.46},
        }
        rankings = {"Alpha": 1, "Bravo": 8, "Charlie": 20, "Delta": 35}
        pickems = choose_pickems(probabilities, rankings=rankings, slots={"3-0": 1, "advance": 2, "0-3": 1})

        details = describe_pickems(probabilities, pickems, rankings=rankings)

        self.assertEqual(details["3-0"][0]["team"], "Alpha")
        self.assertEqual(details["3-0"][0]["category"], "3-0")
        self.assertEqual(details["3-0"][0]["rank"], 1)
        self.assertAlmostEqual(details["3-0"][0]["probability"], 0.42)
        self.assertAlmostEqual(details["3-0"][0]["next_best_probability"], 0.30)
        self.assertAlmostEqual(details["3-0"][0]["selection_margin"], 0.12)
        self.assertEqual(details["advance"][0]["team"], pickems["advance"][0])
        self.assertEqual(len(details["advance"]), 2)
        self.assertEqual(details["0-3"][0]["team"], "Delta")

    def test_describe_pickems_can_use_strategy_scores_for_selection_margin(self):
        from cs2pickem.strategy import choose_pickems, describe_pickem_risk, describe_pickems

        probabilities = {
            "Alpha": {"3-0": 0.44, "advance": 0.95, "0-3": 0.01},
            "Bravo": {"3-0": 0.30, "advance": 0.81, "0-3": 0.03},
            "Charlie": {"3-0": 0.08, "advance": 0.72, "0-3": 0.08},
            "Delta": {"3-0": 0.02, "advance": 0.12, "0-3": 0.50},
        }
        rankings = {"Alpha": 1, "Bravo": 8, "Charlie": 18, "Delta": 35}
        pickems = choose_pickems(probabilities, rankings=rankings, slots={"3-0": 1, "advance": 1, "0-3": 1})
        risk_details = describe_pickem_risk(probabilities, rankings=rankings)

        details = describe_pickems(probabilities, pickems, rankings=rankings, risk_details=risk_details)

        self.assertEqual(pickems, {"3-0": ["Alpha"], "advance": ["Bravo"], "0-3": ["Delta"]})
        self.assertEqual(details["advance"][0]["team"], "Bravo")
        self.assertAlmostEqual(details["advance"][0]["selection_score"], 0.81)
        self.assertAlmostEqual(details["advance"][0]["next_best_score"], 0.72 * 0.75)
        self.assertAlmostEqual(details["advance"][0]["selection_margin"], 0.81 - 0.72 * 0.75)


if __name__ == "__main__":
    unittest.main()
