import math
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class MarketFusionTests(unittest.TestCase):
    def test_default_fusion_method_is_legacy_clip_and_unchanged(self):
        from cs2pickem.strategy import adjust_probability_toward_market_probability

        # Default path = historic arithmetic +/-0.03 truncation toward the market.
        self.assertAlmostEqual(
            adjust_probability_toward_market_probability(0.60, 0.90),
            0.63,
        )
        self.assertAlmostEqual(
            adjust_probability_toward_market_probability(0.60, 0.40),
            0.57,
        )
        # Within the cap the model simply moves to the market value.
        self.assertAlmostEqual(
            adjust_probability_toward_market_probability(0.60, 0.61),
            0.61,
        )
        # Explicitly naming legacy_clip is identical to the default.
        self.assertEqual(
            adjust_probability_toward_market_probability(0.60, 0.90),
            adjust_probability_toward_market_probability(0.60, 0.90, fusion_method="legacy_clip"),
        )

    def test_legacy_clip_respects_custom_max_adjustment_and_clips_to_unit_interval(self):
        from cs2pickem.strategy import adjust_probability_toward_market_probability

        self.assertAlmostEqual(
            adjust_probability_toward_market_probability(0.50, 0.90, max_adjustment=0.10),
            0.60,
        )
        self.assertEqual(
            adjust_probability_toward_market_probability(0.99, 0.10, max_adjustment=0.50),
            0.49,
        )
        self.assertGreaterEqual(adjust_probability_toward_market_probability(0.02, 0.0, max_adjustment=0.5), 0.0)
        self.assertLessEqual(adjust_probability_toward_market_probability(0.98, 1.0, max_adjustment=0.5), 1.0)

    def test_logit_pool_matches_log_odds_average_with_frozen_default_weight(self):
        from cs2pickem.strategy import DEFAULT_MODEL_WEIGHT, adjust_probability_toward_market_probability

        p_model, p_market = 0.60, 0.80
        w = DEFAULT_MODEL_WEIGHT
        expected_logit = w * math.log(p_model / (1 - p_model)) + (1 - w) * math.log(p_market / (1 - p_market))
        expected = 1.0 / (1.0 + math.exp(-expected_logit))
        fused = adjust_probability_toward_market_probability(p_model, p_market, fusion_method="logit_pool")
        self.assertAlmostEqual(fused, expected, places=9)
        # Pro-market default => fused lands between the model and the market.
        self.assertGreater(fused, p_model)
        self.assertLess(fused, p_market)

    def test_logit_pool_weight_boundaries_recover_each_expert(self):
        from cs2pickem.strategy import adjust_probability_toward_market_probability

        # w = 1 -> pure model; w = 0 -> pure market (each clipped to the unit interval).
        self.assertAlmostEqual(
            adjust_probability_toward_market_probability(0.62, 0.81, fusion_method="logit_pool", model_weight=1.0),
            0.62,
            places=6,
        )
        self.assertAlmostEqual(
            adjust_probability_toward_market_probability(0.62, 0.81, fusion_method="logit_pool", model_weight=0.0),
            0.81,
            places=6,
        )
        # Out-of-range weights are clamped (boundary check, not an MLE), not raised.
        self.assertAlmostEqual(
            adjust_probability_toward_market_probability(0.62, 0.81, fusion_method="logit_pool", model_weight=5.0),
            0.62,
            places=6,
        )
        self.assertAlmostEqual(
            adjust_probability_toward_market_probability(0.62, 0.81, fusion_method="logit_pool", model_weight=-3.0),
            0.81,
            places=6,
        )

    def test_logit_pool_never_overflows_on_extreme_probabilities(self):
        from cs2pickem.strategy import adjust_probability_toward_market_probability

        for p_model in (0.0, 1.0, 0.5):
            for p_market in (0.0, 1.0, 0.5):
                for w in (0.0, 0.35, 1.0):
                    fused = adjust_probability_toward_market_probability(
                        p_model, p_market, fusion_method="logit_pool", model_weight=w
                    )
                    self.assertGreater(fused, 0.0)
                    self.assertLess(fused, 1.0)

    def test_logit_pool_consumes_devigged_fair_prob_from_odds(self):
        from cs2pickem.odds import devig_market
        from cs2pickem.strategy import adjust_probability_toward_market_probability

        audit = devig_market(1.50, 2.80)
        fair = audit["fair_prob_team1"]
        # The de-vigged fair prob (margin removed) is the market input, not the raw
        # implied 1/odds. The fused value must lie strictly between the experts.
        self.assertLess(fair, 1.0 / 1.50)
        fused = adjust_probability_toward_market_probability(0.55, fair, fusion_method="logit_pool")
        self.assertGreater(fused, min(0.55, fair))
        self.assertLess(fused, max(0.55, fair))

    def test_adjust_probability_with_market_supports_logit_pool(self):
        from cs2pickem.odds import devig_market
        from cs2pickem.strategy import adjust_probability_with_market

        fused = adjust_probability_with_market(0.55, 1.50, 2.80, fusion_method="logit_pool")
        fair = devig_market(1.50, 2.80)["fair_prob_team1"]
        expected = adjust_probability_with_market(0.55, 1.50, 2.80, fusion_method="logit_pool")
        # adjust_probability_with_market should de-vig internally to the same fair prob.
        from cs2pickem.strategy import adjust_probability_toward_market_probability

        self.assertAlmostEqual(
            fused,
            adjust_probability_toward_market_probability(0.55, fair, fusion_method="logit_pool"),
            places=9,
        )
        self.assertEqual(fused, expected)
        # Single source of truth: adjust_probability_with_market must consume the
        # SAME de-vigged fair prob as odds.devig_market (no second hard-coded
        # de-vig copy that could silently drift if the default method changes).
        self.assertEqual(
            fused,
            adjust_probability_toward_market_probability(0.55, fair, fusion_method="logit_pool"),
        )

    def test_unknown_fusion_method_raises(self):
        from cs2pickem.strategy import adjust_probability_toward_market_probability

        with self.assertRaises(ValueError):
            adjust_probability_toward_market_probability(0.6, 0.7, fusion_method="bogus")


class StageStrategyTests(unittest.TestCase):
    def test_single_match_pick_requires_probability_strictly_above_52_percent(self):
        from cs2pickem.strategy import single_match_pick

        self.assertEqual(single_match_pick(0.52, "Alpha", "Bravo"), "avoid")
        self.assertEqual(single_match_pick(0.48, "Alpha", "Bravo"), "avoid")
        self.assertEqual(single_match_pick(0.5201, "Alpha", "Bravo"), "Alpha")
        self.assertEqual(single_match_pick(0.4799, "Alpha", "Bravo"), "Bravo")

    def test_single_match_pick_accepts_custom_minimum_margin(self):
        from cs2pickem.strategy import single_match_pick

        self.assertEqual(single_match_pick(0.549, "Alpha", "Bravo", minimum_margin=0.05), "avoid")
        self.assertEqual(single_match_pick(0.551, "Alpha", "Bravo", minimum_margin=0.05), "Alpha")
        self.assertEqual(single_match_pick(0.449, "Alpha", "Bravo", minimum_margin=0.05), "Bravo")

    def test_single_match_pick_can_avoid_player_form_counter_signal(self):
        from cs2pickem.strategy import single_match_pick

        self.assertEqual(
            single_match_pick(
                0.61,
                "Alpha",
                "Bravo",
                minimum_margin=0.05,
                player_form_score_diff=-0.06,
                avoid_player_form_counter_signal=True,
            ),
            "avoid",
        )
        self.assertEqual(
            single_match_pick(
                0.39,
                "Alpha",
                "Bravo",
                minimum_margin=0.05,
                player_form_score_diff=0.06,
                avoid_player_form_counter_signal=True,
            ),
            "avoid",
        )
        self.assertEqual(
            single_match_pick(
                0.61,
                "Alpha",
                "Bravo",
                minimum_margin=0.05,
                player_form_score_diff=0.06,
                avoid_player_form_counter_signal=True,
            ),
            "Alpha",
        )

    def test_single_match_pick_counter_signal_can_require_sample_confidence(self):
        from cs2pickem.strategy import single_match_pick

        self.assertEqual(
            single_match_pick(
                0.61,
                "Alpha",
                "Bravo",
                minimum_margin=0.05,
                player_form_score_diff=-0.06,
                player_form_sample_confidence=0.2,
                player_form_counter_min_confidence=0.4,
                avoid_player_form_counter_signal=True,
            ),
            "Alpha",
        )
        self.assertEqual(
            single_match_pick(
                0.61,
                "Alpha",
                "Bravo",
                minimum_margin=0.05,
                player_form_score_diff=-0.06,
                player_form_sample_confidence=0.6,
                player_form_counter_min_confidence=0.4,
                avoid_player_form_counter_signal=True,
            ),
            "avoid",
        )

    def test_single_match_pick_can_require_extra_margin_for_player_status_risk(self):
        from cs2pickem.strategy import single_match_pick

        self.assertEqual(
            single_match_pick(
                0.57,
                "Alpha",
                "Bravo",
                minimum_margin=0.05,
                avoid_player_status_risk=True,
                player_status_min_confidence=0.4,
                player_status_min_margin=0.08,
                team1_player_sample_confidence=0.2,
                team1_substitute_flag=0,
                team2_player_sample_confidence=0.9,
                team2_substitute_flag=0,
            ),
            "avoid",
        )
        self.assertEqual(
            single_match_pick(
                0.61,
                "Alpha",
                "Bravo",
                minimum_margin=0.05,
                avoid_player_status_risk=True,
                player_status_min_confidence=0.4,
                player_status_min_margin=0.08,
                team1_player_sample_confidence=0.2,
                team1_substitute_flag=0,
            ),
            "Alpha",
        )
        self.assertEqual(
            single_match_pick(
                0.43,
                "Alpha",
                "Bravo",
                minimum_margin=0.05,
                avoid_player_status_risk=True,
                player_status_min_confidence=0.4,
                player_status_min_margin=0.08,
                team1_player_sample_confidence=0.9,
                team1_substitute_flag=0,
                team2_player_sample_confidence=0.9,
                team2_substitute_flag=1,
            ),
            "avoid",
        )

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
        self.assertEqual(fragile["player_sample_confidence"], 0.2)
        self.assertEqual(fragile["substitute_flag"], 1)
        self.assertEqual(fragile["player_form_score"], 0.01)
        self.assertEqual(fragile["player_form_trend"], -0.06)
        self.assertTrue(fragile["player_status_risk"])

    def test_three_zero_uses_stronger_player_status_penalty_than_advance(self):
        from cs2pickem.strategy import describe_pickem_risk

        probabilities = {
            "StableCore": {"3-0": 0.30, "advance": 0.74, "0-3": 0.03},
            "LowSampleUpside": {"3-0": 0.326, "advance": 0.78, "0-3": 0.04},
        }
        rankings = {"StableCore": 8, "LowSampleUpside": 9}
        team_features = {
            "StableCore": {"player_sample_confidence": 1.0, "substitute_flag": 0},
            "LowSampleUpside": {"player_sample_confidence": 0.2, "substitute_flag": 0},
        }

        details = describe_pickem_risk(probabilities, rankings=rankings, team_features=team_features)
        fragile_three_zero = next(entry for entry in details["3-0"] if entry["team"] == "LowSampleUpside")
        fragile_advance = next(entry for entry in details["advance"] if entry["team"] == "LowSampleUpside")
        stable_three_zero = next(entry for entry in details["3-0"] if entry["team"] == "StableCore")
        stable_advance = next(entry for entry in details["advance"] if entry["team"] == "StableCore")

        self.assertLess(
            fragile_three_zero["player_availability_multiplier"],
            fragile_advance["player_availability_multiplier"],
        )
        self.assertLess(fragile_three_zero["final_score"], stable_three_zero["final_score"])
        self.assertGreater(fragile_advance["final_score"], stable_advance["final_score"])

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


def _two_outcome_samples(advance_map, three_zero_map=None, zero_three_map=None):
    """Build joint-sample-shaped vectors from boolean maps per category.

    ``advance_map`` etc. map team -> list[bool], one entry per simulation. Missing
    categories default to all-False. Records are not needed by the pickem joint
    objectives (only the per-category booleans), so we leave a placeholder.
    """
    three_zero_map = three_zero_map or {}
    zero_three_map = zero_three_map or {}
    teams = set(advance_map) | set(three_zero_map) | set(zero_three_map)
    n = max((len(values) for values in advance_map.values()), default=0)
    samples = []
    for index in range(n):
        sample = {}
        for team in teams:
            sample[team] = {
                "record": "n/a",
                "advance": bool(advance_map.get(team, [False] * n)[index]),
                "eliminate": False,
                "3-0": bool(three_zero_map.get(team, [False] * n)[index]) if three_zero_map.get(team) else False,
                "3-1": False,
                "3-2": False,
                "0-3": bool(zero_three_map.get(team, [False] * n)[index]) if zero_three_map.get(team) else False,
                "1-3": False,
                "2-3": False,
            }
        samples.append(sample)
    return samples


class PickemObjectiveTests(unittest.TestCase):
    def _probabilities(self):
        return {
            "Alpha": {"3-0": 0.42, "advance": 0.91, "0-3": 0.01},
            "Bravo": {"3-0": 0.30, "advance": 0.82, "0-3": 0.04},
            "Charlie": {"3-0": 0.18, "advance": 0.62, "0-3": 0.12},
            "Delta": {"3-0": 0.08, "advance": 0.31, "0-3": 0.46},
        }

    def test_default_objective_is_expected_hits_and_reproduces_current_output(self):
        # Regression baseline (red line a): the named expected_hits objective and
        # the default path must be IDENTICAL to the historic choose_pickems output.
        from cs2pickem.strategy import DEFAULT_PICKEM_OBJECTIVE, choose_pickems

        self.assertEqual(DEFAULT_PICKEM_OBJECTIVE, "expected_hits")
        probabilities = self._probabilities()
        rankings = {"Alpha": 1, "Bravo": 8, "Charlie": 20, "Delta": 35}
        slots = {"3-0": 1, "advance": 2, "0-3": 1}

        baseline = choose_pickems(probabilities, rankings=rankings, slots=slots)
        named = choose_pickems(
            probabilities, rankings=rankings, slots=slots, objective="expected_hits"
        )
        self.assertEqual(named, baseline)
        # Alpha is consumed by the 3-0 slot, so cross-category dedup keeps it out
        # of advance (historic choose_pickems behaviour, locked here unchanged).
        self.assertEqual(baseline, {"3-0": ["Alpha"], "advance": ["Bravo", "Charlie"], "0-3": ["Delta"]})

    def test_unknown_objective_raises(self):
        from cs2pickem.strategy import choose_pickems

        with self.assertRaises(ValueError):
            choose_pickems(self._probabilities(), objective="bogus")

    def test_threshold_and_leveraged_require_joint_samples(self):
        from cs2pickem.strategy import choose_pickems

        for objective in ("threshold_prob", "leveraged"):
            with self.assertRaises(ValueError):
                choose_pickems(self._probabilities(), objective=objective)
            with self.assertRaises(ValueError):
                choose_pickems(self._probabilities(), objective=objective, joint_samples=[])

    def test_ticket_hits_in_sample_counts_correct_picks(self):
        from cs2pickem.strategy import ticket_hits_in_sample

        sample = {
            "Alpha": {"advance": True, "3-0": True, "0-3": False},
            "Bravo": {"advance": True, "3-0": False, "0-3": False},
            "Delta": {"advance": False, "3-0": False, "0-3": True},
        }
        ticket = {"3-0": ["Alpha"], "advance": ["Alpha", "Bravo"], "0-3": ["Delta"]}
        # Alpha-3-0 hit, Alpha-advance hit, Bravo-advance hit, Delta-0-3 hit => 4.
        self.assertEqual(ticket_hits_in_sample(ticket, sample), 4)
        # Flip Bravo advance to a miss => 3.
        sample["Bravo"]["advance"] = False
        self.assertEqual(ticket_hits_in_sample(ticket, sample), 3)

    def test_threshold_probability_uses_joint_samples_not_marginals(self):
        from cs2pickem.strategy import choose_pickems, ticket_threshold_probability

        # Two advance slots, three candidates. Solo has the HIGHEST marginal
        # advance rate (7/10) so the expected-hits seed picks it, but it is
        # perfectly ANTI-correlated with the Pair: whenever Solo advances neither
        # Pair member does. {Solo, Pair1} therefore NEVER scores 2 advance hits.
        # Pair1 and Pair2 always co-advance (6/10) => P(2 hits) = 0.6. A marginal
        # (expected_hits) optimiser cannot see this joint structure; the
        # threshold objective must, by reading the shared samples.
        T, F = True, False
        advance = {
            "Solo": [F, F, F, F, F, F, T, T, T, T],     # 4/10, disjoint from Pair
            "Pair1": [T, T, T, T, T, T, F, F, F, F],    # 6/10
            "Pair2": [T, T, T, T, T, T, F, F, F, F],    # identical to Pair1
        }
        samples = _two_outcome_samples(advance)
        probabilities = {
            # Marginal scores order Solo above the Pair so the EV seed takes Solo.
            "Solo": {"3-0": 0.0, "advance": 0.7, "0-3": 0.0},
            "Pair1": {"3-0": 0.0, "advance": 0.5, "0-3": 0.0},
            "Pair2": {"3-0": 0.0, "advance": 0.49, "0-3": 0.0},
        }
        slots = {"3-0": 0, "advance": 2, "0-3": 0}

        # Expected-hits seed (marginal) takes Solo first.
        seed = choose_pickems(probabilities, slots=slots, objective="expected_hits")
        self.assertIn("Solo", seed["advance"])
        self.assertAlmostEqual(ticket_threshold_probability(seed, samples, 2), 0.0)

        chosen = choose_pickems(
            probabilities,
            slots=slots,
            objective="threshold_prob",
            joint_samples=samples,
            threshold=2,
        )
        # The joint objective swaps to the co-advancing pair.
        self.assertEqual(set(chosen["advance"]), {"Pair1", "Pair2"})
        self.assertAlmostEqual(
            ticket_threshold_probability(chosen, samples, 2), 0.6
        )
        self.assertGreater(
            ticket_threshold_probability(chosen, samples, 2),
            ticket_threshold_probability(seed, samples, 2),
        )

    def test_evaluate_ticket_distribution_reports_samples_and_confidence(self):
        from cs2pickem.strategy import evaluate_ticket_distribution

        T, F = True, False
        advance = {"Pair1": [T, T, T, F], "Pair2": [T, T, T, F]}
        samples = _two_outcome_samples(advance)
        ticket = {"3-0": [], "advance": ["Pair1", "Pair2"], "0-3": []}

        summary = evaluate_ticket_distribution(ticket, samples, threshold=2)
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["total_picks"], 2)
        self.assertEqual(summary["threshold"], 2)
        # 3 of 4 samples have both picks correct.
        self.assertAlmostEqual(summary["threshold_probability"], 0.75)
        self.assertAlmostEqual(summary["expected_hits"], (2 + 2 + 2 + 0) / 4)
        self.assertEqual(summary["hit_histogram"], {0: 1, 2: 3})
        lo, hi = summary["threshold_probability_ci95"]
        self.assertLessEqual(lo, summary["threshold_probability"])
        self.assertGreaterEqual(hi, summary["threshold_probability"])

    def test_leveraged_objective_fades_chalk_toward_contrarian_pick(self):
        from cs2pickem.strategy import choose_pickems

        # One advance slot. Chalk advances in ALL sims (field always nails it ->
        # zero leverage). Contrarian advances in only some sims, but the field
        # rarely gets it -> high pool-share reward. Expected-hits picks Chalk;
        # leveraged should fade to the contrarian.
        T, F = True, False
        advance = {
            "Chalk": [T] * 10,
            "Contra": [T, F, T, F, F, F, F, F, F, F],  # 2/10 advance rate
        }
        samples = _two_outcome_samples(advance)
        probabilities = {
            "Chalk": {"3-0": 0.0, "advance": 0.99, "0-3": 0.0},
            "Contra": {"3-0": 0.0, "advance": 0.20, "0-3": 0.0},
        }
        slots = {"3-0": 0, "advance": 1, "0-3": 0}

        seed = choose_pickems(probabilities, slots=slots, objective="expected_hits")
        self.assertEqual(seed["advance"], ["Chalk"])

        leveraged = choose_pickems(
            probabilities,
            slots=slots,
            objective="leveraged",
            joint_samples=samples,
            leverage_strength=2.0,
        )
        self.assertEqual(leveraged["advance"], ["Contra"])

    def test_leveraged_with_zero_strength_collapses_to_expected_hits(self):
        from cs2pickem.strategy import choose_pickems

        T, F = True, False
        advance = {
            "Chalk": [T] * 10,
            "Contra": [T, F, T, F, F, F, F, F, F, F],
        }
        samples = _two_outcome_samples(advance)
        probabilities = {
            "Chalk": {"3-0": 0.0, "advance": 0.99, "0-3": 0.0},
            "Contra": {"3-0": 0.0, "advance": 0.20, "0-3": 0.0},
        }
        slots = {"3-0": 0, "advance": 1, "0-3": 0}

        # leverage_strength=0 removes the contrarian tilt: reward == expected hits,
        # so the higher-hit-rate Chalk wins again.
        leveraged = choose_pickems(
            probabilities,
            slots=slots,
            objective="leveraged",
            joint_samples=samples,
            leverage_strength=0.0,
        )
        self.assertEqual(leveraged["advance"], ["Chalk"])

    def test_joint_objectives_preserve_slot_counts_and_dedup(self):
        from cs2pickem.strategy import choose_pickems

        probabilities = {
            "Alpha": {"3-0": 0.42, "advance": 0.91, "0-3": 0.01},
            "Bravo": {"3-0": 0.30, "advance": 0.82, "0-3": 0.04},
            "Charlie": {"3-0": 0.18, "advance": 0.62, "0-3": 0.12},
            "Delta": {"3-0": 0.08, "advance": 0.31, "0-3": 0.46},
        }
        rankings = {"Alpha": 1, "Bravo": 8, "Charlie": 20, "Delta": 35}
        slots = {"3-0": 1, "advance": 2, "0-3": 1}
        T, F = True, False
        advance = {
            "Alpha": [T, T, T, T, F],
            "Bravo": [T, T, F, T, F],
            "Charlie": [F, T, T, F, T],
            "Delta": [F, F, F, T, T],
        }
        three_zero = {"Alpha": [T, F, T, F, F], "Bravo": [F, T, F, F, T]}
        zero_three = {"Delta": [T, T, F, T, T], "Charlie": [F, F, T, F, F]}
        samples = _two_outcome_samples(advance, three_zero, zero_three)

        for objective, kwargs in (
            ("threshold_prob", {"threshold": 3}),
            ("leveraged", {"leverage_strength": 1.5}),
        ):
            ticket = choose_pickems(
                probabilities,
                rankings=rankings,
                slots=slots,
                objective=objective,
                joint_samples=samples,
                **kwargs,
            )
            self.assertEqual(len(ticket["3-0"]), 1, objective)
            self.assertEqual(len(ticket["advance"]), 2, objective)
            self.assertEqual(len(ticket["0-3"]), 1, objective)
            all_teams = [t for teams in ticket.values() for t in teams]
            self.assertEqual(len(all_teams), len(set(all_teams)), objective)
            for team in all_teams:
                self.assertIn(team, probabilities, objective)

    def test_threshold_default_k_matches_flattened_picks_not_slot_sum(self):
        # WF-2E review fix: when a category cannot fill its slots, the default
        # threshold-search K must equal the ticket's flattened pick count (what
        # evaluate_ticket_distribution reports by default), NOT sum(slots). With
        # K=sum(slots) the search would optimise an unreachable threshold and the
        # reported distribution would use a different K.
        from cs2pickem.strategy import (
            _choose_pickems_joint,
            _flatten_ticket,
            evaluate_ticket_distribution,
        )

        T, F = True, False
        # Two co-advancing teams, but slots ask for THREE advance picks.
        advance = {"Pair1": [T, T, T, F], "Pair2": [T, T, T, F]}
        samples = _two_outcome_samples(advance)
        probabilities = {
            "Pair1": {"3-0": 0.0, "advance": 0.75, "0-3": 0.0},
            "Pair2": {"3-0": 0.0, "advance": 0.75, "0-3": 0.0},
        }
        slots = {"3-0": 0, "advance": 3, "0-3": 0}  # 3 slots, only 2 candidates
        seed = {"3-0": [], "advance": ["Pair1", "Pair2"], "0-3": []}

        chosen = _choose_pickems_joint(
            "threshold_prob",
            seed_ticket=seed,
            team_probabilities=probabilities,
            rankings={},
            slots=slots,
            upset_rank_limit=15,
            stage="default",
            team_features={},
            joint_samples=samples,
            threshold=None,  # exercise the default-K path
            crowd_probabilities=None,
            leverage_strength=1.0,
            max_swaps=64,
        )
        flattened = len(_flatten_ticket(chosen))
        self.assertEqual(flattened, 2)  # only two teams could fill the slots
        # Optimiser's default K and the distribution's default K must agree.
        distribution = evaluate_ticket_distribution(chosen, samples)
        self.assertEqual(distribution["threshold"], flattened)
        # And P(hits >= 2) is genuinely reachable (3/4 samples), not the
        # identically-zero P(hits >= sum(slots)=3) the old default targeted.
        self.assertAlmostEqual(distribution["threshold_probability"], 0.75)

    def test_leveraged_reward_floors_field_prob_at_inverse_sample_count(self):
        # WF-2E review fix: a longshot the field hits only 1/N times must not be
        # rewarded as if field_prob -> 0 (MC tail noise). The field probability is
        # floored at 1/N, so its weight is capped at (N)^strength rather than
        # exploding without bound.
        from cs2pickem.strategy import _leveraged_reward

        T, F = True, False
        n = 20
        # "Longshot" hits exactly once (1/20); field also hits exactly once.
        advance = {"Longshot": [T] + [F] * (n - 1)}
        samples = _two_outcome_samples(advance)
        field = {"Longshot": {"advance": 1.0 / n}}  # 1/N field hit rate
        ticket = {"3-0": [], "advance": ["Longshot"], "0-3": []}

        strength = 2.0
        reward = _leveraged_reward(ticket, samples, field, strength)
        # Floored weight = (1 / (1/N))^strength = N^strength; the single hit (1/N
        # of samples) yields reward = N^strength / N = N^(strength-1).
        self.assertAlmostEqual(reward, n ** (strength - 1.0))
        # A field_prob BELOW the floor would have produced a strictly larger
        # reward; flooring caps it at the value above.
        unfloored_field = {"Longshot": {"advance": 1.0 / (10 * n)}}
        unfloored = _leveraged_reward(ticket, samples, unfloored_field, strength)
        self.assertAlmostEqual(unfloored, reward)


if __name__ == "__main__":
    unittest.main()
