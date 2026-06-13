import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class SeriesWinProbHomogeneousTests(unittest.TestCase):
    def test_best_of_1_is_identity(self):
        from cs2pickem.series import series_win_prob

        for p in (0.0, 0.25, 0.5, 0.73, 1.0):
            self.assertAlmostEqual(series_win_prob(p, 1), p)

    def test_best_of_3_closed_form_identity(self):
        from cs2pickem.series import series_win_prob

        # BO3 first-to-2: p^2 * (3 - 2p)
        for p in (0.1, 0.3, 0.5, 0.62, 0.88):
            expected = p * p * (3.0 - 2.0 * p)
            self.assertAlmostEqual(series_win_prob(p, 3), expected)

    def test_best_of_5_closed_form_identity(self):
        from cs2pickem.series import series_win_prob

        # BO5 first-to-3: p^3 * (10 - 15p + 6p^2)
        for p in (0.1, 0.3, 0.5, 0.62, 0.88):
            expected = p ** 3 * (10.0 - 15.0 * p + 6.0 * p * p)
            self.assertAlmostEqual(series_win_prob(p, 5), expected)

    def test_fair_coin_maps_to_fair_series(self):
        from cs2pickem.series import series_win_prob

        for best_of in (1, 3, 5, 7):
            self.assertAlmostEqual(series_win_prob(0.5, best_of), 0.5)

    def test_certain_map_win_gives_certain_series(self):
        from cs2pickem.series import series_win_prob

        for best_of in (1, 3, 5):
            self.assertAlmostEqual(series_win_prob(1.0, best_of), 1.0)
            self.assertAlmostEqual(series_win_prob(0.0, best_of), 0.0)

    def test_monotone_in_map_probability(self):
        from cs2pickem.series import series_win_prob

        previous = -1.0
        for p in [i / 20.0 for i in range(21)]:
            value = series_win_prob(p, 5)
            self.assertGreaterEqual(value + 1e-12, previous)
            previous = value

    def test_best_of_7_closed_form(self):
        from cs2pickem.series import series_win_prob

        # First-to-4 of 7. Sum over k=0..3 of C(3+k,k) p^4 (1-p)^k
        from math import comb

        for p in (0.2, 0.5, 0.77):
            expected = sum(comb(3 + k, k) * p ** 4 * (1.0 - p) ** k for k in range(4))
            self.assertAlmostEqual(series_win_prob(p, 7), expected)


class SeriesWinProbHeterogeneousTests(unittest.TestCase):
    def test_heterogeneous_paths_sum_to_one(self):
        from cs2pickem.series import series_win_prob

        # Probability of team1 winning the series + team2 winning = 1.
        for maps in ([0.6, 0.4, 0.55], [0.3, 0.8, 0.5], [0.1, 0.9, 0.45]):
            win_team1 = series_win_prob(maps, 3)
            win_team2 = series_win_prob([1.0 - q for q in maps], 3)
            self.assertAlmostEqual(win_team1 + win_team2, 1.0)

    def test_heterogeneous_best_of_5_paths_sum_to_one(self):
        from cs2pickem.series import series_win_prob

        maps = [0.6, 0.4, 0.55, 0.3, 0.7]
        win_team1 = series_win_prob(maps, 5)
        win_team2 = series_win_prob([1.0 - q for q in maps], 5)
        self.assertAlmostEqual(win_team1 + win_team2, 1.0)

    def test_heterogeneous_equal_probs_match_closed_form(self):
        from cs2pickem.series import series_win_prob

        # When every per-map probability equals p, the ordered enumeration must
        # reproduce the homogeneous closed form.
        for p in (0.3, 0.5, 0.71):
            self.assertAlmostEqual(series_win_prob([p, p, p], 3), p * p * (3.0 - 2.0 * p))
            self.assertAlmostEqual(
                series_win_prob([p, p, p, p, p], 5),
                p ** 3 * (10.0 - 15.0 * p + 6.0 * p * p),
            )

    def test_scalar_and_list_agree(self):
        from cs2pickem.series import series_win_prob

        self.assertAlmostEqual(series_win_prob(0.62, 3), series_win_prob([0.62, 0.62, 0.62], 3))

    def test_too_few_maps_raises(self):
        from cs2pickem.series import series_win_prob

        with self.assertRaises(ValueError):
            series_win_prob([0.6, 0.4], 3)  # BO3 needs at least 3 ordered maps

    def test_probabilities_clamped(self):
        from cs2pickem.series import series_win_prob

        # Out-of-range inputs are clamped to [0, 1] rather than producing junk.
        self.assertAlmostEqual(series_win_prob(1.5, 3), 1.0)
        self.assertAlmostEqual(series_win_prob(-0.2, 3), 0.0)


class ScoreDistributionTests(unittest.TestCase):
    def test_best_of_3_scores_sum_to_one(self):
        from cs2pickem.series import score_distribution

        for p in (0.1, 0.4, 0.5, 0.83):
            dist = score_distribution(p, 3)
            self.assertAlmostEqual(sum(dist.values()), 1.0)

    def test_best_of_3_score_components(self):
        from cs2pickem.series import score_distribution

        p = 0.62
        dist = score_distribution(p, 3)
        self.assertAlmostEqual(dist["2-0"], p * p)
        self.assertAlmostEqual(dist["2-1"], 2.0 * p * p * (1.0 - p))
        self.assertAlmostEqual(dist["1-2"], 2.0 * (1.0 - p) ** 2 * p)
        self.assertAlmostEqual(dist["0-2"], (1.0 - p) ** 2)

    def test_winning_scores_sum_to_series_win(self):
        from cs2pickem.series import score_distribution, series_win_prob

        # 2-0 + 2-1 must equal series_win for BO3.
        for p in (0.2, 0.5, 0.77):
            dist = score_distribution(p, 3)
            self.assertAlmostEqual(dist["2-0"] + dist["2-1"], series_win_prob(p, 3))

    def test_best_of_5_winning_scores_sum_to_series_win(self):
        from cs2pickem.series import score_distribution, series_win_prob

        for p in (0.2, 0.5, 0.77):
            dist = score_distribution(p, 5)
            won = dist["3-0"] + dist["3-1"] + dist["3-2"]
            self.assertAlmostEqual(won, series_win_prob(p, 5))

    def test_best_of_5_scores_sum_to_one(self):
        from cs2pickem.series import score_distribution

        for p in (0.1, 0.4, 0.5, 0.83):
            dist = score_distribution(p, 5)
            self.assertAlmostEqual(sum(dist.values()), 1.0)

    def test_best_of_1_score_distribution(self):
        from cs2pickem.series import score_distribution

        dist = score_distribution(0.7, 1)
        self.assertAlmostEqual(dist["1-0"], 0.7)
        self.assertAlmostEqual(dist["0-1"], 0.3)
        self.assertAlmostEqual(sum(dist.values()), 1.0)

    def test_heterogeneous_score_distribution_sums_to_one(self):
        from cs2pickem.series import score_distribution

        dist = score_distribution([0.6, 0.4, 0.55], 3)
        self.assertAlmostEqual(sum(dist.values()), 1.0)

    def test_heterogeneous_score_distribution_matches_series_win(self):
        from cs2pickem.series import score_distribution, series_win_prob

        maps = [0.6, 0.4, 0.55]
        dist = score_distribution(maps, 3)
        self.assertAlmostEqual(dist["2-0"] + dist["2-1"], series_win_prob(maps, 3))


class MapsVetoWeightingTests(unittest.TestCase):
    def test_average_unknown_map_prediction_backward_compatible_default(self):
        # Default behavior (no best_of / no series weighting) is unchanged: a flat
        # average over candidate maps, locking the historic contract.
        from cs2pickem.maps import average_unknown_map_prediction

        team1 = {"ban_top3": [], "prefer_top3": ["mirage", "inferno"], "map_winrates": {"mirage": 0.7, "inferno": 0.6}}
        team2 = {"ban_top3": [], "prefer_top3": ["mirage", "inferno"], "map_winrates": {"mirage": 0.4, "inferno": 0.5}}
        seen = []

        def predictor(row):
            seen.append((row["map"], row["team1_map_winrate"], row["team2_map_winrate"]))
            return row["team1_map_winrate"] - row["team2_map_winrate"] + 0.5

        result = average_unknown_map_prediction(
            {"team1": "Alpha", "team2": "Bravo", "best_of": 1},
            team1,
            team2,
            predictor,
            map_pool=["mirage", "inferno"],
            top_n=2,
        )

        self.assertEqual(result["candidate_maps"], ["mirage", "inferno"])
        self.assertEqual(seen, [("mirage", 0.7, 0.4), ("inferno", 0.6, 0.5)])
        self.assertAlmostEqual(result["average_probability_team1"], 0.7)

    def test_veto_weighted_average_is_convex_combination(self):
        from cs2pickem.maps import average_unknown_map_prediction

        team1 = {"ban_top3": [], "prefer_top3": ["mirage", "inferno"], "map_winrates": {"mirage": 0.7, "inferno": 0.6}}
        team2 = {"ban_top3": [], "prefer_top3": ["mirage", "inferno"], "map_winrates": {"mirage": 0.4, "inferno": 0.5}}

        def predictor(row):
            return row["team1_map_winrate"] - row["team2_map_winrate"] + 0.5

        result = average_unknown_map_prediction(
            {"team1": "Alpha", "team2": "Bravo"},
            team1,
            team2,
            predictor,
            map_pool=["mirage", "inferno"],
            top_n=2,
            veto_weighted=True,
        )

        per_map = result["per_map_probability_team1"]
        weights = result["map_weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        # The weighted average lies within the convex hull of the per-map values.
        self.assertGreaterEqual(result["average_probability_team1"], min(per_map.values()) - 1e-9)
        self.assertLessEqual(result["average_probability_team1"], max(per_map.values()) + 1e-9)

    def test_series_win_probability_field_present_when_best_of_given(self):
        from cs2pickem.maps import average_unknown_map_prediction
        from cs2pickem.series import series_win_prob

        team1 = {"ban_top3": [], "prefer_top3": ["mirage", "inferno"], "map_winrates": {"mirage": 0.7, "inferno": 0.6}}
        team2 = {"ban_top3": [], "prefer_top3": ["mirage", "inferno"], "map_winrates": {"mirage": 0.4, "inferno": 0.5}}

        def predictor(row):
            return 0.6  # constant per-map win prob

        result = average_unknown_map_prediction(
            {"team1": "Alpha", "team2": "Bravo"},
            team1,
            team2,
            predictor,
            map_pool=["mirage", "inferno"],
            top_n=2,
            best_of=3,
        )

        # Constant per-map prob 0.6 -> BO3 series win should match the closed form.
        self.assertAlmostEqual(result["series_win_probability_team1"], series_win_prob(0.6, 3))
        self.assertIn("series_score_distribution", result)
        self.assertAlmostEqual(sum(result["series_score_distribution"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
