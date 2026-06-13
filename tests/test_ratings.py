import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class EloRatingTests(unittest.TestCase):
    def test_wins_raise_rating_losses_lower_it_and_stay_zero_sum(self):
        from cs2pickem.ratings import compute_elo_ratings

        matches = [
            {"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"},
            {"date": "2026-01-02", "team1": "A", "team2": "B", "winner": "A"},
        ]
        per_match, final = compute_elo_ratings(matches, base=1500.0, k=32.0)

        # First match: both teams enter at base (pre-match, no leakage)
        self.assertAlmostEqual(per_match[0]["team1_elo_pre"], 1500.0)
        self.assertAlmostEqual(per_match[0]["team2_elo_pre"], 1500.0)
        # Winner rises, loser falls, and Elo updates are zero-sum
        self.assertGreater(final["A"], 1500.0)
        self.assertLess(final["B"], 1500.0)
        self.assertAlmostEqual(final["A"] + final["B"], 3000.0, places=3)

    def test_pre_match_rating_excludes_that_match_result(self):
        from cs2pickem.ratings import compute_elo_ratings

        matches = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        per_match, final = compute_elo_ratings(matches, base=1500.0, k=32.0)

        self.assertEqual(per_match[0]["team1_elo_pre"], 1500.0)  # before the update
        self.assertNotEqual(final["A"], 1500.0)                  # after the update

    def test_beating_a_stronger_opponent_gains_more_rating(self):
        from cs2pickem.ratings import compute_elo_ratings

        match = [{"date": "2026-01-01", "team1": "A", "team2": "B", "winner": "A"}]
        _, equal = compute_elo_ratings(match, base=1500.0, k=32.0)
        _, weak = compute_elo_ratings(match, base=1500.0, k=32.0, initial_ratings={"B": 1200.0})

        self.assertGreater(equal["A"] - 1500.0, weak["A"] - 1500.0)


def _bt_prob(theta_a: float, theta_b: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-(theta_a - theta_b)))


class BradleyTerryTests(unittest.TestCase):
    def _round_robin(self, win_counts):
        """Build a result list from a {(winner, loser): n} mapping."""
        matches = []
        day = 1
        for (winner, loser), count in win_counts.items():
            for _ in range(count):
                matches.append(
                    {
                        "date": f"2026-01-{day:02d}",
                        "team1": winner,
                        "team2": loser,
                        "winner": winner,
                    }
                )
                day += 1
        return matches

    def test_returns_mean_centered_log_strengths(self):
        from cs2pickem.ratings import compute_bradley_terry

        matches = self._round_robin({("A", "B"): 4, ("A", "C"): 3, ("B", "C"): 2})
        theta = compute_bradley_terry(matches)

        # Every observed team gets a strength.
        self.assertEqual(set(theta), {"A", "B", "C"})
        # Mean-centered for identifiability.
        self.assertAlmostEqual(sum(theta.values()) / len(theta), 0.0, places=6)

    def test_stronger_team_gets_higher_theta(self):
        from cs2pickem.ratings import compute_bradley_terry

        # A dominates, C is dominated.
        matches = self._round_robin({("A", "B"): 8, ("A", "C"): 9, ("B", "C"): 7})
        matches += self._round_robin({("B", "A"): 2, ("C", "A"): 1, ("C", "B"): 3})
        theta = compute_bradley_terry(matches)

        self.assertGreater(theta["A"], theta["B"])
        self.assertGreater(theta["B"], theta["C"])

    def test_predicted_probability_matches_win_frequency_two_teams(self):
        from cs2pickem.ratings import compute_bradley_terry

        # For an isolated pair, the BT MLE reproduces the empirical pairwise win
        # frequency exactly: A wins 70 of 100, so P(A>B) -> 0.70.
        matches = self._round_robin({("A", "B"): 70, ("B", "A"): 30})
        theta = compute_bradley_terry(matches, ridge=0.0, max_iter=5000, tol=1e-12)

        prob_a_beats_b = _bt_prob(theta["A"], theta["B"])
        self.assertAlmostEqual(prob_a_beats_b, 0.70, delta=0.005)

    def test_mle_stationarity_observed_equals_expected_wins(self):
        from cs2pickem.ratings import compute_bradley_terry

        # The defining property of the BT MLE: for every team, observed wins equal
        # the model's expected wins summed over its games. This is the rigorous
        # "P(i>j) consistent with results" contract on a non-trivial 3-team graph.
        games = {("A", "B"): 100, ("A", "C"): 100, ("B", "C"): 100}
        win_counts = {("A", "B"): 70, ("B", "A"): 30, ("A", "C"): 50, ("C", "A"): 50, ("B", "C"): 50, ("C", "B"): 50}
        matches = self._round_robin(win_counts)
        theta = compute_bradley_terry(matches, ridge=0.0, max_iter=20000, tol=1e-13)

        import math

        pi = {team: math.exp(value) for team, value in theta.items()}
        observed = {"A": 120.0, "B": 80.0, "C": 100.0}  # total wins per team
        for team in ("A", "B", "C"):
            expected = 0.0
            for (i, j), n in games.items():
                if team == i:
                    expected += n * pi[i] / (pi[i] + pi[j])
                elif team == j:
                    expected += n * pi[j] / (pi[i] + pi[j])
            self.assertAlmostEqual(expected, observed[team], places=3)

    def test_converges_within_iteration_budget(self):
        from cs2pickem.ratings import compute_bradley_terry

        matches = self._round_robin({("A", "B"): 5, ("B", "C"): 5, ("C", "A"): 5})
        # tol-driven convergence: a tiny max_iter still returns finite values, and
        # a generous budget gives a stable fixed point (running more iters barely moves it).
        coarse = compute_bradley_terry(matches, max_iter=2, ridge=1e-3)
        fine = compute_bradley_terry(matches, max_iter=500, ridge=1e-3)
        for team in ("A", "B", "C"):
            self.assertTrue(math_isfinite(coarse[team]))
            self.assertTrue(math_isfinite(fine[team]))
        again = compute_bradley_terry(matches, max_iter=1000, ridge=1e-3)
        for team in ("A", "B", "C"):
            self.assertAlmostEqual(fine[team], again[team], places=5)

    def test_ridge_keeps_undefeated_team_finite(self):
        from cs2pickem.ratings import compute_bradley_terry

        # A is undefeated, D is winless: without shrinkage theta -> +/-inf.
        matches = self._round_robin({("A", "B"): 3, ("A", "C"): 3, ("B", "D"): 3, ("C", "D"): 3})
        theta = compute_bradley_terry(matches, ridge=1.0)
        for value in theta.values():
            self.assertTrue(math_isfinite(value))
        self.assertGreater(theta["A"], theta["D"])

    def test_prior_pulls_strengths_toward_supplied_center(self):
        from cs2pickem.ratings import compute_bradley_terry

        matches = self._round_robin({("A", "B"): 6, ("B", "A"): 4})
        weak_prior = compute_bradley_terry(matches, ridge=20.0)
        # Heavy ridge toward the uniform prior collapses the spread toward 0.
        spread = abs(weak_prior["A"] - weak_prior["B"])
        light = compute_bradley_terry(matches, ridge=0.01)
        self.assertGreater(abs(light["A"] - light["B"]), spread)


class BradleyTerryPerMapTests(unittest.TestCase):
    def _map_match(self, day, t1, t2, winner, map_name):
        return {
            "date": f"2026-02-{day:02d}",
            "team1": t1,
            "team2": t2,
            "winner": winner,
            "map": map_name,
        }

    def test_per_map_fit_returns_strengths_per_map(self):
        from cs2pickem.ratings import compute_map_bradley_terry

        matches = [
            self._map_match(1, "A", "B", "A", "mirage"),
            self._map_match(2, "A", "B", "A", "mirage"),
            self._map_match(3, "B", "A", "B", "inferno"),
            self._map_match(4, "B", "A", "B", "inferno"),
        ]
        per_map = compute_map_bradley_terry(matches, ridge=1.0)

        self.assertIn("mirage", per_map)
        self.assertIn("inferno", per_map)
        # A wins on mirage, loses on inferno -> map-specific orientation.
        self.assertGreater(per_map["mirage"]["A"], per_map["mirage"]["B"])
        self.assertGreater(per_map["inferno"]["B"], per_map["inferno"]["A"])

    def test_sparse_map_shrinks_toward_global(self):
        from cs2pickem.ratings import compute_bradley_terry, compute_map_bradley_terry

        # Dense overall history establishes A >> B globally.
        overall = []
        for day in range(1, 21):
            overall.append(self._map_match(day, "A", "B", "A", "mirage" if day % 2 else "inferno"))
        # On 'nuke' there is a single noisy game where B beats A.
        sparse = [self._map_match(25, "B", "A", "B", "nuke")]
        matches = overall + sparse

        global_theta = compute_bradley_terry(matches, ridge=1.0)
        per_map = compute_map_bradley_terry(matches, ridge=8.0)

        # Despite the lone nuke upset, heavy ridge keeps A above B on nuke
        # (shrunk toward the global prior where A dominates).
        self.assertGreater(per_map["nuke"]["A"], per_map["nuke"]["B"])
        # And the nuke gap is much smaller than the dense-map gap, reflecting shrinkage.
        nuke_gap = per_map["nuke"]["A"] - per_map["nuke"]["B"]
        mirage_gap = per_map["mirage"]["A"] - per_map["mirage"]["B"]
        self.assertLess(nuke_gap, mirage_gap)
        # Sanity: global prior really does favour A.
        self.assertGreater(global_theta["A"], global_theta["B"])

    def test_empty_map_collapses_to_global_prior(self):
        from cs2pickem.ratings import compute_map_bradley_terry

        matches = [
            self._map_match(1, "A", "B", "A", "mirage"),
            self._map_match(2, "A", "B", "A", "mirage"),
        ]
        global_theta = {"A": 1.0, "B": -1.0}
        per_map = compute_map_bradley_terry(
            matches, ridge=8.0, global_theta=global_theta, map_pool=["mirage", "vertigo"]
        )
        # 'vertigo' has zero games -> every team sits at its global prior (centered).
        self.assertIn("vertigo", per_map)
        self.assertGreater(per_map["vertigo"]["A"], per_map["vertigo"]["B"])


def math_isfinite(value):
    import math

    return math.isfinite(value)


if __name__ == "__main__":
    unittest.main()
