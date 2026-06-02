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


if __name__ == "__main__":
    unittest.main()
