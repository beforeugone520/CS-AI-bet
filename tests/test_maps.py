import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class MapPredictionTests(unittest.TestCase):
    def test_candidate_maps_respect_bans_and_preference_scores(self):
        from cs2pickem.maps import candidate_maps_from_bp

        team1 = {
            "ban_top3": ["nuke"],
            "prefer_top3": ["mirage", "inferno", "ancient"],
            "map_winrates": {"mirage": 0.72, "inferno": 0.61, "ancient": 0.57, "anubis": 0.5},
        }
        team2 = {
            "ban_top3": ["vertigo"],
            "prefer_top3": ["mirage", "anubis", "inferno"],
            "map_winrates": {"mirage": 0.55, "inferno": 0.59, "ancient": 0.49, "anubis": 0.66},
        }

        candidates = candidate_maps_from_bp(team1, team2, map_pool=["mirage", "inferno", "ancient", "anubis", "nuke", "vertigo"])

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0], "mirage")
        self.assertNotIn("nuke", candidates)
        self.assertNotIn("vertigo", candidates)

    def test_average_unknown_map_prediction_clones_rows_per_candidate_map(self):
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


if __name__ == "__main__":
    unittest.main()
