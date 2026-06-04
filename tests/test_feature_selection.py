import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class FeatureSelectionTests(unittest.TestCase):
    def test_feature_builder_encodes_version_tag(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {"team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "version_tag": "patch-a", "best_of": 1},
            {"team1": "Charlie", "team2": "Delta", "winner": "Delta", "version_tag": "patch-b", "best_of": 3},
        ]
        dataset = FeatureBuilder().fit_transform(rows)

        self.assertIn("version_tag_code", dataset.feature_names)
        version_index = dataset.feature_names.index("version_tag_code")
        self.assertNotEqual(dataset.rows[0][version_index], dataset.rows[1][version_index])

    def test_feature_builder_one_hot_encodes_maps_and_label_encodes_categories(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {"date": "2026-05-01", "event": "IEM A", "event_tier": "S", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "map": "mirage"},
            {"date": "2026-05-02", "event": "IEM B", "event_tier": "A", "team1": "Charlie", "team2": "Delta", "winner": "Delta", "map": "inferno"},
        ]
        dataset = FeatureBuilder().fit_transform(rows)

        self.assertIn("map_mirage", dataset.feature_names)
        self.assertIn("map_inferno", dataset.feature_names)
        self.assertIn("team1_code", dataset.feature_names)
        self.assertIn("team2_code", dataset.feature_names)
        self.assertIn("event_code", dataset.feature_names)
        self.assertIn("event_tier_code", dataset.feature_names)
        self.assertEqual(dataset.rows[0][dataset.feature_names.index("map_mirage")], 1.0)
        self.assertEqual(dataset.rows[0][dataset.feature_names.index("map_inferno")], 0.0)

    def test_feature_builder_adds_major_history_and_swiss_distance_features(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {
                "team1": "Alpha",
                "team2": "Bravo",
                "winner": "Alpha",
                "best_of": 1,
                "team1_major_best_placement": 1,
                "team2_major_best_placement": 8,
                "team1_wins": 2,
                "team1_losses": 0,
                "team2_wins": 0,
                "team2_losses": 2,
            },
            {
                "team1": "Charlie",
                "team2": "Delta",
                "winner": "Delta",
                "best_of": 3,
                "team1_major_best_placement": 12,
                "team2_major_best_placement": 4,
                "team1_wins": 0,
                "team1_losses": 2,
                "team2_wins": 2,
                "team2_losses": 0,
            },
        ]
        dataset = FeatureBuilder().fit_transform(rows)

        for name in [
            "major_best_placement_diff",
            "wins_needed_to_advance_diff",
            "losses_until_elimination_diff",
        ]:
            self.assertIn(name, dataset.feature_names)
            index = dataset.feature_names.index(name)
            self.assertGreater(dataset.rows[0][index], dataset.rows[1][index])

    def test_feature_selector_removes_low_variance_and_correlated_features(self):
        from cs2pickem.selection import FeatureSelector

        rows = [
            [0.0, 0.0, 1.0, 0.2],
            [0.2, 0.2, 1.0, 0.8],
            [0.4, 0.4, 1.0, 0.3],
            [0.6, 0.6, 1.0, 0.7],
        ]
        labels = [0, 0, 1, 1]
        names = ["rank_diff", "duplicate_rank_diff", "constant", "map_winrate_diff"]

        selector = FeatureSelector(variance_threshold=0.01, correlation_threshold=0.8, top_k=2)
        reduced = selector.fit_transform(rows, labels, names)

        self.assertEqual(len(reduced.feature_names), 2)
        self.assertIn("rank_diff", reduced.feature_names)
        self.assertNotIn("duplicate_rank_diff", reduced.feature_names)
        self.assertNotIn("constant", reduced.feature_names)
        self.assertEqual(len(reduced.rows[0]), 2)

    def test_feature_selector_reports_importance_scores(self):
        from cs2pickem.selection import FeatureSelector

        rows = [[0.1, 0.8], [0.2, 0.1], [0.8, 0.3], [0.9, 0.6]]
        labels = [0, 0, 1, 1]
        selector = FeatureSelector(top_k=1)
        selector.fit(rows, labels, ["strong_signal", "inverse_signal"])

        self.assertEqual(len(selector.importance_scores), 2)
        self.assertEqual(selector.selected_feature_names, ["strong_signal"])

    def test_feature_selector_preserves_required_features_within_top_k(self):
        from cs2pickem.selection import FeatureSelector

        rows = [
            [0.05, 0.05, 0.1],
            [0.15, 0.10, 0.9],
            [0.85, 0.90, 0.2],
            [0.95, 0.95, 0.8],
        ]
        labels = [0, 0, 1, 1]
        selector = FeatureSelector(
            top_k=2,
            required_feature_names=["player_sample_confidence_diff"],
        )
        selector.fit(rows, labels, ["strong_signal", "second_signal", "player_sample_confidence_diff"])

        self.assertEqual(len(selector.selected_feature_names), 2)
        self.assertIn("strong_signal", selector.selected_feature_names)
        self.assertIn("player_sample_confidence_diff", selector.selected_feature_names)

    def test_feature_selector_reports_required_feature_availability(self):
        from cs2pickem.selection import FeatureSelector

        rows = [
            [0.1, 0.0, 0.2],
            [0.2, 0.0, 0.8],
            [0.8, 0.0, 0.3],
            [0.9, 0.0, 0.7],
        ]
        labels = [0, 0, 1, 1]
        selector = FeatureSelector(
            top_k=2,
            required_feature_names=["constant_status", "variable_status", "missing_status"],
        )
        selector.fit(rows, labels, ["strong_signal", "constant_status", "variable_status"])

        self.assertEqual(selector.required_feature_report["requested"], ["constant_status", "variable_status", "missing_status"])
        self.assertEqual(selector.required_feature_report["available"], ["variable_status"])
        self.assertEqual(selector.required_feature_report["selected"], ["variable_status"])
        self.assertEqual(selector.required_feature_report["unavailable"], ["constant_status", "missing_status"])


if __name__ == "__main__":
    unittest.main()
