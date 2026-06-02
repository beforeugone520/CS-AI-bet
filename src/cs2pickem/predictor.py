from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from .cleaning import clean_matches
from .features import FeatureBuilder
from .imbalance import rebalance_training_data
from .maps import average_unknown_map_prediction
from .models import default_ensemble, model_hyperparameters
from .selection import FeatureSelector


UNKNOWN_MAP_VALUES = {"", "unknown", "tbd", "bo1_unknown", "map_unknown", "none"}


class MatchPredictor:
    def __init__(
        self,
        builder: FeatureBuilder,
        selector: FeatureSelector,
        model: object,
        trained_matches: int,
        selected_feature_names: list[str],
        imbalance_report: Dict[str, object] | None = None,
        ensemble_weights: Dict[str, float] | None = None,
        hyperparameters: Dict[str, object] | None = None,
    ) -> None:
        self.builder = builder
        self.selector = selector
        self.model = model
        self.trained_matches = trained_matches
        self.selected_feature_names = selected_feature_names
        self.imbalance_report = imbalance_report or {}
        self.ensemble_weights = ensemble_weights or dict(getattr(model, "weights", {}))
        self.model_hyperparameters = hyperparameters or {}

    @classmethod
    def train(
        cls,
        history_rows: Iterable[Mapping[str, Any]],
        reference_date: str,
        top_k: int = 25,
        epochs: int = 50,
        seed: int = 53,
        max_age_days: int = 90,
        ensemble_weights: Dict[str, float] | None = None,
    ) -> "MatchPredictor":
        cleaned_history = sorted(clean_matches([dict(row) for row in history_rows], reference_date=reference_date, max_age_days=max_age_days), key=lambda row: row["date"])
        builder = FeatureBuilder()
        dataset = builder.fit_transform(cleaned_history)
        selector = FeatureSelector(top_k=top_k)
        selected = selector.fit_transform(dataset.rows, dataset.labels, dataset.feature_names)
        rebalanced = rebalance_training_data(selected.rows, dataset.labels)
        model = default_ensemble(seed=seed, epochs=epochs, weights=ensemble_weights).fit(rebalanced.rows, rebalanced.labels, sample_weights=rebalanced.sample_weights)
        return cls(
            builder,
            selector,
            model,
            len(cleaned_history),
            selected.feature_names,
            rebalanced.report,
            ensemble_weights=dict(model.weights),
            hyperparameters=model_hyperparameters(epochs=epochs),
        )

    def predict_probability(self, row: Mapping[str, Any]) -> float:
        return self.predict_probability_details(row)["model_probability_team1"]

    def predict_probability_details(self, row: Mapping[str, Any]) -> Dict[str, object]:
        transformed = self.builder.transform([dict(row)])
        selected_rows = self.selector.transform(transformed).rows
        probability = self.model.predict_proba(selected_rows)[0]
        components = {}
        if hasattr(self.model, "predict_components"):
            raw_components = self.model.predict_components(selected_rows)
            components = {name: values[0] for name, values in raw_components.items()}
        weights = dict(getattr(self.model, "weights", {}))
        contributions = {name: weights.get(name, 0.0) * probability for name, probability in components.items()}
        return {
            "model_probability_team1": probability,
            "model_probabilities_team1": components,
            "model_weights": weights,
            "weighted_model_contributions_team1": contributions,
        }

    def predict_with_maps(
        self,
        row: Mapping[str, Any],
        profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> tuple[float, Dict[str, object]]:
        profiles = profiles or {}
        map_name = str(row.get("map", "unknown")).strip().lower()
        team1 = str(row.get("team1"))
        team2 = str(row.get("team2"))
        if map_name in UNKNOWN_MAP_VALUES and team1 in profiles and team2 in profiles:
            result = average_unknown_map_prediction(row, profiles[team1], profiles[team2], self.predict_probability)
            details = self._average_probability_details(row, result["candidate_maps"], profiles[team1], profiles[team2])
            return float(result["average_probability_team1"]), {
                "candidate_maps": result["candidate_maps"],
                "per_map_probability_team1": result["per_map_probability_team1"],
                **details,
            }
        details = self.predict_probability_details(row)
        return float(details["model_probability_team1"]), {"candidate_maps": [], "per_map_probability_team1": {}, **details}

    def _average_probability_details(
        self,
        row: Mapping[str, Any],
        candidate_maps: list[str],
        team1_profile: Mapping[str, Any],
        team2_profile: Mapping[str, Any],
    ) -> Dict[str, object]:
        if not candidate_maps:
            return {
                "model_probabilities_team1": {},
                "model_weights": dict(getattr(self.model, "weights", {})),
                "weighted_model_contributions_team1": {},
            }
        component_totals: Dict[str, float] = {}
        for map_name in candidate_maps:
            map_row = dict(row)
            map_row["map"] = map_name
            map_row["team1_map_winrate"] = _map_winrate(team1_profile, map_name)
            map_row["team2_map_winrate"] = _map_winrate(team2_profile, map_name)
            details = self.predict_probability_details(map_row)
            for name, probability in dict(details["model_probabilities_team1"]).items():
                component_totals[name] = component_totals.get(name, 0.0) + float(probability)
        components = {name: total / len(candidate_maps) for name, total in component_totals.items()}
        weights = dict(getattr(self.model, "weights", {}))
        contributions = {name: weights.get(name, 0.0) * probability for name, probability in components.items()}
        return {
            "model_probabilities_team1": components,
            "model_weights": weights,
            "weighted_model_contributions_team1": contributions,
        }


def _map_winrate(profile: Mapping[str, Any], map_name: str) -> float:
    winrates = profile.get("map_winrates") or {}
    if not isinstance(winrates, Mapping):
        return 0.5
    value = winrates.get(map_name, winrates.get(map_name.title(), 0.5))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.5
