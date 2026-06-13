from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from .calibration import ProbabilityCalibrator, make_calibrator
from .cleaning import clean_matches
from .features import FeatureBuilder
from .imbalance import rebalance_training_data
from .maps import average_unknown_map_prediction
from .models import default_ensemble, model_hyperparameters
from .reliability import (
    PLAYER_STATUS_REQUIRED_FEATURES,
    UNSTABLE_IDENTITY_FEATURES,
    apply_final_bt_to_match,
    apply_final_elo_to_match,
    apply_final_glicko_to_match,
    prepare_reliability_features,
)
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
        calibrator: ProbabilityCalibrator | None = None,
        calibration_report: Dict[str, object] | None = None,
        team_elo_ratings: Mapping[str, float] | None = None,
        feature_preparation: Dict[str, object] | None = None,
        team_bt_strengths: Mapping[str, float] | None = None,
        team_bt_map_strengths: Mapping[str, Mapping[str, float]] | None = None,
        team_glicko_state: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.builder = builder
        self.selector = selector
        self.model = model
        self.trained_matches = trained_matches
        self.selected_feature_names = selected_feature_names
        self.imbalance_report = imbalance_report or {}
        self.ensemble_weights = ensemble_weights or dict(getattr(model, "weights", {}))
        self.model_hyperparameters = hyperparameters or {}
        self.calibrator = calibrator
        self.calibration_report = calibration_report or {"basis": "not_applied", "calibration_count": 0}
        self.team_elo_ratings = dict(team_elo_ratings or {})
        # Serve-side rating state for leakage-free scoring of upcoming fixtures. These are
        # populated by train() only when the matching inject_* axis is on; when empty the
        # corresponding apply_final_* injection is a no-op, so the default (Elo-only) serve
        # path is byte-identical to before. Keeping the train/serve injections paired is the
        # anti-skew invariant: a BT/Glicko column injected at fit time but constant-0 at score
        # time would be worse than not injecting at all.
        self.team_bt_strengths = dict(team_bt_strengths or {})
        self.team_bt_map_strengths = {
            map_name: dict(strengths)
            for map_name, strengths in (team_bt_map_strengths or {}).items()
        }
        self.team_glicko_state = {
            key: dict(value)
            for key, value in (team_glicko_state or {}).items()
        }
        self.feature_preparation = feature_preparation or {
            "elo": {"basis": "not_applied", "rows": trained_matches, "teams": 0},
            "excluded_feature_names": list(UNSTABLE_IDENTITY_FEATURES),
            "required_feature_names": list(PLAYER_STATUS_REQUIRED_FEATURES),
        }

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
        calibration_ratio: float = 0.15,
        minimum_calibration_rows: int = 30,
        inject_elo: bool = True,
        calibration_method: str = "platt",
        calibration_cv_folds: int = 0,
        inject_bt: bool = False,
        rating_mode: str = "elo",
        inject_glicko: bool = False,
    ) -> "MatchPredictor":
        # The added rating engine is selected by rating_mode ('elo' default keeps the engine
        # off) OR explicitly by inject_glicko; rating_mode='glicko' implies Glicko injection.
        # inject_bt is an orthogonal switch (BT can ride alongside either rating mode). Both
        # default off so the historic Elo-only train/serve behaviour is bit-for-bit unchanged.
        use_glicko = bool(inject_glicko) or str(rating_mode).strip().lower() == "glicko"
        cleaned_history = sorted(clean_matches([dict(row) for row in history_rows], reference_date=reference_date, max_age_days=max_age_days), key=lambda row: row["date"])
        prepared_history, final_elo, feature_preparation = prepare_reliability_features(
            cleaned_history,
            inject_elo=inject_elo,
            inject_bt=inject_bt,
            inject_glicko=use_glicko,
        )
        # Snapshot the final full-history rating fits so predict_probability_details can score
        # upcoming fixtures with the same engine the model was trained on (serve-side anti-skew).
        bt_report = feature_preparation.get("bt", {}) if isinstance(feature_preparation, Mapping) else {}
        glicko_report = feature_preparation.get("glicko", {}) if isinstance(feature_preparation, Mapping) else {}
        final_bt = dict(bt_report.get("final", {})) if inject_bt else {}
        final_bt_map = (
            {map_name: dict(strengths) for map_name, strengths in (bt_report.get("final_map", {}) or {}).items()}
            if inject_bt
            else {}
        )
        final_glicko = (
            {key: dict(value) for key, value in (glicko_report.get("final", {}) or {}).items()}
            if use_glicko
            else {}
        )
        model_rows, calibration_rows = _model_calibration_split(
            prepared_history,
            calibration_ratio=calibration_ratio,
            minimum_calibration_rows=minimum_calibration_rows,
        )
        builder = FeatureBuilder()
        dataset = builder.fit_transform(model_rows)
        selector = FeatureSelector(
            top_k=top_k,
            excluded_feature_names=feature_preparation["excluded_feature_names"],
            required_feature_names=feature_preparation.get("required_feature_names", []),
        )
        selected = selector.fit_transform(dataset.rows, dataset.labels, dataset.feature_names)
        rebalanced = rebalance_training_data(selected.rows, dataset.labels)
        model = default_ensemble(seed=seed, epochs=epochs, weights=ensemble_weights).fit(rebalanced.rows, rebalanced.labels, sample_weights=rebalanced.sample_weights)
        calibrator, calibration_report = _fit_holdout_calibrator(
            builder,
            selector,
            model,
            calibration_rows,
            method=calibration_method,
            cv_folds=calibration_cv_folds,
        )
        return cls(
            builder,
            selector,
            model,
            len(cleaned_history),
            selected.feature_names,
            rebalanced.report,
            ensemble_weights=dict(model.weights),
            hyperparameters=model_hyperparameters(epochs=epochs),
            calibrator=calibrator,
            calibration_report=calibration_report,
            team_elo_ratings=final_elo,
            feature_preparation=feature_preparation,
            team_bt_strengths=final_bt,
            team_bt_map_strengths=final_bt_map,
            team_glicko_state=final_glicko,
        )

    def predict_probability(self, row: Mapping[str, Any]) -> float:
        return self.predict_probability_details(row)["model_probability_team1"]

    def predict_probability_details(self, row: Mapping[str, Any]) -> Dict[str, object]:
        prepared_row = apply_final_elo_to_match(row, self.team_elo_ratings)
        # Serve-side rating injection mirrors the engines used at train time so the model never
        # sees a column that was non-zero at fit time but constant-0 at score time (train/serve
        # anti-skew). Each step is gated on the matching final state being populated, which
        # train() only does when the corresponding inject_* axis was on -> the default Elo-only
        # path runs neither branch and is byte-identical to before.
        if self.team_bt_strengths or self.team_bt_map_strengths:
            prepared_row = apply_final_bt_to_match(
                prepared_row, self.team_bt_strengths, self.team_bt_map_strengths
            )
        if self.team_glicko_state:
            prepared_row = apply_final_glicko_to_match(prepared_row, self.team_glicko_state)
        transformed = self.builder.transform([prepared_row])
        selected_rows = self.selector.transform(transformed).rows
        raw_probability = self.model.predict_proba(selected_rows)[0]
        probability = self.calibrator.transform_one(raw_probability) if self.calibrator is not None else raw_probability
        components = {}
        if hasattr(self.model, "predict_components"):
            raw_components = self.model.predict_components(selected_rows)
            components = {name: values[0] for name, values in raw_components.items()}
        weights = dict(getattr(self.model, "weights", {}))
        contributions = {name: weights.get(name, 0.0) * probability for name, probability in components.items()}
        return {
            "model_probability_team1": probability,
            "uncalibrated_model_probability_team1": raw_probability,
            "model_probabilities_team1": components,
            "model_weights": weights,
            "weighted_model_contributions_team1": contributions,
            "probability_calibration": self.calibration_report,
            "feature_preparation": self.feature_preparation,
            "team1_elo": prepared_row.get("team1_elo"),
            "team2_elo": prepared_row.get("team2_elo"),
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
            base_details = self.predict_probability_details(row)
            return {
                "model_probabilities_team1": {},
                "model_weights": dict(getattr(self.model, "weights", {})),
                "weighted_model_contributions_team1": {},
                "probability_calibration": self.calibration_report,
                "feature_preparation": self.feature_preparation,
                "team1_elo": base_details.get("team1_elo"),
                "team2_elo": base_details.get("team2_elo"),
            }
        component_totals: Dict[str, float] = {}
        base_details = self.predict_probability_details(row)
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
            "probability_calibration": self.calibration_report,
            "feature_preparation": self.feature_preparation,
            "team1_elo": base_details.get("team1_elo"),
            "team2_elo": base_details.get("team2_elo"),
        }


def _model_calibration_split(
    rows: list[dict],
    calibration_ratio: float,
    minimum_calibration_rows: int,
) -> tuple[list[dict], list[dict]]:
    if len(rows) < max(1, minimum_calibration_rows) * 3:
        return rows, []
    calibration_count = max(minimum_calibration_rows, int(len(rows) * max(0.0, calibration_ratio)))
    calibration_count = min(calibration_count, len(rows) // 3)
    if calibration_count <= 0:
        return rows, []
    return rows[:-calibration_count], rows[-calibration_count:]


def _fit_holdout_calibrator(
    builder: FeatureBuilder,
    selector: FeatureSelector,
    model: object,
    calibration_rows: list[dict],
    *,
    method: str = "platt",
    cv_folds: int = 0,
) -> tuple[ProbabilityCalibrator | None, Dict[str, object]]:
    if not calibration_rows:
        return None, {"basis": "not_applied", "calibration_count": 0}
    transformed = builder.transform(calibration_rows)
    selected_rows = selector.transform(transformed).rows
    labels = [1 if row.get("winner") == row.get("team1") else 0 for row in calibration_rows]
    probabilities = model.predict_proba(selected_rows)
    calibrator = make_calibrator(method, cv_folds=cv_folds).fit(probabilities, labels)
    report = calibrator.report()
    # Preserve the historic basis string for the default platt single-split path
    # (locked by tests); non-default methods get a descriptive '<scope>_<method>'
    # basis so reports/readers can tell which calibrator was applied.
    basis = "holdout_platt_logistic" if method == "platt" and not cv_folds else f"holdout_{method}"
    report.update({"basis": basis, "calibration_count": len(calibration_rows)})
    return calibrator, report


def _map_winrate(profile: Mapping[str, Any], map_name: str) -> float:
    winrates = profile.get("map_winrates") or {}
    if not isinstance(winrates, Mapping):
        return 0.5
    value = winrates.get(map_name, winrates.get(map_name.title(), 0.5))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.5
