from __future__ import annotations

import os
from typing import Dict, Optional

from .bp import merge_bp_file
from .data import read_matches_csv, write_json, write_matches_csv
from .export import build_pickem_answer_sheet
from .forecast import forecast_fixtures_file
from .odds import merge_odds_file
from .pickem import model_driven_pickems_file
from .pipeline import enrich_matches_file, train_evaluate
from .players import merge_player_stats_file
from .readiness import DEFAULT_PICKEM_SLOTS, audit_readiness_file
from .sources import annotate_version_tags, parse_version_log
from .strategy import DEFAULT_PICKEM_OBJECTIVE
from .update import _augment_parsed_rows
from .visualization import write_training_visualizations


def run_end_to_end_pipeline(
    history_path: str,
    fixtures_path: str,
    teams_path: str,
    reference_date: str,
    output_dir: str,
    odds_path: Optional[str] = None,
    players_path: Optional[str] = None,
    bp_path: Optional[str] = None,
    simulations: int = 100000,
    top_k: int = 25,
    epochs: int = 50,
    stage: str = "default",
    cv_folds: int = 5,
    window_days: int = 15,
    max_age_days: int = 90,
    train_end_date: str | None = None,
    validation_end_date: str | None = None,
    minimum_rows: int = 8000,
    required_teams: int = 80,
    participants_path: Optional[str] = None,
    top_teams_path: Optional[str] = None,
    version_log_path: Optional[str] = None,
    pickem_backtest_report_path: Optional[str] = None,
    pickem_pass_rate_target: float = 0.38,
    minimum_pickem_simulations: int | None = 100000,
    required_pickem_slots: Dict[str, int] | None = DEFAULT_PICKEM_SLOTS,
    minimum_pickem_selection_margin: float | None = 0.04,
    minimum_pickem_market_adjusted_matchups: int | None = None,
    require_forecast_low_confidence_avoidance: bool = True,
    source_manifest_paths: list[str] | None = None,
    required_sources: list[str] | None = None,
    source_reference_time: str | None = None,
    maximum_source_age_hours: int | None = None,
    require_validation_tuned_weights: bool = True,
    pickem_objective: str = DEFAULT_PICKEM_OBJECTIVE,
    pickem_threshold: int | None = None,
    pickem_pairing: str = "legacy",
    series_uplift: bool = False,
    leverage_strength: float = 1.0,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    artifacts = _artifact_paths(output_dir)

    enrich_report = enrich_matches_file(history_path, artifacts["enriched_matches"], artifacts["profiles"])
    history_rows = read_matches_csv(artifacts["enriched_matches"])
    history_augmentation: Dict[str, object] = {}
    if version_log_path:
        with open(version_log_path, encoding="utf-8") as handle:
            history_rows = annotate_version_tags(history_rows, parse_version_log(handle.read()))
        history_augmentation["version_log"] = {"applied": True, "path": os.path.abspath(version_log_path)}
    history_rows, field_augmentation = _augment_parsed_rows(
        history_rows,
        team_metadata_path=teams_path,
        player_stats_path=players_path,
        player_window_days=window_days,
        default_swiss_state=True,
    )
    history_augmentation.update(field_augmentation)
    if history_augmentation:
        write_matches_csv(artifacts["enriched_matches"], history_rows)
        enrich_report["training_augmentation"] = history_augmentation

    fixture_input = fixtures_path
    odds_report = None
    if odds_path:
        odds_report = merge_odds_file(fixture_input, odds_path, artifacts["fixtures_with_odds"])
        fixture_input = artifacts["fixtures_with_odds"]

    players_report = None
    if players_path:
        players_report = merge_player_stats_file(fixture_input, players_path, artifacts["fixtures_ready"], window_days=window_days)
        fixture_input = artifacts["fixtures_ready"]

    fixture_rows, fixtures_augmentation = _augment_parsed_rows(
        read_matches_csv(fixture_input),
        team_metadata_path=teams_path,
        player_stats_path=None,
        player_window_days=window_days,
        default_swiss_state=True,
    )
    write_matches_csv(artifacts["fixtures_ready"], fixture_rows)
    fixture_input = artifacts["fixtures_ready"]

    bp_report = None
    if bp_path:
        bp_report = merge_bp_file(fixture_input, bp_path, artifacts["fixtures_with_bp"])
        fixture_input = artifacts["fixtures_with_bp"]

    train_report = train_evaluate(
        read_matches_csv(artifacts["enriched_matches"]),
        reference_date=reference_date,
        epochs=epochs,
        top_k=top_k,
        cv_folds=cv_folds,
        max_age_days=max_age_days,
        train_end_date=train_end_date,
        validation_end_date=validation_end_date,
    )
    write_json(artifacts["train_report"], train_report)
    visualization_report = write_training_visualizations(train_report, os.path.join(output_dir, "visualizations"), prefix="training")

    tuned_ensemble_weights = _recommended_ensemble_weights(train_report)

    forecast_report = forecast_fixtures_file(
        history_path=artifacts["enriched_matches"],
        fixtures_path=fixture_input,
        reference_date=reference_date,
        profiles_path=artifacts["profiles"],
        bp_path=None,
        top_k=top_k,
        epochs=epochs,
        max_age_days=max_age_days,
        ensemble_weights=tuned_ensemble_weights,
    )
    write_json(artifacts["forecast_report"], forecast_report)

    pickem_report = model_driven_pickems_file(
        history_path=artifacts["enriched_matches"],
        teams_path=teams_path,
        reference_date=reference_date,
        profiles_path=artifacts["profiles"],
        simulations=simulations,
        top_k=top_k,
        epochs=epochs,
        stage=stage,
        max_age_days=max_age_days,
        ensemble_weights=tuned_ensemble_weights,
        fixtures_path=fixture_input,
        pickem_objective=pickem_objective,
        pickem_threshold=pickem_threshold,
        pickem_pairing=pickem_pairing,
        series_uplift=series_uplift,
        leverage_strength=leverage_strength,
    )
    write_json(artifacts["pickem_report"], pickem_report)

    readiness_report = audit_readiness_file(
        artifacts["enriched_matches"],
        artifacts["train_report"],
        minimum_rows=minimum_rows,
        required_teams=required_teams,
        participants_path=participants_path,
        top_teams_path=top_teams_path,
        expected_train_end_date=train_end_date,
        expected_validation_end_date=validation_end_date,
        minimum_max_age_days=max_age_days,
        sample_reference_date=reference_date,
        maximum_sample_age_days=max_age_days,
        pickem_backtest_report_path=pickem_backtest_report_path,
        pickem_pass_rate_target=pickem_pass_rate_target,
        pickem_report_path=artifacts["pickem_report"],
        forecast_report_path=artifacts["forecast_report"],
        minimum_pickem_simulations=minimum_pickem_simulations,
        required_pickem_slots=required_pickem_slots,
        minimum_pickem_selection_margin=minimum_pickem_selection_margin,
        minimum_pickem_market_adjusted_matchups=_market_adjustment_gate(odds_path, minimum_pickem_market_adjusted_matchups),
        require_forecast_low_confidence_avoidance=require_forecast_low_confidence_avoidance,
        source_manifest_paths=source_manifest_paths,
        required_sources=required_sources,
        source_reference_time=source_reference_time,
        maximum_source_age_hours=maximum_source_age_hours,
        require_validation_tuned_weights=require_validation_tuned_weights,
    )
    write_json(artifacts["readiness_report"], readiness_report)

    answer_sheet = build_pickem_answer_sheet(
        pickem_report,
        readiness_report=readiness_report,
        minimum_selection_margin=minimum_pickem_selection_margin or 0.0,
        required_slots=required_pickem_slots,
    )
    write_json(artifacts["pickem_answer_sheet"], answer_sheet)

    report = {
        "artifacts": artifacts,
        "enrich": enrich_report,
        "history_augmentation": history_augmentation,
        "fixtures_augmentation": fixtures_augmentation,
        "odds": odds_report,
        "players": players_report,
        "bp": bp_report,
        "train": train_report,
        "visualization": visualization_report,
        "readiness": readiness_report,
        "forecast": forecast_report,
        "pickem": pickem_report,
        "answer_sheet": answer_sheet,
    }
    write_json(artifacts["pipeline_manifest"], report)
    return report


def _recommended_ensemble_weights(train_report: Dict[str, object]) -> Dict[str, float]:
    tuning = train_report.get("validation_tuned_ensemble_weights")
    if isinstance(tuning, dict) and tuning.get("basis") == "validation_log_loss" and isinstance(tuning.get("weights"), dict):
        return {str(name): float(value) for name, value in tuning["weights"].items()}
    weights = train_report.get("ensemble_weights")
    if isinstance(weights, dict):
        return {str(name): float(value) for name, value in weights.items()}
    return {}


def _market_adjustment_gate(odds_path: Optional[str], configured_minimum: int | None) -> int | None:
    if configured_minimum is not None:
        return configured_minimum
    return 1 if odds_path else None


def _artifact_paths(output_dir: str) -> Dict[str, str]:
    return {
        "enriched_matches": os.path.abspath(os.path.join(output_dir, "enriched_matches.csv")),
        "profiles": os.path.abspath(os.path.join(output_dir, "team_profiles.json")),
        "fixtures_with_odds": os.path.abspath(os.path.join(output_dir, "fixtures_with_odds.csv")),
        "fixtures_ready": os.path.abspath(os.path.join(output_dir, "fixtures_ready.csv")),
        "fixtures_with_bp": os.path.abspath(os.path.join(output_dir, "fixtures_with_bp.csv")),
        "train_report": os.path.abspath(os.path.join(output_dir, "train_report.json")),
        "readiness_report": os.path.abspath(os.path.join(output_dir, "readiness_report.json")),
        "forecast_report": os.path.abspath(os.path.join(output_dir, "forecast_report.json")),
        "pickem_report": os.path.abspath(os.path.join(output_dir, "pickem_report.json")),
        "pickem_answer_sheet": os.path.abspath(os.path.join(output_dir, "pickem_answer_sheet.json")),
        "pipeline_manifest": os.path.abspath(os.path.join(output_dir, "pipeline_manifest.json")),
    }
