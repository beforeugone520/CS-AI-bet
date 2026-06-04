from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping

from .cleaning import INVALID_STATUSES, SECONDARY_MARKERS, TEMPORARY_MARKERS, VALID_TIERS, normalize_tier, parse_date
from .data import read_matches_csv


DEFAULT_PICKEM_SLOTS = {"3-0": 2, "advance": 6, "0-3": 2}
EXPECTED_ENSEMBLE_MODELS = {"logistic", "random_forest", "xgboost", "neural_network"}


REQUIRED_FIELDS = {
    "date",
    "event",
    "event_tier",
    "status",
    "team1",
    "team2",
    "winner",
    "best_of",
    "map",
    "team1_rank",
    "team2_rank",
    "team1_rmr_points",
    "team2_rmr_points",
    "team1_major_best_placement",
    "team2_major_best_placement",
    "team1_matches_30d",
    "team2_matches_30d",
    "team1_recent_winrate_5",
    "team2_recent_winrate_5",
    "team1_recent_winrate_10",
    "team2_recent_winrate_10",
    "team1_bo1_winrate_6m",
    "team2_bo1_winrate_6m",
    "team1_bo3_winrate_6m",
    "team2_bo3_winrate_6m",
    "team1_map_winrate",
    "team2_map_winrate",
    "team1_rating",
    "team2_rating",
    "team1_kd",
    "team2_kd",
    "team1_opening_success",
    "team2_opening_success",
    "team1_clutch_winrate",
    "team2_clutch_winrate",
    "team1_star_rating",
    "team2_star_rating",
    "team1_substitute_flag",
    "team2_substitute_flag",
    "team1_player_sample",
    "team2_player_sample",
    "h2h_team1_winrate",
    "odds_team1",
    "odds_team2",
    "swiss_round",
    "team1_wins",
    "team1_losses",
    "team2_wins",
    "team2_losses",
    "team1_current_streak",
    "team2_current_streak",
    "version_tag",
}


def audit_readiness(
    rows: Iterable[Mapping[str, Any]],
    training_report: Mapping[str, Any],
    minimum_rows: int = 8000,
    required_teams: int = 80,
    participant_teams: Iterable[str] | None = None,
    top_teams: Iterable[str] | None = None,
    expected_train_end_date: str | None = None,
    expected_validation_end_date: str | None = None,
    minimum_max_age_days: int | None = None,
    sample_reference_date: str | None = None,
    maximum_sample_age_days: int | None = 180,
    bo1_accuracy_target: float = 0.68,
    bo1_auc_target: float = 0.72,
    bo3_accuracy_target: float = 0.75,
    bo3_auc_target: float = 0.78,
    pickem_backtest_report: Mapping[str, Any] | None = None,
    pickem_pass_rate_target: float = 0.38,
    pickem_report: Mapping[str, Any] | None = None,
    minimum_pickem_simulations: int | None = None,
    required_pickem_slots: Mapping[str, int] | None = None,
    minimum_pickem_selection_margin: float | None = None,
    minimum_pickem_market_adjusted_matchups: int | None = None,
    forecast_report: Mapping[str, Any] | None = None,
    require_forecast_low_confidence_avoidance: bool = False,
    source_manifests: Iterable[Mapping[str, Any]] | None = None,
    required_sources: Iterable[str] | None = None,
    source_reference_time: str | None = None,
    maximum_source_age_hours: int | None = None,
    require_validation_tuned_weights: bool = False,
    minimum_player_status_features: int | None = None,
) -> Dict[str, object]:
    materialized = [dict(row) for row in rows]
    teams = {str(row.get("team1")) for row in materialized} | {str(row.get("team2")) for row in materialized}
    checks = {
        "minimum_rows": _check(len(materialized) >= minimum_rows, len(materialized), minimum_rows, "filtered high-quality match rows"),
        "team_coverage": _check(len(teams) >= required_teams, len(teams), required_teams, "distinct professional teams"),
        "data_quality_scope": _data_quality_scope_check(materialized, sample_reference_date, maximum_sample_age_days),
        "required_fields": _required_fields_check(materialized),
        "bo1_performance": _segment_check(training_report, "BO1", bo1_accuracy_target, bo1_auc_target),
        "bo3_performance": _segment_check(training_report, "BO3", bo3_accuracy_target, bo3_auc_target),
        "ensemble_beats_single_models": _ensemble_check(training_report),
    }
    if expected_train_end_date or expected_validation_end_date:
        checks["calendar_split"] = _calendar_split_check(training_report, expected_train_end_date, expected_validation_end_date)
    if minimum_max_age_days is not None:
        checks["freshness_window"] = _freshness_window_check(training_report, minimum_max_age_days)
    if participant_teams is not None:
        checks["participant_coverage"] = _team_list_check(teams, participant_teams, "all declared Major participant teams covered")
    if top_teams is not None:
        checks["top_team_coverage"] = _team_list_check(teams, top_teams, "all declared Top-team training scope covered")
    if pickem_backtest_report is not None:
        checks["pickem_backtest_pass_rate"] = _pickem_backtest_check(pickem_backtest_report, pickem_pass_rate_target)
    if minimum_pickem_simulations is not None:
        checks["pickem_simulations"] = _pickem_simulations_check(pickem_report or {}, minimum_pickem_simulations)
    if required_pickem_slots is not None:
        checks["pickem_slots"] = _pickem_slots_check(pickem_report or {}, required_pickem_slots)
    if minimum_pickem_selection_margin is not None:
        checks["pickem_selection_margin"] = _pickem_selection_margin_check(pickem_report or {}, minimum_pickem_selection_margin)
    if minimum_pickem_market_adjusted_matchups is not None:
        checks["pickem_market_adjustment"] = _pickem_market_adjustment_check(pickem_report or {}, minimum_pickem_market_adjusted_matchups)
    if require_forecast_low_confidence_avoidance:
        checks["forecast_low_confidence_avoidance"] = _forecast_low_confidence_avoidance_check(forecast_report or {})
    if require_validation_tuned_weights:
        checks["validation_tuned_weights"] = _validation_tuned_weights_check(training_report)
    if minimum_player_status_features is not None:
        checks["player_status_features"] = _player_status_features_check(training_report, minimum_player_status_features)
    if source_manifests is not None:
        checks["source_freshness"] = _source_freshness_check(
            source_manifests,
            required_sources=required_sources,
            reference_time=source_reference_time,
            maximum_age_hours=maximum_source_age_hours if maximum_source_age_hours is not None else 24,
        )
    failed = [name for name, result in checks.items() if not result["passed"]]
    return {
        "ready": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def audit_readiness_file(
    matches_path: str,
    training_report_path: str,
    minimum_rows: int = 8000,
    required_teams: int = 80,
    participants_path: str | None = None,
    top_teams_path: str | None = None,
    expected_train_end_date: str | None = None,
    expected_validation_end_date: str | None = None,
    minimum_max_age_days: int | None = None,
    sample_reference_date: str | None = None,
    maximum_sample_age_days: int | None = 180,
    pickem_backtest_report_path: str | None = None,
    pickem_pass_rate_target: float = 0.38,
    pickem_report_path: str | None = None,
    minimum_pickem_simulations: int | None = None,
    required_pickem_slots: Mapping[str, int] | None = None,
    minimum_pickem_selection_margin: float | None = None,
    minimum_pickem_market_adjusted_matchups: int | None = None,
    forecast_report_path: str | None = None,
    require_forecast_low_confidence_avoidance: bool = False,
    source_manifest_paths: Iterable[str] | None = None,
    required_sources: Iterable[str] | None = None,
    source_reference_time: str | None = None,
    maximum_source_age_hours: int | None = None,
    require_validation_tuned_weights: bool = False,
    minimum_player_status_features: int | None = None,
) -> Dict[str, object]:
    with open(training_report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    pickem_backtest_report = None
    if pickem_backtest_report_path:
        with open(pickem_backtest_report_path, encoding="utf-8") as handle:
            pickem_backtest_report = json.load(handle)
    pickem_report = None
    if pickem_report_path:
        with open(pickem_report_path, encoding="utf-8") as handle:
            pickem_report = json.load(handle)
    forecast_report = None
    if forecast_report_path:
        with open(forecast_report_path, encoding="utf-8") as handle:
            forecast_report = json.load(handle)
    source_manifests = _read_source_manifests(source_manifest_paths or [])
    return audit_readiness(
        read_matches_csv(matches_path),
        report,
        minimum_rows=minimum_rows,
        required_teams=required_teams,
        participant_teams=_read_team_names(participants_path) if participants_path else None,
        top_teams=_read_team_names(top_teams_path) if top_teams_path else None,
        expected_train_end_date=expected_train_end_date,
        expected_validation_end_date=expected_validation_end_date,
        minimum_max_age_days=minimum_max_age_days,
        sample_reference_date=sample_reference_date,
        maximum_sample_age_days=maximum_sample_age_days,
        pickem_backtest_report=pickem_backtest_report,
        pickem_pass_rate_target=pickem_pass_rate_target,
        pickem_report=pickem_report,
        minimum_pickem_simulations=minimum_pickem_simulations,
        required_pickem_slots=required_pickem_slots,
        minimum_pickem_selection_margin=minimum_pickem_selection_margin,
        minimum_pickem_market_adjusted_matchups=minimum_pickem_market_adjusted_matchups,
        forecast_report=forecast_report,
        require_forecast_low_confidence_avoidance=require_forecast_low_confidence_avoidance,
        source_manifests=source_manifests if source_manifest_paths else None,
        required_sources=required_sources,
        source_reference_time=source_reference_time,
        maximum_source_age_hours=maximum_source_age_hours,
        require_validation_tuned_weights=require_validation_tuned_weights,
        minimum_player_status_features=minimum_player_status_features,
    )


def _required_fields_check(rows: List[Dict[str, Any]]) -> Dict[str, object]:
    missing = sorted(field for field in REQUIRED_FIELDS if any(row.get(field) in (None, "") for row in rows))
    return {
        "passed": not missing,
        "actual": [] if not missing else missing,
        "target": "all required fields populated",
        "detail": "required modeling fields",
    }


def _data_quality_scope_check(rows: List[Dict[str, Any]], reference_date: str | None, maximum_age_days: int | None) -> Dict[str, object]:
    parsed_dates = []
    invalid_date_rows = 0
    for row in rows:
        try:
            parsed_dates.append(parse_date(row.get("date")))
        except ValueError:
            invalid_date_rows += 1
    ref = parse_date(reference_date) if reference_date else max(parsed_dates, default=None)
    stale_rows = 0
    if ref and maximum_age_days is not None:
        stale_rows = sum(1 for played_at in parsed_dates if (ref - played_at).days > maximum_age_days)
    actual = {
        "invalid_tier_rows": sum(1 for row in rows if normalize_tier(row.get("event_tier", "")) not in VALID_TIERS),
        "stale_rows": stale_rows,
        "invalid_status_rows": sum(1 for row in rows if str(row.get("status", "")).strip().lower() in INVALID_STATUSES),
        "secondary_team_rows": sum(1 for row in rows if _has_secondary_team(row)),
        "temporary_team_rows": sum(1 for row in rows if _has_temporary_team(row)),
        "invalid_date_rows": invalid_date_rows,
    }
    return {
        "passed": all(value == 0 for value in actual.values()),
        "actual": actual,
        "target": {
            "event_tiers": sorted(VALID_TIERS),
            "maximum_sample_age_days": maximum_age_days,
            "invalid_rows": 0,
        },
        "detail": "S/A official six-month high-quality sample scope",
    }


def _segment_check(report: Mapping[str, Any], segment: str, accuracy_target: float, auc_target: float) -> Dict[str, object]:
    metrics = dict(report.get("segment_metrics", {}).get(segment, {}))
    accuracy = float(metrics.get("accuracy", 0.0))
    auc = float(metrics.get("auc", 0.0))
    return {
        "passed": accuracy >= accuracy_target and auc >= auc_target,
        "actual": {"accuracy": accuracy, "auc": auc},
        "target": {"accuracy": accuracy_target, "auc": auc_target},
        "detail": f"{segment} holdout performance",
    }


def _ensemble_check(report: Mapping[str, Any]) -> Dict[str, object]:
    comparison = dict(report.get("model_comparison", {}))
    ensemble_auc = float(comparison.get("ensemble", {}).get("auc", 0.0))
    single_aucs = [
        float(metrics.get("auc", 0.0))
        for name, metrics in comparison.items()
        if name != "ensemble" and isinstance(metrics, Mapping)
    ]
    best_single = max(single_aucs) if single_aucs else 0.0
    return {
        "passed": bool(single_aucs) and ensemble_auc >= best_single,
        "actual": {"ensemble_auc": ensemble_auc, "best_single_auc": best_single},
        "target": "ensemble auc >= best single-model auc",
        "detail": "fusion model improvement",
    }


def _pickem_backtest_check(report: Mapping[str, Any], pass_rate_target: float) -> Dict[str, object]:
    pass_rate = _float_or_zero(report.get("pass_rate"))
    cases = _int_or_zero(report.get("cases"))
    passed_cases = _int_or_zero(report.get("passed_cases"))
    target = max(pass_rate_target, _float_or_zero(report.get("pass_rate_target")))
    return {
        "passed": cases > 0 and pass_rate >= target and bool(report.get("meets_pass_rate_target", pass_rate >= target)),
        "actual": {"cases": cases, "passed_cases": passed_cases, "pass_rate": pass_rate},
        "target": {"pass_rate": target},
        "detail": "historical Major Pick'em backtest pass rate",
    }


def _pickem_simulations_check(report: Mapping[str, Any], minimum_simulations: int) -> Dict[str, object]:
    actual = _int_or_zero(report.get("simulations"))
    return {
        "passed": actual >= minimum_simulations,
        "actual": actual,
        "target": f">= {minimum_simulations}",
        "detail": "Swiss Monte Carlo simulation count",
    }


def _pickem_slots_check(report: Mapping[str, Any], required_slots: Mapping[str, int]) -> Dict[str, object]:
    pickems = report.get("pickems", {})
    if not isinstance(pickems, Mapping):
        pickems = {}
    actual = {category: len(_team_list(pickems.get(category, []))) for category in required_slots}
    selected_teams: List[str] = []
    for category in required_slots:
        selected_teams.extend(_team_list(pickems.get(category, [])))
    duplicate_teams = sorted({team for team in selected_teams if selected_teams.count(team) > 1})
    simulated_teams = report.get("team_probabilities", {})
    known_teams = set(simulated_teams) if isinstance(simulated_teams, Mapping) else set()
    unknown_teams = sorted({team for team in selected_teams if known_teams and team not in known_teams})
    return {
        "passed": actual == dict(required_slots) and not duplicate_teams and not unknown_teams,
        "actual": {**actual, "duplicate_teams": duplicate_teams, "unknown_teams": unknown_teams},
        "target": dict(required_slots),
        "detail": "complete non-overlapping Pick'em answer slots",
    }


def _pickem_selection_margin_check(report: Mapping[str, Any], minimum_margin: float) -> Dict[str, object]:
    details = report.get("pickem_details", {})
    if not isinstance(details, Mapping):
        details = {}
    margins: List[float] = []
    low_margin_picks: List[Dict[str, object]] = []
    missing_margin_picks: List[Dict[str, object]] = []
    for category, entries in details.items():
        if isinstance(entries, Mapping):
            entries = [entries]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            margin = _float_or_none(entry.get("selection_margin"))
            pick = {
                "category": str(entry.get("category", category)),
                "team": str(entry.get("team", "")),
            }
            if margin is None:
                missing_margin_picks.append(pick)
                continue
            margins.append(margin)
            if margin < minimum_margin:
                low_margin_picks.append({**pick, "selection_margin": margin})
    return {
        "passed": bool(margins) and not low_margin_picks and not missing_margin_picks,
        "actual": {
            "minimum_margin": min(margins) if margins else None,
            "low_margin_picks": low_margin_picks,
            "missing_margin_picks": missing_margin_picks,
        },
        "target": f">= {minimum_margin}",
        "detail": "minimum Pick'em selection margin versus next candidate",
    }


def _pickem_market_adjustment_check(report: Mapping[str, Any], minimum_adjusted_matchups: int) -> Dict[str, object]:
    summary = report.get("market_adjustment_summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    adjusted_matchups = _int_or_zero(summary.get("adjusted_matchups"))
    cached_matchups = _int_or_zero(summary.get("cached_matchups"))
    return {
        "passed": adjusted_matchups >= minimum_adjusted_matchups,
        "actual": {
            "cached_matchups": cached_matchups,
            "adjusted_matchups": adjusted_matchups,
        },
        "target": f">= {minimum_adjusted_matchups}",
        "detail": "Pick'em Swiss matchup probabilities using real market odds",
    }


def _forecast_low_confidence_avoidance_check(report: Mapping[str, Any]) -> Dict[str, object]:
    predictions = report.get("predictions", [])
    if not isinstance(predictions, list):
        predictions = []
    low_confidence = []
    low_confidence_non_avoids = []
    for row in predictions:
        if not isinstance(row, Mapping):
            continue
        if not _truthy(row.get("low_confidence")):
            continue
        item = {
            "team1": str(row.get("team1", "")),
            "team2": str(row.get("team2", "")),
            "pick": str(row.get("pick", "")),
            "confidence_margin": _float_or_none(row.get("confidence_margin")),
        }
        low_confidence.append(item)
        if item["pick"] != "avoid":
            low_confidence_non_avoids.append(item)
    return {
        "passed": not low_confidence_non_avoids,
        "actual": {
            "low_confidence_predictions": len(low_confidence),
            "low_confidence_non_avoids": low_confidence_non_avoids,
        },
        "target": "all low-confidence forecasts pick avoid",
        "detail": "single-match low-confidence avoidance",
    }


def _validation_tuned_weights_check(report: Mapping[str, Any]) -> Dict[str, object]:
    tuning = report.get("validation_tuned_ensemble_weights", {})
    if not isinstance(tuning, Mapping):
        tuning = {}
    weights = tuning.get("weights", {})
    if not isinstance(weights, Mapping):
        weights = {}
    numeric_weights = {str(name): _float_or_zero(value) for name, value in weights.items()}
    weight_sum = sum(numeric_weights.values())
    missing_models = sorted(EXPECTED_ENSEMBLE_MODELS - set(numeric_weights))
    actual = {
        "basis": tuning.get("basis"),
        "validation_count": _int_or_zero(tuning.get("validation_count")),
        "models": sorted(numeric_weights),
        "missing_models": missing_models,
        "weight_sum": weight_sum,
    }
    return {
        "passed": (
            actual["basis"] == "validation_log_loss"
            and actual["validation_count"] > 0
            and not missing_models
            and all(value > 0 for value in numeric_weights.values())
            and abs(weight_sum - 1.0) <= 1e-6
        ),
        "actual": actual,
        "target": {
            "basis": "validation_log_loss",
            "validation_count": "> 0",
            "models": sorted(EXPECTED_ENSEMBLE_MODELS),
            "weight_sum": 1.0,
        },
        "detail": "pre-event validation log-loss tuned ensemble weights",
    }


def _player_status_features_check(report: Mapping[str, Any], minimum_selected: int) -> Dict[str, object]:
    feature_selection = report.get("feature_selection", {})
    if not isinstance(feature_selection, Mapping):
        feature_selection = {}
    required = feature_selection.get("required_features", {})
    if not isinstance(required, Mapping):
        required = {}
    requested = _string_list(required.get("requested"))
    available = _string_list(required.get("available"))
    selected = _string_list(required.get("selected"))
    unavailable = _string_list(required.get("unavailable"))
    actual = {
        "requested": requested,
        "available": available,
        "selected": selected,
        "unavailable": unavailable,
        "selected_count": len(selected),
    }
    return {
        "passed": len(selected) >= minimum_selected,
        "actual": actual,
        "target": {
            "minimum_selected": minimum_selected,
            "source": "training_report.feature_selection.required_features.selected",
        },
        "detail": "selected player-status modeling features",
    }


def _source_freshness_check(
    manifests: Iterable[Mapping[str, Any]],
    required_sources: Iterable[str] | None,
    reference_time: str | None,
    maximum_age_hours: int,
) -> Dict[str, object]:
    materialized = [dict(manifest) for manifest in manifests]
    reference = _parse_timestamp(reference_time) if reference_time else datetime.now(timezone.utc)
    fresh_sources: List[str] = []
    stale_sources: List[Dict[str, object]] = []
    invalid_sources: List[str] = []
    seen_sources: set[str] = set()

    for index, manifest in enumerate(materialized, start=1):
        source = _manifest_source(manifest, index)
        seen_sources.add(source)
        timestamp = _manifest_timestamp(manifest)
        if timestamp is None:
            invalid_sources.append(source)
            continue
        age_hours = round((reference - timestamp).total_seconds() / 3600.0, 3)
        if age_hours <= maximum_age_hours:
            fresh_sources.append(source)
        else:
            stale_sources.append({"source": source, "age_hours": age_hours})

    required = sorted({str(source).strip() for source in (required_sources or []) if str(source).strip()})
    missing_sources = [source for source in required if source not in seen_sources]
    actual = {
        "fresh_sources": sorted(fresh_sources),
        "stale_sources": sorted(stale_sources, key=lambda item: str(item["source"])),
        "missing_sources": missing_sources,
        "invalid_sources": sorted(invalid_sources),
    }
    return {
        "passed": not stale_sources and not missing_sources and not invalid_sources and bool(materialized),
        "actual": actual,
        "target": {
            "maximum_source_age_hours": maximum_age_hours,
            "required_sources": required,
        },
        "detail": "pre-event source manifests refreshed within the configured window",
    }


def _calendar_split_check(report: Mapping[str, Any], train_end_date: str | None, validation_end_date: str | None) -> Dict[str, object]:
    boundaries = dict(report.get("split_boundaries") or {})
    actual = {
        "split_strategy": report.get("split_strategy"),
        "train_end_date": boundaries.get("train_end_date"),
        "validation_end_date": boundaries.get("validation_end_date"),
    }
    target = {
        "split_strategy": "date_boundaries",
        "train_end_date": train_end_date,
        "validation_end_date": validation_end_date,
    }
    return {
        "passed": actual == target,
        "actual": actual,
        "target": target,
        "detail": "explicit production calendar split",
    }


def _freshness_window_check(report: Mapping[str, Any], minimum_max_age_days: int) -> Dict[str, object]:
    actual = _int_or_zero(report.get("max_age_days"))
    return {
        "passed": actual >= minimum_max_age_days,
        "actual": actual,
        "target": f">= {minimum_max_age_days}",
        "detail": "training freshness window",
    }


def _read_source_manifests(paths: Iterable[str]) -> List[Dict[str, Any]]:
    manifests = []
    for path in paths:
        manifests.extend(_read_source_manifest_path(path))
    return manifests


def _read_source_manifest_path(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    job_reports = payload.get("job_reports")
    if isinstance(job_reports, list):
        expanded: List[Dict[str, Any]] = []
        for job in job_reports:
            if not isinstance(job, Mapping):
                continue
            manifest_path = job.get("manifest_path")
            if manifest_path in (None, ""):
                expanded.append(dict(job))
                continue
            nested_path = str(manifest_path)
            if not nested_path.startswith("/"):
                nested_path = os.path.abspath(os.path.join(os.path.dirname(path), nested_path))
            expanded.extend(_read_source_manifest_path(nested_path))
        return expanded
    return [dict(payload)]


def _manifest_source(manifest: Mapping[str, Any], index: int) -> str:
    for key in ("source", "source_name", "name"):
        value = manifest.get(key)
        if value not in (None, ""):
            return str(value)
    return f"manifest-{index}"


def _manifest_timestamp(manifest: Mapping[str, Any]) -> datetime | None:
    for key in ("updated_at", "fetched_at", "generated_at", "created_at"):
        value = manifest.get(key)
        if value in (None, ""):
            continue
        try:
            return _parse_timestamp(str(value))
        except ValueError:
            return None
    return None


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _team_list_check(covered_teams: set[str], required_list: Iterable[str], detail: str) -> Dict[str, object]:
    required = sorted({_normalize_team(team) for team in required_list if _normalize_team(team)})
    covered_lookup = {_normalize_team(team): team for team in covered_teams}
    missing = [team for team in required if team not in covered_lookup]
    return {
        "passed": not missing,
        "actual": {"covered": len(required) - len(missing), "required": len(required), "missing": missing},
        "target": "100% coverage",
        "detail": detail,
    }


def _team_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return []


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Iterable):
        return []
    return [str(item) for item in value if str(item)]


def _read_team_names(path: str) -> List[str]:
    names = []
    for row in read_matches_csv(path):
        value = row.get("team") or row.get("name") or row.get("team_name")
        if value not in (None, ""):
            names.append(str(value))
    return names


def _normalize_team(value: Any) -> str:
    return str(value).strip()


def _has_secondary_team(row: Mapping[str, Any]) -> bool:
    return _team_has_marker(row, SECONDARY_MARKERS, "is_secondary")


def _has_temporary_team(row: Mapping[str, Any]) -> bool:
    return _team_has_marker(row, TEMPORARY_MARKERS, "is_temporary")


def _team_has_marker(row: Mapping[str, Any], markers: Iterable[str], flag_suffix: str) -> bool:
    for prefix in ("team1", "team2"):
        if _truthy(row.get(f"{prefix}_{flag_suffix}")):
            return True
        team_name = str(row.get(prefix, "")).strip().lower()
        if any(marker in team_name for marker in markers):
            return True
    return False


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check(passed: bool, actual: Any, target: Any, detail: str) -> Dict[str, object]:
    return {"passed": passed, "actual": actual, "target": target, "detail": detail}
