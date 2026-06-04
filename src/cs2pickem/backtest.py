from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping

from .data import read_matches_csv, read_teams_csv, write_json
from .pickem import model_driven_pickems


PICKEM_CATEGORIES = ("3-0", "advance", "0-3")
PLAYER_FORM_COUNTER_CONFIDENCE_CANDIDATES = (0.0, 0.2, 0.4, 0.6, 0.8)


def evaluate_pickem_result(
    pickems: Mapping[str, Iterable[str]],
    result_rows: Iterable[Mapping[str, Any]],
    pass_threshold: int = 5,
) -> Dict[str, object]:
    normalized_pickems = {category: [str(team) for team in teams] for category, teams in pickems.items()}
    standings = {_team_name(row.get("team") or row.get("name") or row.get("team_name")): row for row in result_rows}
    category_scores: Dict[str, Dict[str, object]] = {}
    correct_picks: List[Dict[str, str]] = []
    missed_picks: List[Dict[str, str]] = []

    for category in PICKEM_CATEGORIES:
        teams = normalized_pickems.get(category, [])
        correct = []
        missed = []
        for team in teams:
            row = standings.get(_team_name(team))
            matched = _category_matches(category, row)
            item = {"category": category, "team": team}
            if matched:
                correct.append(team)
                correct_picks.append(item)
            else:
                missed.append(team)
                missed_picks.append(item)
        category_scores[category] = {
            "correct": len(correct),
            "total": len(teams),
            "teams_correct": correct,
            "teams_missed": missed,
        }

    total_picks = sum(score["total"] for score in category_scores.values())
    correct_count = len(correct_picks)
    return {
        "correct": correct_count,
        "total_picks": total_picks,
        "accuracy": correct_count / total_picks if total_picks else 0.0,
        "pass_threshold": pass_threshold,
        "passed": correct_count >= pass_threshold,
        "category_scores": category_scores,
        "correct_picks": correct_picks,
        "missed_picks": missed_picks,
    }


def backtest_pickem_file(
    pickems_path: str,
    results_path: str,
    output_path: str | None = None,
    pass_threshold: int = 5,
) -> Dict[str, object]:
    with open(pickems_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    pickems = _extract_pickems(payload)
    report = evaluate_pickem_result(pickems, read_matches_csv(results_path), pass_threshold=pass_threshold)
    report["pickems_path"] = pickems_path
    report["results_path"] = results_path
    if output_path:
        write_json(output_path, report)
    return report


def evaluate_pickem_checkpoint(
    pickems: Mapping[str, Iterable[str]],
    standings_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, object]:
    normalized_pickems = {category: [str(team) for team in teams] for category, teams in pickems.items()}
    standings = {_team_name(row.get("team") or row.get("name") or row.get("team_name")): row for row in standings_rows}
    pick_reports: List[Dict[str, object]] = []
    summary = {"locked": 0, "alive": 0, "broken": 0, "missing": 0}
    for category in PICKEM_CATEGORIES:
        for team in normalized_pickems.get(category, []):
            row = standings.get(_team_name(team))
            status = _checkpoint_status(category, row)
            summary[status] += 1
            pick_reports.append(
                {
                    "category": category,
                    "team": team,
                    "wins": _int(row.get("wins")) if row else None,
                    "losses": _int(row.get("losses")) if row else None,
                    "status": status,
                }
            )
    return {"summary": summary, "picks": pick_reports}


def checkpoint_pickem_file(
    pickems_path: str,
    standings_path: str,
    output_path: str | None = None,
) -> Dict[str, object]:
    with open(pickems_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    report = evaluate_pickem_checkpoint(_extract_pickems(payload), read_matches_csv(standings_path))
    report["pickems_path"] = pickems_path
    report["standings_path"] = standings_path
    if output_path:
        write_json(output_path, report)
    return report


def evaluate_forecast_result(
    predictions: Iterable[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, object]:
    materialized_predictions = [dict(row) for row in predictions]
    lookup = _forecast_result_lookup(result_rows)
    match_reports: List[Dict[str, object]] = []
    unmatched_predictions: List[Dict[str, object]] = []

    for prediction in materialized_predictions:
        result = _lookup_forecast_result(prediction, lookup)
        if result is None:
            unmatched_predictions.append(_forecast_prediction_identity(prediction))
            continue
        team1 = _team_name(prediction.get("team1"))
        team2 = _team_name(prediction.get("team2"))
        winner = _team_name(result.get("winner"))
        probability_team1 = _float(
            prediction.get("adjusted_probability_team1"),
            _float(prediction.get("model_probability_team1"), 0.5),
        )
        directional_pick = team1 if probability_team1 >= 0.5 else team2
        pick = _team_name(prediction.get("pick"))
        actionable = pick.lower() != "avoid" and bool(pick)
        correct = actionable and _team_key(pick) == _team_key(winner)
        directional_correct = _team_key(directional_pick) == _team_key(winner)
        player_form_diff = _player_form_diff(prediction)
        player_form_sample_confidence = _player_form_sample_confidence(prediction)
        confidence_margin = _float(prediction.get("confidence_margin"), abs(probability_team1 - 0.5))
        match_reports.append(
            {
                "date": prediction.get("date"),
                "team1": team1,
                "team2": team2,
                "winner": winner,
                "score": result.get("score"),
                "map": result.get("map"),
                "result_note": result.get("note"),
                "result_source": result.get("source"),
                "pick": pick or None,
                "actionable": actionable,
                "correct": correct if actionable else None,
                "directional_pick": directional_pick,
                "directional_correct": directional_correct,
                "adjusted_probability_team1": probability_team1,
                "confidence_margin": confidence_margin,
                "low_confidence": bool(prediction.get("low_confidence")),
                "market_adjustment_applied": bool(prediction.get("market_adjustment_applied")),
                "player_form_diff": player_form_diff,
                "player_form_sample_confidence": player_form_sample_confidence,
            }
        )

    actionable_matches = [row for row in match_reports if row["actionable"]]
    avoid_matches = [row for row in match_reports if not row["actionable"]]
    correct_actionable = sum(1 for row in actionable_matches if row["correct"])
    directional_correct = sum(1 for row in match_reports if row["directional_correct"])
    return {
        "forecast_predictions": len(materialized_predictions),
        "matched": len(match_reports),
        "unmatched": len(unmatched_predictions),
        "unmatched_predictions": unmatched_predictions,
        "actionable_picks": len(actionable_matches),
        "correct_actionable": correct_actionable,
        "missed_actionable": len(actionable_matches) - correct_actionable,
        "actionable_accuracy": correct_actionable / len(actionable_matches) if actionable_matches else 0.0,
        "avoid_picks": len(avoid_matches),
        "avoid_directional_correct": sum(1 for row in avoid_matches if row["directional_correct"]),
        "directional_correct": directional_correct,
        "directional_accuracy": directional_correct / len(match_reports) if match_reports else 0.0,
        "model_upsets": sum(1 for row in match_reports if not row["directional_correct"]),
        "low_confidence_avoids": sum(1 for row in avoid_matches if row["low_confidence"]),
        "market_adjusted_matches": sum(1 for row in match_reports if row["market_adjustment_applied"]),
        "player_form_diagnostics": _forecast_player_form_diagnostics(match_reports),
        "policy_diagnostics": _forecast_policy_diagnostics(match_reports),
        "matches": match_reports,
    }


def backtest_forecast_file(
    forecast_report_path: str,
    results_path: str,
    output_path: str | None = None,
) -> Dict[str, object]:
    with open(forecast_report_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    predictions = payload.get("predictions", payload)
    if not isinstance(predictions, list):
        raise ValueError("forecast report must be a list or an object with a predictions list")
    report = evaluate_forecast_result(predictions, read_matches_csv(results_path))
    report["forecast_report_path"] = forecast_report_path
    report["results_path"] = results_path
    if output_path:
        write_json(output_path, report)
    return report


def backtest_pickem_suite_file(
    suite_path: str,
    output_path: str | None = None,
    pass_threshold: int = 5,
    pass_rate_target: float = 0.38,
) -> Dict[str, object]:
    with open(suite_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_cases = payload.get("cases", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("pickem backtest suite must be a list or an object with a cases list")
    base_dir = os.path.dirname(os.path.abspath(suite_path))
    cases = [_materialize_suite_case(case, base_dir) for case in raw_cases]
    report = evaluate_pickem_backtest_suite(cases, pass_threshold=pass_threshold, pass_rate_target=pass_rate_target)
    report["suite_path"] = suite_path
    if output_path:
        write_json(output_path, report)
    return report


def replay_pickem_backtest_suite_file(
    suite_path: str,
    output_path: str | None = None,
    pass_threshold: int = 5,
    pass_rate_target: float = 0.38,
    simulations: int = 100000,
    top_k: int = 25,
    epochs: int = 50,
    max_age_days: int = 90,
) -> Dict[str, object]:
    with open(suite_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_cases = payload.get("cases", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("replay pickem backtest suite must be a list or an object with a cases list")
    base_dir = os.path.dirname(os.path.abspath(suite_path))
    cases = [_materialize_replay_case(case, base_dir) for case in raw_cases]
    report = replay_pickem_backtest_suite(
        cases,
        pass_threshold=pass_threshold,
        pass_rate_target=pass_rate_target,
        simulations=simulations,
        top_k=top_k,
        epochs=epochs,
        max_age_days=max_age_days,
    )
    report["suite_path"] = suite_path
    if output_path:
        write_json(output_path, report)
    return report


def evaluate_pickem_backtest_suite(
    cases: Iterable[Mapping[str, Any]],
    pass_threshold: int = 5,
    pass_rate_target: float = 0.38,
) -> Dict[str, object]:
    case_reports = []
    for index, case in enumerate(cases, start=1):
        report = evaluate_pickem_result(
            case.get("pickems", {}),
            case.get("results", []),
            pass_threshold=pass_threshold,
        )
        report["name"] = str(case.get("name") or f"case-{index}")
        if case.get("pickems_path"):
            report["pickems_path"] = str(case["pickems_path"])
        if case.get("results_path"):
            report["results_path"] = str(case["results_path"])
        case_reports.append(report)
    passed_cases = sum(1 for report in case_reports if report["passed"])
    total_cases = len(case_reports)
    pass_rate = passed_cases / total_cases if total_cases else 0.0
    return {
        "cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate": pass_rate,
        "pass_rate_target": pass_rate_target,
        "meets_pass_rate_target": pass_rate >= pass_rate_target,
        "case_reports": case_reports,
    }


def replay_pickem_backtest_suite(
    cases: Iterable[Mapping[str, Any]],
    pass_threshold: int = 5,
    pass_rate_target: float = 0.38,
    simulations: int = 100000,
    top_k: int = 25,
    epochs: int = 50,
    max_age_days: int = 90,
) -> Dict[str, object]:
    case_reports = []
    for index, case in enumerate(cases, start=1):
        name = str(case.get("name") or f"case-{index}")
        pickem_report = model_driven_pickems(
            history_rows=case.get("history", []),
            team_rows=case.get("teams", []),
            reference_date=str(case.get("reference_date") or ""),
            profiles=case.get("profiles"),
            simulations=int(case.get("simulations", simulations)),
            seed=int(case.get("seed", 13)),
            top_k=int(case.get("top_k", top_k)),
            epochs=int(case.get("epochs", epochs)),
            slots=case.get("slots"),
            stage=str(case.get("stage", "default")),
            max_age_days=int(case.get("max_age_days", max_age_days)),
            ensemble_weights=case.get("ensemble_weights"),
            fixture_rows=case.get("fixtures"),
        )
        score_report = evaluate_pickem_result(
            pickem_report.get("pickems", {}),
            case.get("results", []),
            pass_threshold=pass_threshold,
        )
        report = {
            "name": name,
            "passed": score_report["passed"],
            "correct": score_report["correct"],
            "total_picks": score_report["total_picks"],
            "generated_pickems": pickem_report.get("pickems", {}),
            "generated_summary": {
                "trained_matches": pickem_report.get("trained_matches"),
                "simulations": pickem_report.get("simulations"),
                "selected_feature_names": pickem_report.get("selected_feature_names", []),
                "ensemble_weights": pickem_report.get("ensemble_weights", {}),
                "probability_calibration": pickem_report.get("probability_calibration", {}),
                "market_adjustment_summary": pickem_report.get("market_adjustment_summary", {}),
            },
            "score_report": score_report,
        }
        for key in ("history_path", "teams_path", "fixtures_path", "profiles_path", "results_path"):
            if case.get(key):
                report[key] = str(case[key])
        case_reports.append(report)
    passed_cases = sum(1 for report in case_reports if report["passed"])
    total_cases = len(case_reports)
    pass_rate = passed_cases / total_cases if total_cases else 0.0
    return {
        "cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate": pass_rate,
        "pass_rate_target": pass_rate_target,
        "meets_pass_rate_target": pass_rate >= pass_rate_target,
        "case_reports": case_reports,
    }


def _materialize_suite_case(case: Mapping[str, Any], base_dir: str) -> Dict[str, object]:
    output: Dict[str, object] = {"name": str(case.get("name") or "")}
    if case.get("pickems_path"):
        pickems_path = _resolve_suite_path(str(case["pickems_path"]), base_dir)
        with open(pickems_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        output["pickems"] = _extract_pickems(payload)
        output["pickems_path"] = pickems_path
    else:
        output["pickems"] = _extract_pickems(case.get("pickems", {}))
    if case.get("results_path"):
        results_path = _resolve_suite_path(str(case["results_path"]), base_dir)
        output["results"] = read_matches_csv(results_path)
        output["results_path"] = results_path
    else:
        output["results"] = case.get("results", [])
    return output


def _materialize_replay_case(case: Mapping[str, Any], base_dir: str) -> Dict[str, object]:
    output: Dict[str, object] = {
        "name": str(case.get("name") or ""),
        "reference_date": str(case.get("reference_date") or ""),
    }
    _copy_optional_replay_settings(case, output)
    if case.get("history_path"):
        history_path = _resolve_suite_path(str(case["history_path"]), base_dir)
        output["history"] = read_matches_csv(history_path)
        output["history_path"] = history_path
    else:
        output["history"] = case.get("history", [])
    if case.get("teams_path"):
        teams_path = _resolve_suite_path(str(case["teams_path"]), base_dir)
        output["teams"] = read_teams_csv(teams_path)
        output["teams_path"] = teams_path
    else:
        output["teams"] = case.get("teams", [])
    if case.get("results_path"):
        results_path = _resolve_suite_path(str(case["results_path"]), base_dir)
        output["results"] = read_matches_csv(results_path)
        output["results_path"] = results_path
    else:
        output["results"] = case.get("results", [])
    if case.get("fixtures_path"):
        fixtures_path = _resolve_suite_path(str(case["fixtures_path"]), base_dir)
        output["fixtures"] = read_matches_csv(fixtures_path)
        output["fixtures_path"] = fixtures_path
    else:
        output["fixtures"] = case.get("fixtures")
    if case.get("profiles_path"):
        profiles_path = _resolve_suite_path(str(case["profiles_path"]), base_dir)
        with open(profiles_path, encoding="utf-8") as handle:
            output["profiles"] = json.load(handle)
        output["profiles_path"] = profiles_path
    else:
        output["profiles"] = case.get("profiles")
    return output


def _copy_optional_replay_settings(case: Mapping[str, Any], output: Dict[str, object]) -> None:
    for key in (
        "simulations",
        "seed",
        "top_k",
        "epochs",
        "slots",
        "stage",
        "max_age_days",
        "ensemble_weights",
    ):
        if key in case:
            output[key] = case[key]


def _resolve_suite_path(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))


def _checkpoint_status(category: str, row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "missing"
    wins = _int(row.get("wins"))
    losses = _int(row.get("losses"))
    if category == "3-0":
        if wins >= 3 and losses == 0:
            return "locked"
        return "broken" if losses > 0 else "alive"
    if category == "advance":
        if wins >= 3:
            return "locked"
        return "broken" if losses >= 3 else "alive"
    if category == "0-3":
        if losses >= 3 and wins == 0:
            return "locked"
        return "broken" if wins > 0 else "alive"
    return "missing"


def _forecast_result_lookup(
    result_rows: Iterable[Mapping[str, Any]],
) -> Dict[tuple[str, str, str], List[Mapping[str, Any]]]:
    lookup: Dict[tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in result_rows:
        key = _forecast_match_key(row)
        lookup.setdefault(key, []).append(row)
        if key[0]:
            fallback_key = ("", key[1], key[2])
            lookup.setdefault(fallback_key, []).append(row)
    return lookup


def _lookup_forecast_result(
    prediction: Mapping[str, Any],
    lookup: Mapping[tuple[str, str, str], List[Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    prediction_key = _forecast_match_key(prediction)
    for key in (prediction_key, ("", prediction_key[1], prediction_key[2])):
        candidates = lookup.get(key, [])
        if len(candidates) == 1:
            return candidates[0]
    return None


def _forecast_match_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    teams = sorted((_team_key(row.get("team1")), _team_key(row.get("team2"))))
    return (str(row.get("date") or "")[:10], teams[0], teams[1])


def _forecast_prediction_identity(prediction: Mapping[str, Any]) -> Dict[str, object]:
    return {
        "date": prediction.get("date"),
        "team1": prediction.get("team1"),
        "team2": prediction.get("team2"),
    }


def _forecast_player_form_diagnostics(match_reports: Iterable[Mapping[str, Any]]) -> Dict[str, object]:
    materialized = list(match_reports)
    correct_actionable = [row for row in materialized if row.get("actionable") and row.get("correct")]
    missed_actionable = [row for row in materialized if row.get("actionable") and not row.get("correct")]
    directional_correct = [row for row in materialized if row.get("directional_correct")]
    directional_missed = [row for row in materialized if not row.get("directional_correct")]
    return {
        "available_matches": sum(1 for row in materialized if row.get("player_form_diff")),
        "correct_actionable_avg_score_diff": _avg_player_form_value(correct_actionable, "score"),
        "missed_actionable_avg_score_diff": _avg_player_form_value(missed_actionable, "score"),
        "directional_correct_avg_score_diff": _avg_player_form_value(directional_correct, "score"),
        "directional_missed_avg_score_diff": _avg_player_form_value(directional_missed, "score"),
        "directional_missed_avg_trend_diff": _avg_player_form_value(directional_missed, "trend"),
        "directional_missed_avg_sample_confidence_diff": _avg_player_form_value(directional_missed, "sample_confidence"),
    }


def _forecast_policy_diagnostics(match_reports: Iterable[Mapping[str, Any]]) -> Dict[str, object]:
    materialized = list(match_reports)
    current_actionable = [row for row in materialized if row.get("actionable")]
    threshold_candidates = [
        _forecast_threshold_candidate(materialized, threshold)
        for threshold in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15)
    ]
    recommended = _recommended_threshold_candidate(threshold_candidates)
    return {
        "current_policy": {
            "actionable_picks": len(current_actionable),
            "correct_actionable": sum(1 for row in current_actionable if row.get("correct")),
            "actionable_accuracy": (
                sum(1 for row in current_actionable if row.get("correct")) / len(current_actionable)
                if current_actionable
                else 0.0
            ),
        },
        "threshold_candidates": threshold_candidates,
        "recommended_minimum_margin": recommended.get("minimum_margin"),
        "recommendation_basis": (
            "highest_accuracy_with_minimum_two_picks"
            if recommended.get("actionable_picks", 0) >= 2
            else "insufficient_actionable_sample"
        ),
        "player_form_counter_signal": _player_form_counter_signal_risk(materialized),
        "player_form_policy_candidates": _player_form_policy_candidates(materialized),
    }


def _forecast_threshold_candidate(rows: Iterable[Mapping[str, Any]], minimum_margin: float) -> Dict[str, object]:
    materialized = list(rows)
    selected_indexes = {
        index
        for index, row in enumerate(materialized)
        if _float(row.get("confidence_margin"), 0.0) >= minimum_margin
    }
    selected = [row for index, row in enumerate(materialized) if index in selected_indexes]
    correct = sum(1 for row in selected if row.get("directional_correct"))
    avoided = [row for index, row in enumerate(materialized) if index not in selected_indexes]
    return {
        "minimum_margin": minimum_margin,
        "actionable_picks": len(selected),
        "correct_actionable": correct,
        "missed_actionable": len(selected) - correct,
        "actionable_accuracy": correct / len(selected) if selected else 0.0,
        "coverage": len(selected) / len(materialized) if materialized else 0.0,
        "avoided_wins": sum(1 for row in avoided if row.get("directional_correct")),
        "avoided_losses": sum(1 for row in avoided if not row.get("directional_correct")),
    }


def _recommended_threshold_candidate(candidates: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [row for row in candidates if row.get("actionable_picks", 0) >= 2]
    if not eligible:
        return {"minimum_margin": None, "actionable_picks": 0}
    return max(
        eligible,
        key=lambda row: (
            _float(row.get("actionable_accuracy"), 0.0),
            int(row.get("correct_actionable", 0)),
            -_float(row.get("minimum_margin"), 0.0),
        ),
    )


def _player_form_counter_signal_risk(rows: Iterable[Mapping[str, Any]]) -> Dict[str, object]:
    available = []
    counter_signal = []
    aligned = []
    for row in rows:
        diff = row.get("player_form_diff")
        if not isinstance(diff, Mapping) or not diff:
            continue
        available.append(row)
        directional_score = _directional_player_form_score(row, diff)
        if directional_score < 0:
            counter_signal.append(row)
        else:
            aligned.append(row)
    counter_losses = sum(1 for row in counter_signal if not row.get("directional_correct"))
    aligned_losses = sum(1 for row in aligned if not row.get("directional_correct"))
    return {
        "available_matches": len(available),
        "counter_signal_matches": len(counter_signal),
        "counter_signal_losses": counter_losses,
        "counter_signal_loss_rate": counter_losses / len(counter_signal) if counter_signal else 0.0,
        "aligned_matches": len(aligned),
        "aligned_losses": aligned_losses,
        "aligned_loss_rate": aligned_losses / len(aligned) if aligned else 0.0,
    }


def _player_form_policy_candidates(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, object]]:
    materialized = list(rows)
    return [
        _player_form_policy_candidate(materialized, min_confidence)
        for min_confidence in PLAYER_FORM_COUNTER_CONFIDENCE_CANDIDATES
    ]


def _player_form_policy_candidate(
    rows: Iterable[Mapping[str, Any]],
    min_confidence: float,
) -> Dict[str, object]:
    materialized = list(rows)
    avoided_indexes = set()
    counter_signal_matches = 0
    for index, row in enumerate(materialized):
        diff = row.get("player_form_diff")
        if not isinstance(diff, Mapping) or not diff:
            continue
        if _float(row.get("player_form_sample_confidence"), 0.0) < min_confidence:
            continue
        if _directional_player_form_score(row, diff) < 0:
            counter_signal_matches += 1
            avoided_indexes.add(index)
    selected = [row for index, row in enumerate(materialized) if index not in avoided_indexes]
    avoided = [row for index, row in enumerate(materialized) if index in avoided_indexes]
    correct = sum(1 for row in selected if row.get("directional_correct"))
    return {
        "player_form_counter_min_confidence": min_confidence,
        "counter_signal_matches": counter_signal_matches,
        "actionable_picks": len(selected),
        "correct_actionable": correct,
        "missed_actionable": len(selected) - correct,
        "actionable_accuracy": correct / len(selected) if selected else 0.0,
        "coverage": len(selected) / len(materialized) if materialized else 0.0,
        "avoided_wins": sum(1 for row in avoided if row.get("directional_correct")),
        "avoided_losses": sum(1 for row in avoided if not row.get("directional_correct")),
    }


def _directional_player_form_score(row: Mapping[str, Any], diff: Mapping[str, Any]) -> float:
    score = _float(diff.get("score"), 0.0)
    if _team_key(row.get("directional_pick")) == _team_key(row.get("team2")):
        return -score
    return score


def _avg_player_form_value(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        diff = row.get("player_form_diff")
        if isinstance(diff, Mapping) and diff.get(key) not in (None, ""):
            values.append(_float(diff.get(key), 0.0))
    return sum(values) / len(values) if values else None


def _player_form_diff(prediction: Mapping[str, Any]) -> Dict[str, float]:
    summary = prediction.get("player_form_summary")
    if not isinstance(summary, Mapping):
        return {}
    diff = summary.get("diff")
    if not isinstance(diff, Mapping):
        return {}
    return {
        "score": _float(diff.get("score"), 0.0),
        "trend": _float(diff.get("trend"), 0.0),
        "sample_confidence": _float(diff.get("sample_confidence"), 0.0),
    }


def _player_form_sample_confidence(prediction: Mapping[str, Any]) -> float:
    summary = prediction.get("player_form_summary")
    if not isinstance(summary, Mapping):
        return 0.0
    team1 = summary.get("team1")
    team2 = summary.get("team2")
    if isinstance(team1, Mapping) and isinstance(team2, Mapping):
        return min(
            _float(team1.get("sample_confidence"), 0.0),
            _float(team2.get("sample_confidence"), 0.0),
        )
    diff = summary.get("diff")
    if isinstance(diff, Mapping):
        return abs(_float(diff.get("sample_confidence"), 0.0))
    return 0.0


def _extract_pickems(payload: Any) -> Dict[str, List[str]]:
    if not isinstance(payload, Mapping):
        raise ValueError("pickems payload must be a JSON object")
    if isinstance(payload.get("pickems"), Mapping):
        return _normalize_pickem_mapping(payload["pickems"])
    if "picks" in payload:
        picks = payload["picks"]
        if isinstance(picks, Mapping):
            return _normalize_pickem_mapping(picks)
        if isinstance(picks, list):
            return _normalize_pickem_rows(picks)
    return _normalize_pickem_mapping(payload)


def _normalize_pickem_mapping(raw_pickems: Mapping[str, Any]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for category in PICKEM_CATEGORIES:
        normalized[category] = []
        for item in _pickem_items(raw_pickems.get(category, [])):
            team = _pickem_team(item)
            if team:
                normalized[category].append(team)
    return normalized


def _normalize_pickem_rows(rows: Iterable[Any]) -> Dict[str, List[str]]:
    normalized = {category: [] for category in PICKEM_CATEGORIES}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        category = str(row.get("category") or "").strip()
        if category not in normalized:
            continue
        team = _pickem_team(row)
        if team:
            normalized[category].append(team)
    return normalized


def _pickem_items(value: Any) -> Iterable[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable):
        return value
    return []


def _pickem_team(item: Any) -> str:
    if isinstance(item, Mapping):
        return _team_name(item.get("team") or item.get("name") or item.get("team_name"))
    return _team_name(item)


def _category_matches(category: str, row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    wins = _int(row.get("wins", row.get("final_wins", 0)))
    losses = _int(row.get("losses", row.get("final_losses", 0)))
    if category == "3-0":
        return wins >= 3 and losses == 0
    if category == "advance":
        return wins >= 3
    if category == "0-3":
        return wins == 0 and losses >= 3
    return False


def _team_name(value: Any) -> str:
    return str(value or "").strip()


def _team_key(value: Any) -> str:
    return _team_name(value).lower().replace(" ", "")


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
