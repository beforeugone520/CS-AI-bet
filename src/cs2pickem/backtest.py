from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping

from .data import read_matches_csv, read_teams_csv, write_json, write_matches_csv
from .pickem import model_driven_pickems


PICKEM_CATEGORIES = ("3-0", "advance", "0-3")
PLAYER_FORM_COUNTER_CONFIDENCE_CANDIDATES = (0.0, 0.2, 0.4, 0.6, 0.8)
FAVORITE_UPSET_MIN_PROBABILITY = 0.55
MARKET_FAVORITE_FORM_COUNTER_PROBABILITY_CANDIDATES = (0.55, 0.60, 0.65, 0.70)
PLAYER_STATUS_CONFIDENCE_CANDIDATES = (0.2, 0.4, 0.6)
PLAYER_STATUS_MARGIN_CANDIDATES = (0.06, 0.08, 0.10)
BO1_MINIMUM_MARGIN_CANDIDATES = (0.02, 0.05, 0.08, 0.10, 0.12, 0.15)


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
    pick_details: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    candidate_scoreboard: Mapping[str, Any] | None = None,
) -> Dict[str, object]:
    normalized_pickems = {category: [str(team) for team in teams] for category, teams in pickems.items()}
    standings = {_team_name(row.get("team") or row.get("name") or row.get("team_name")): row for row in standings_rows}
    pick_reports: List[Dict[str, object]] = []
    summary = {"locked": 0, "alive": 0, "broken": 0, "missing": 0}
    detail_lookup = pick_details or {}
    for category in PICKEM_CATEGORIES:
        for team in normalized_pickems.get(category, []):
            row = standings.get(_team_name(team))
            status = _checkpoint_status(category, row)
            summary[status] += 1
            pick_report = {
                "category": category,
                "team": team,
                "wins": _int(row.get("wins")) if row else None,
                "losses": _int(row.get("losses")) if row else None,
                "status": status,
            }
            pick_report.update(_checkpoint_detail_fields(detail_lookup.get((category, _team_key(team)), {})))
            pick_report.update(_checkpoint_pressure_fields(category, status, row))
            pick_reports.append(pick_report)
    report = {
        "summary": summary,
        "status_diagnostics": _checkpoint_status_diagnostics(pick_reports),
        "category_diagnostics": _checkpoint_category_diagnostics(pick_reports),
        "picks": pick_reports,
    }
    if candidate_scoreboard is not None:
        candidate_reports = _checkpoint_candidate_scoreboard(candidate_scoreboard, standings)
        report["candidate_scoreboard_checkpoint"] = candidate_reports
        report["candidate_scoreboard_diagnostics"] = _candidate_scoreboard_diagnostics(candidate_reports)
        report["candidate_scoreboard_policy_diagnostics"] = _candidate_scoreboard_policy_diagnostics(candidate_reports)
    return report


def checkpoint_pickem_file(
    pickems_path: str,
    standings_path: str,
    output_path: str | None = None,
) -> Dict[str, object]:
    with open(pickems_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    report = evaluate_pickem_checkpoint(
        _extract_pickems(payload),
        read_matches_csv(standings_path),
        pick_details=_pickem_detail_lookup(payload),
        candidate_scoreboard=_extract_candidate_scoreboard(payload),
    )
    report["pickems_path"] = pickems_path
    report["standings_path"] = standings_path
    if output_path:
        write_json(output_path, report)
    return report


def standings_from_results(
    result_rows: Iterable[Mapping[str, Any]],
    source_label: str | None = None,
) -> List[Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    for row in result_rows:
        team1 = _team_name(row.get("team1"))
        team2 = _team_name(row.get("team2"))
        winner = _team_name(row.get("winner"))
        if not team1 or not team2 or not winner:
            continue
        if _team_key(winner) == _team_key(team1):
            loser = team2
        elif _team_key(winner) == _team_key(team2):
            loser = team1
        else:
            continue
        winner_record = _ensure_record(records, winner)
        loser_record = _ensure_record(records, loser)
        winner_record["wins"] = int(winner_record["wins"]) + 1
        loser_record["losses"] = int(loser_record["losses"]) + 1
    rows = []
    for record in records.values():
        wins = int(record["wins"])
        losses = int(record["losses"])
        row = {
            "team": record["team"],
            "wins": wins,
            "losses": losses,
            "status": _record_status(wins, losses),
        }
        if source_label:
            row["source"] = source_label
        rows.append(row)
    return sorted(rows, key=lambda row: (-int(row["wins"]), int(row["losses"]), str(row["team"]).lower()))


def standings_from_results_file(
    results_path: str,
    output_path: str | None = None,
    source_label: str | None = None,
) -> List[Dict[str, object]]:
    rows = standings_from_results(read_matches_csv(results_path), source_label=source_label)
    if output_path:
        write_matches_csv(output_path, rows)
    return rows


def merge_standings_into_fixtures(
    fixture_rows: Iterable[Mapping[str, Any]],
    standings_rows: Iterable[Mapping[str, Any]],
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    standings = {
        _team_key(row.get("team") or row.get("name") or row.get("team_name")): row
        for row in standings_rows
    }
    merged: List[Dict[str, object]] = []
    matched_fixtures = 0
    partially_matched_fixtures = 0
    unmatched_sides = 0
    for fixture in fixture_rows:
        copied = dict(fixture)
        matched_sides = 0
        rounds_played = []
        sources = set()
        for prefix in ("team1", "team2"):
            team = _team_name(copied.get(prefix))
            standing = standings.get(_team_key(team))
            if standing is None:
                unmatched_sides += 1
                continue
            wins = _int(standing.get("wins"))
            losses = _int(standing.get("losses"))
            copied[f"{prefix}_wins"] = wins
            copied[f"{prefix}_losses"] = losses
            copied[f"{prefix}_record"] = f"{wins}-{losses}"
            copied[f"{prefix}_record_status"] = str(standing.get("status") or _record_status(wins, losses))
            rounds_played.append(wins + losses)
            matched_sides += 1
            if standing.get("source"):
                sources.add(str(standing["source"]))
        if rounds_played:
            copied["swiss_round"] = max(rounds_played) + 1
        if sources:
            copied["standings_source"] = "+".join(sorted(sources))
        copied["swiss_match_type"] = _swiss_match_type(copied)
        if matched_sides == 2:
            matched_fixtures += 1
        elif matched_sides == 1:
            partially_matched_fixtures += 1
        merged.append(copied)
    return merged, {
        "fixtures": len(merged),
        "matched_fixtures": matched_fixtures,
        "partially_matched_fixtures": partially_matched_fixtures,
        "unmatched_sides": unmatched_sides,
        "standings_teams": len(standings),
    }


def merge_standings_file(
    fixtures_path: str,
    standings_path: str,
    output_path: str,
) -> Dict[str, object]:
    merged, report = merge_standings_into_fixtures(
        read_matches_csv(fixtures_path),
        read_matches_csv(standings_path),
    )
    write_matches_csv(output_path, merged)
    return report


def evaluate_forecast_result(
    predictions: Iterable[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    decision_policy: Mapping[str, Any] | None = None,
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
        avoid_reason = _forecast_avoid_reason(prediction, actionable)
        player_form_diff = _player_form_diff(prediction)
        player_form_sample_confidence = _player_form_sample_confidence(prediction)
        confidence_margin = _float(prediction.get("confidence_margin"), abs(probability_team1 - 0.5))
        model_probability_team1 = _optional_probability(prediction.get("model_probability_team1"))
        market_probability_team1 = _market_probability_team1(prediction)
        adjusted_favorite = _favorite_from_probability(probability_team1, team1, team2)
        model_favorite = _favorite_from_probability(model_probability_team1, team1, team2)
        market_favorite = _favorite_from_probability(market_probability_team1, team1, team2)
        player_form_directional_score = (
            _directional_player_form_score(
                {"team2": team2, "directional_pick": directional_pick},
                player_form_diff,
            )
            if player_form_diff
            else None
        )
        picked_player_status = _picked_player_status(prediction, directional_pick, team1, team2)
        match_reports.append(
            {
                "date": prediction.get("date"),
                "team1": team1,
                "team2": team2,
                "best_of": _best_of(prediction),
                "swiss_round": prediction.get("swiss_round"),
                "team1_record": prediction.get("team1_record"),
                "team2_record": prediction.get("team2_record"),
                "team1_wins": prediction.get("team1_wins"),
                "team1_losses": prediction.get("team1_losses"),
                "team2_wins": prediction.get("team2_wins"),
                "team2_losses": prediction.get("team2_losses"),
                "swiss_match_type": _forecast_swiss_match_type(prediction),
                "winner": winner,
                "score": result.get("score"),
                "map": result.get("map"),
                "result_note": result.get("note"),
                "result_source": result.get("source"),
                "pick": pick or None,
                "actionable": actionable,
                "avoid_reason": avoid_reason,
                "correct": correct if actionable else None,
                "directional_pick": directional_pick,
                "directional_correct": directional_correct,
                "adjusted_probability_team1": probability_team1,
                "confidence_margin": confidence_margin,
                "model_probability_team1": model_probability_team1,
                "market_probability_team1": market_probability_team1,
                "adjusted_favorite": adjusted_favorite["team"],
                "adjusted_favorite_probability": adjusted_favorite["probability"],
                "model_favorite": model_favorite["team"] if model_favorite else None,
                "model_favorite_probability": model_favorite["probability"] if model_favorite else None,
                "market_favorite": market_favorite["team"] if market_favorite else None,
                "market_favorite_probability": market_favorite["probability"] if market_favorite else None,
                "low_confidence": bool(prediction.get("low_confidence")),
                "market_adjustment_applied": bool(prediction.get("market_adjustment_applied")),
                "player_form_diff": player_form_diff,
                "player_form_sample_confidence": player_form_sample_confidence,
                "player_form_directional_score": player_form_directional_score,
                "picked_player_sample_confidence": picked_player_status.get("sample_confidence"),
                "picked_substitute_flag": picked_player_status.get("substitute_flag"),
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
        "avoid_reason_diagnostics": _forecast_avoid_reason_diagnostics(match_reports),
        "player_form_diagnostics": _forecast_player_form_diagnostics(match_reports),
        "favorite_upset_diagnostics": _forecast_favorite_upset_diagnostics(match_reports),
        "swiss_pressure_diagnostics": _forecast_swiss_pressure_diagnostics(match_reports),
        "policy_diagnostics": _forecast_policy_diagnostics(match_reports, decision_policy=decision_policy),
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
    decision_policy = payload.get("decision_policy") if isinstance(payload, Mapping) else None
    report = evaluate_forecast_result(
        predictions,
        read_matches_csv(results_path),
        decision_policy=decision_policy if isinstance(decision_policy, Mapping) else None,
    )
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


def _checkpoint_detail_fields(detail: Mapping[str, Any]) -> Dict[str, object]:
    allowed = (
        "confidence",
        "tier",
        "signals_agree",
        "expert_votes",
        "market_win_prob_r1",
        "model",
        "raw_fused_score",
        "player_availability_multiplier",
        "status_adjusted_score",
        "player_status_risk",
        "player_sample_confidence",
        "substitute_flag",
        "player_form_score",
        "player_form_trend",
    )
    return {key: detail[key] for key in allowed if key in detail}


def _checkpoint_candidate_scoreboard(
    candidate_scoreboard: Mapping[str, Any],
    standings: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, object]]:
    candidate_reports: List[Dict[str, object]] = []
    if not isinstance(candidate_scoreboard, Mapping):
        return candidate_reports
    for category in PICKEM_CATEGORIES:
        for item in _pickem_items(candidate_scoreboard.get(category, [])):
            if not isinstance(item, Mapping):
                continue
            team = _pickem_team(item)
            if not team:
                continue
            row = standings.get(_team_name(team))
            status = _checkpoint_status(category, row)
            candidate_report = dict(item)
            candidate_report.update(
                {
                    "category": category,
                    "team": team,
                    "wins": _int(row.get("wins")) if row else None,
                    "losses": _int(row.get("losses")) if row else None,
                    "status": status,
                }
            )
            candidate_report.update(_checkpoint_pressure_fields(category, status, row))
            candidate_reports.append(candidate_report)
    return candidate_reports


def _candidate_scoreboard_diagnostics(
    candidates: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, object]]:
    diagnostics: Dict[str, Dict[str, object]] = {}
    materialized = list(candidates)
    for category in PICKEM_CATEGORIES:
        category_rows = [row for row in materialized if row.get("category") == category]
        selected_rows = [row for row in category_rows if _truthy(row.get("selected"))]
        unselected_rows = [row for row in category_rows if not _truthy(row.get("selected"))]
        locked_rows = [row for row in category_rows if row.get("status") == "locked"]
        unselected_locked_rows = [row for row in unselected_rows if row.get("status") == "locked"]
        selected_broken_rows = [row for row in selected_rows if row.get("status") == "broken"]
        best_unselected_rank = _best_adjusted_rank(unselected_locked_rows)
        row: Dict[str, object] = {
            "candidates": len(category_rows),
            "selected_candidates": len(selected_rows),
            "locked_candidates": len(locked_rows),
            "selected_locked_candidates": sum(
                1 for candidate in selected_rows if candidate.get("status") == "locked"
            ),
            "unselected_locked_candidates": len(unselected_locked_rows),
            "selected_broken_candidates": len(selected_broken_rows),
            "best_unselected_locked_adjusted_rank": best_unselected_rank,
            "unselected_locked_teams": [str(candidate.get("team")) for candidate in unselected_locked_rows],
        }
        for status in ("locked", "alive", "broken", "missing"):
            status_rows = [candidate for candidate in category_rows if candidate.get("status") == status]
            row[status] = len(status_rows)
        diagnostics[category] = row
    return diagnostics


def _candidate_scoreboard_policy_diagnostics(
    candidates: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, object]]:
    diagnostics: Dict[str, Dict[str, object]] = {}
    materialized = list(candidates)
    for category in PICKEM_CATEGORIES:
        category_rows = [row for row in materialized if row.get("category") == category]
        slot_count = sum(1 for row in category_rows if _truthy(row.get("selected")))
        policy_reports = [
            _candidate_policy_report(category, category_rows, slot_count, policy)
            for policy in _candidate_policy_names()
        ]
        diagnostics[category] = {
            "slot_count": slot_count,
            "policies": policy_reports,
            "recommendation": _candidate_policy_recommendation(policy_reports),
        }
    return diagnostics


def _candidate_policy_names() -> tuple[str, ...]:
    return (
        "status_adjusted_score",
        "raw_fused_score",
        "confidence",
        "expert_category_votes",
        "model_category_probability",
        "market_category_signal",
        "extreme_consensus_composite",
        "status_model_market_composite",
    )


def _candidate_policy_report(
    category: str,
    rows: Iterable[Mapping[str, Any]],
    slot_count: int,
    policy: str,
) -> Dict[str, object]:
    ranked = _rank_candidate_policy(category, list(rows), policy)
    top_rows = ranked[:slot_count]
    return {
        "policy": policy,
        "top_k": slot_count,
        "top_k_teams": [str(row.get("team")) for row in top_rows],
        "top_k_locked": sum(1 for row in top_rows if row.get("status") == "locked"),
        "top_k_alive": sum(1 for row in top_rows if row.get("status") == "alive"),
        "top_k_broken": sum(1 for row in top_rows if row.get("status") == "broken"),
        "top_k_missing": sum(1 for row in top_rows if row.get("status") == "missing"),
        "selected_overlap": sum(1 for row in top_rows if _truthy(row.get("selected"))),
        "locked_teams": [str(row.get("team")) for row in top_rows if row.get("status") == "locked"],
    }


def _candidate_policy_recommendation(
    policy_reports: List[Mapping[str, Any]],
) -> Dict[str, object]:
    baseline = next(
        (row for row in policy_reports if row.get("policy") == "status_adjusted_score"),
        None,
    )
    if baseline is None:
        return {
            "baseline_policy": "status_adjusted_score",
            "recommended_policy": None,
            "action": "insufficient_data",
            "locked_delta_vs_baseline": None,
            "broken_delta_vs_baseline": None,
            "reason": "baseline policy missing",
        }
    best = min(
        policy_reports,
        key=lambda row: _candidate_policy_recommendation_sort_key(row),
    )
    locked_delta = _int(best.get("top_k_locked")) - _int(baseline.get("top_k_locked"))
    broken_delta = _int(best.get("top_k_broken")) - _int(baseline.get("top_k_broken"))
    improves = (locked_delta > 0 and broken_delta <= 0) or (locked_delta >= 0 and broken_delta < 0)
    return {
        "baseline_policy": baseline.get("policy"),
        "recommended_policy": best.get("policy") if improves else baseline.get("policy"),
        "action": "review_candidate_policy" if improves else "keep_current_policy",
        "locked_delta_vs_baseline": locked_delta if improves else 0,
        "broken_delta_vs_baseline": broken_delta if improves else 0,
        "reason": _candidate_policy_recommendation_reason(best, baseline, improves),
    }


def _candidate_policy_recommendation_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        -_int(row.get("top_k_locked")),
        _int(row.get("top_k_broken")),
        -_int(row.get("top_k_alive")),
        _candidate_policy_preference_rank(str(row.get("policy") or "")),
    )


def _candidate_policy_preference_rank(policy: str) -> int:
    order = {
        "extreme_consensus_composite": 0,
        "status_model_market_composite": 1,
        "status_adjusted_score": 2,
        "raw_fused_score": 3,
        "confidence": 4,
        "expert_category_votes": 5,
        "market_category_signal": 6,
        "model_category_probability": 7,
    }
    return order.get(policy, 100)


def _candidate_policy_recommendation_reason(
    best: Mapping[str, Any],
    baseline: Mapping[str, Any],
    improves: bool,
) -> str:
    if improves:
        return (
            f"{best.get('policy')} improves top-k locked/broken profile versus "
            f"{baseline.get('policy')}"
        )
    return "no candidate policy improves the baseline locked/broken profile"


def _rank_candidate_policy(
    category: str,
    rows: List[Mapping[str, Any]],
    policy: str,
) -> List[Mapping[str, Any]]:
    if policy in _candidate_composite_policy_weights():
        scores = _candidate_composite_scores(category, rows, policy)
        return [
            row
            for index, row in sorted(
                enumerate(rows),
                key=lambda indexed: _candidate_policy_sort_key(
                    category,
                    policy,
                    indexed[1],
                    scores[indexed[0]],
                ),
            )
        ]
    return sorted(
        rows,
        key=lambda row: _candidate_policy_sort_key(category, policy, row),
    )


def _candidate_policy_sort_key(
    category: str,
    policy: str,
    row: Mapping[str, Any],
    score_override: float | None = None,
) -> tuple[bool, float, int, str]:
    score = score_override if score_override is not None else _candidate_policy_score(category, policy, row)
    adjusted_rank = _optional_int(row.get("adjusted_rank"))
    return (
        score is None,
        -score if score is not None else 0.0,
        adjusted_rank if adjusted_rank is not None else 1_000_000,
        _team_key(row.get("team")),
    )


def _candidate_composite_policy_weights() -> Dict[str, Dict[str, float]]:
    return {
        "extreme_consensus_composite": {
            "status_adjusted_score": 0.15,
            "confidence": 0.30,
            "expert_category_votes": 0.35,
            "market_category_signal": 0.20,
        },
        "status_model_market_composite": {
            "status_adjusted_score": 0.40,
            "model_category_probability": 0.35,
            "market_category_signal": 0.25,
        },
    }


def _candidate_composite_scores(
    category: str,
    rows: List[Mapping[str, Any]],
    policy: str,
) -> List[float | None]:
    weights = _candidate_composite_policy_weights()[policy]
    normalized = {
        signal: _candidate_normalized_signal_scores(category, rows, signal)
        for signal in weights
    }
    scores: List[float | None] = []
    for index in range(len(rows)):
        weighted_sum = 0.0
        weight_sum = 0.0
        for signal, weight in weights.items():
            value = normalized[signal][index]
            if value is None:
                continue
            weighted_sum += value * weight
            weight_sum += weight
        scores.append(weighted_sum / weight_sum if weight_sum else None)
    return scores


def _candidate_normalized_signal_scores(
    category: str,
    rows: List[Mapping[str, Any]],
    signal: str,
) -> List[float | None]:
    values = [_candidate_policy_score(category, signal, row) for row in rows]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return [None for _ in values]
    low = min(numeric)
    high = max(numeric)
    if high == low:
        return [0.5 if value is not None else None for value in values]
    return [
        (value - low) / (high - low) if value is not None else None
        for value in values
    ]


def _candidate_policy_score(
    category: str,
    policy: str,
    row: Mapping[str, Any],
) -> float | None:
    if policy in {"status_adjusted_score", "raw_fused_score", "confidence"}:
        return _optional_float(row.get(policy))
    if policy == "expert_category_votes":
        expert_votes = row.get("expert_votes")
        if isinstance(expert_votes, Mapping):
            return _optional_float(expert_votes.get(category))
        return None
    if policy == "model_category_probability":
        model = row.get("model")
        if isinstance(model, Mapping):
            return _optional_float(model.get(category))
        return None
    if policy == "market_category_signal":
        market_win = _optional_float(row.get("market_win_prob_r1"))
        if market_win is None:
            return None
        return 1.0 - market_win if category == "0-3" else market_win
    return None


def _best_adjusted_rank(candidates: Iterable[Mapping[str, Any]]) -> int | None:
    ranks = [
        rank
        for rank in (_optional_int(candidate.get("adjusted_rank")) for candidate in candidates)
        if rank is not None
    ]
    return min(ranks) if ranks else None


def _checkpoint_status_diagnostics(picks: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, object]]:
    diagnostics: Dict[str, Dict[str, object]] = {}
    materialized = list(picks)
    for status in ("locked", "alive", "broken", "missing"):
        status_rows = [row for row in materialized if row.get("status") == status]
        diagnostics[status] = {
            "picks": len(status_rows),
            "avg_confidence": _checkpoint_avg_confidence(status_rows),
        }
    return diagnostics


def _checkpoint_category_diagnostics(picks: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, object]]:
    diagnostics: Dict[str, Dict[str, object]] = {}
    materialized = list(picks)
    for category in PICKEM_CATEGORIES:
        category_rows = [row for row in materialized if row.get("category") == category]
        alive_rows = [row for row in category_rows if row.get("status") == "alive"]
        pressure_rows = [
            row
            for row in alive_rows
            if bool(row.get("next_match_can_lock")) or bool(row.get("next_match_can_break"))
        ]
        status_risk_rows = [row for row in category_rows if _truthy(row.get("player_status_risk"))]
        non_status_risk_rows = [row for row in category_rows if not _truthy(row.get("player_status_risk"))]
        broken_status_risk = sum(1 for pick in status_risk_rows if pick.get("status") == "broken")
        broken_non_status_risk = sum(1 for pick in non_status_risk_rows if pick.get("status") == "broken")
        row: Dict[str, object] = {
            "picks": len(category_rows),
            "avg_confidence": _checkpoint_avg_confidence(category_rows),
            "high_tier_broken": sum(
                1
                for pick in category_rows
                if pick.get("status") == "broken" and str(pick.get("tier") or "").lower() == "high"
            ),
            "player_status_risk_picks": len(status_risk_rows),
            "broken_player_status_risk": broken_status_risk,
            "player_status_risk_broken_rate": (
                broken_status_risk / len(status_risk_rows)
                if status_risk_rows
                else None
            ),
            "non_status_risk_broken_rate": (
                broken_non_status_risk / len(non_status_risk_rows)
                if non_status_risk_rows
                else None
            ),
            "alive_next_match_can_lock": sum(1 for pick in alive_rows if pick.get("next_match_can_lock")),
            "alive_next_match_can_break": sum(1 for pick in alive_rows if pick.get("next_match_can_break")),
            "alive_pressure_picks": len(pressure_rows),
            "alive_status_risk_pressure_picks": sum(
                1
                for pick in pressure_rows
                if _truthy(pick.get("player_status_risk"))
            ),
        }
        for status in ("locked", "alive", "broken", "missing"):
            status_rows = [pick for pick in category_rows if pick.get("status") == status]
            row[status] = len(status_rows)
            row[f"{status}_avg_confidence"] = _checkpoint_avg_confidence(status_rows)
        diagnostics[category] = row
    return diagnostics


def _checkpoint_pressure_fields(
    category: str,
    status: str,
    row: Mapping[str, Any] | None,
) -> Dict[str, object]:
    if row is None:
        return {
            "wins_to_lock": None,
            "losses_to_lock": None,
            "losses_to_break": None,
            "wins_to_break": None,
            "next_match_can_lock": False,
            "next_match_can_break": False,
        }
    wins = _int(row.get("wins"))
    losses = _int(row.get("losses"))
    if category == "3-0":
        return {
            "wins_to_lock": max(0, 3 - wins),
            "losses_to_lock": None,
            "losses_to_break": max(0, 1 - losses),
            "wins_to_break": None,
            "next_match_can_lock": status == "alive" and wins == 2,
            "next_match_can_break": status == "alive" and losses == 0,
        }
    if category == "advance":
        return {
            "wins_to_lock": max(0, 3 - wins),
            "losses_to_lock": None,
            "losses_to_break": max(0, 3 - losses),
            "wins_to_break": None,
            "next_match_can_lock": status == "alive" and wins == 2,
            "next_match_can_break": status == "alive" and losses == 2,
        }
    if category == "0-3":
        return {
            "wins_to_lock": None,
            "losses_to_lock": max(0, 3 - losses),
            "losses_to_break": None,
            "wins_to_break": max(0, 1 - wins),
            "next_match_can_lock": status == "alive" and losses == 2,
            "next_match_can_break": status == "alive" and wins == 0,
        }
    return {
        "wins_to_lock": None,
        "losses_to_lock": None,
        "losses_to_break": None,
        "wins_to_break": None,
        "next_match_can_lock": False,
        "next_match_can_break": False,
    }


def _checkpoint_avg_confidence(rows: Iterable[Mapping[str, Any]]) -> float | None:
    values = [
        _float(row.get("confidence"), 0.0)
        for row in rows
        if row.get("confidence") not in (None, "")
    ]
    return sum(values) / len(values) if values else None


def _pickem_detail_lookup(payload: Any) -> Dict[tuple[str, str], Mapping[str, Any]]:
    details: Dict[tuple[str, str], Dict[str, Any]] = {}
    if not isinstance(payload, Mapping):
        return details
    for category, item in _pickem_detail_items(payload):
        team = _pickem_team(item)
        if team and isinstance(item, Mapping):
            key = (category, _team_key(team))
            details.setdefault(key, {}).update(item)
    return details


def _extract_candidate_scoreboard(payload: Any) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    candidate_scoreboard = payload.get("candidate_scoreboard")
    if isinstance(candidate_scoreboard, Mapping):
        return candidate_scoreboard
    return None


def _pickem_detail_items(payload: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    raw = payload.get("picks", payload.get("pickems", payload))
    if isinstance(raw, Mapping):
        for category in PICKEM_CATEGORIES:
            for item in _pickem_items(raw.get(category, [])):
                yield category, item
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                category = str(item.get("category") or "").strip()
                if category in PICKEM_CATEGORIES:
                    yield category, item
    for detail_key in ("pickem_details", "pickem_risk_details"):
        detail_section = payload.get(detail_key)
        if not isinstance(detail_section, Mapping):
            continue
        for category in PICKEM_CATEGORIES:
            for item in _pickem_items(detail_section.get(category, [])):
                yield category, item


def _ensure_record(records: Dict[str, Dict[str, object]], team: str) -> Dict[str, object]:
    key = _team_key(team)
    if key not in records:
        records[key] = {"team": team, "wins": 0, "losses": 0}
    return records[key]


def _record_status(wins: int, losses: int) -> str:
    if wins >= 3:
        return "advanced"
    if losses >= 3:
        return "eliminated"
    return "alive"


def _swiss_match_type(row: Mapping[str, Any]) -> str:
    team1_wins = _optional_int(row.get("team1_wins"))
    team2_wins = _optional_int(row.get("team2_wins"))
    team1_losses = _optional_int(row.get("team1_losses"))
    team2_losses = _optional_int(row.get("team2_losses"))
    if None in (team1_wins, team2_wins, team1_losses, team2_losses):
        return "unknown"
    if team1_wins == 2 and team2_wins == 2:
        return "advancement"
    if team1_losses == 2 and team2_losses == 2:
        return "elimination"
    return "standard"


def _forecast_swiss_match_type(row: Mapping[str, Any]) -> str:
    raw = str(row.get("swiss_match_type") or "").strip().lower()
    if raw in {"advancement", "elimination", "standard"}:
        return raw
    return _swiss_match_type(row)


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


def _forecast_avoid_reason(prediction: Mapping[str, Any], actionable: bool) -> str | None:
    if actionable:
        return None
    reason = str(prediction.get("avoid_reason") or "").strip()
    if reason:
        return reason
    if prediction.get("low_confidence"):
        return "low_confidence"
    return "avoid"


def _forecast_avoid_reason_diagnostics(match_reports: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, object]]:
    diagnostics: Dict[str, Dict[str, object]] = {}
    for row in match_reports:
        if row.get("actionable"):
            continue
        reason = str(row.get("avoid_reason") or "avoid")
        bucket = diagnostics.setdefault(
            reason,
            {
                "avoid_picks": 0,
                "avoided_wins": 0,
                "avoided_losses": 0,
            },
        )
        bucket["avoid_picks"] = int(bucket["avoid_picks"]) + 1
        if row.get("directional_correct"):
            bucket["avoided_wins"] = int(bucket["avoided_wins"]) + 1
        else:
            bucket["avoided_losses"] = int(bucket["avoided_losses"]) + 1
    for bucket in diagnostics.values():
        avoid_picks = int(bucket["avoid_picks"])
        bucket["avoided_loss_rate"] = (
            int(bucket["avoided_losses"]) / avoid_picks
            if avoid_picks
            else 0.0
        )
    return diagnostics


def _forecast_favorite_upset_diagnostics(
    match_reports: Iterable[Mapping[str, Any]],
    minimum_probability: float = FAVORITE_UPSET_MIN_PROBABILITY,
) -> Dict[str, object]:
    materialized = list(match_reports)
    adjusted_favorites = _forecast_favorite_rows(materialized, "adjusted", minimum_probability)
    model_favorites = _forecast_favorite_rows(materialized, "model", minimum_probability)
    market_favorites = _forecast_favorite_rows(materialized, "market", minimum_probability)
    agree_favorites = [
        row
        for row in materialized
        if row.get("model_favorite")
        and row.get("market_favorite")
        and _team_key(row.get("model_favorite")) == _team_key(row.get("market_favorite"))
        and _float(row.get("model_favorite_probability"), 0.0) >= minimum_probability
        and _float(row.get("market_favorite_probability"), 0.0) >= minimum_probability
    ]
    adjusted_losses = [row for row in adjusted_favorites if _forecast_favorite_lost(row, "adjusted")]
    model_losses = [row for row in model_favorites if _forecast_favorite_lost(row, "model")]
    market_losses = [row for row in market_favorites if _forecast_favorite_lost(row, "market")]
    agree_losses = [row for row in agree_favorites if _forecast_favorite_lost(row, "model")]
    return {
        "minimum_favorite_probability": minimum_probability,
        "adjusted_favorites": len(adjusted_favorites),
        "adjusted_favorite_losses": len(adjusted_losses),
        "adjusted_favorite_loss_rate": (
            len(adjusted_losses) / len(adjusted_favorites)
            if adjusted_favorites
            else 0.0
        ),
        "model_favorites": len(model_favorites),
        "model_favorite_losses": len(model_losses),
        "model_favorite_loss_rate": (
            len(model_losses) / len(model_favorites)
            if model_favorites
            else 0.0
        ),
        "market_favorites": len(market_favorites),
        "market_favorite_losses": len(market_losses),
        "market_favorite_loss_rate": (
            len(market_losses) / len(market_favorites)
            if market_favorites
            else 0.0
        ),
        "model_market_agree_favorites": len(agree_favorites),
        "model_market_agree_favorite_losses": len(agree_losses),
        "favorite_losses_with_player_form_counter_signal": sum(
            1
            for row in adjusted_losses
            if row.get("player_form_directional_score") is not None
            and _float(row.get("player_form_directional_score"), 0.0) < 0
        ),
        "favorite_loss_examples": [
            _forecast_favorite_loss_example(row, "adjusted")
            for row in adjusted_losses[:5]
        ],
        "market_favorite_loss_examples": [
            _forecast_favorite_loss_example(row, "market")
            for row in market_losses[:5]
        ],
    }


def _forecast_favorite_rows(
    rows: Iterable[Mapping[str, Any]],
    kind: str,
    minimum_probability: float,
) -> List[Mapping[str, Any]]:
    favorite_key = f"{kind}_favorite"
    probability_key = f"{kind}_favorite_probability"
    return [
        row
        for row in rows
        if row.get(favorite_key)
        and _float(row.get(probability_key), 0.0) >= minimum_probability
    ]


def _forecast_favorite_lost(row: Mapping[str, Any], kind: str) -> bool:
    favorite = row.get(f"{kind}_favorite")
    if not favorite:
        return False
    return _team_key(favorite) != _team_key(row.get("winner"))


def _forecast_favorite_loss_example(row: Mapping[str, Any], kind: str) -> Dict[str, object]:
    return {
        "date": row.get("date"),
        "team1": row.get("team1"),
        "team2": row.get("team2"),
        "favorite": row.get(f"{kind}_favorite"),
        "favorite_probability": row.get(f"{kind}_favorite_probability"),
        "winner": row.get("winner"),
        "model_favorite": row.get("model_favorite"),
        "model_favorite_probability": row.get("model_favorite_probability"),
        "market_favorite": row.get("market_favorite"),
        "market_favorite_probability": row.get("market_favorite_probability"),
        "player_form_directional_score": row.get("player_form_directional_score"),
        "player_form_sample_confidence": row.get("player_form_sample_confidence"),
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


def _forecast_swiss_pressure_diagnostics(match_reports: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, object]]:
    materialized = list(match_reports)
    match_types = ["advancement", "elimination", "standard", "unknown"]
    observed_types = sorted(
        {
            _forecast_swiss_match_type(row)
            for row in materialized
            if _forecast_swiss_match_type(row) not in match_types
        }
    )
    return {
        match_type: _forecast_swiss_pressure_bucket(
            row
            for row in materialized
            if _forecast_swiss_match_type(row) == match_type
        )
        for match_type in [*match_types, *observed_types]
    }


def _forecast_swiss_pressure_bucket(rows: Iterable[Mapping[str, Any]]) -> Dict[str, object]:
    materialized = list(rows)
    actionable = [row for row in materialized if row.get("actionable")]
    avoid = [row for row in materialized if not row.get("actionable")]
    correct_actionable = sum(1 for row in actionable if row.get("correct"))
    directional_correct = sum(1 for row in materialized if row.get("directional_correct"))
    return {
        "matched": len(materialized),
        "actionable_picks": len(actionable),
        "correct_actionable": correct_actionable,
        "missed_actionable": len(actionable) - correct_actionable,
        "actionable_accuracy": correct_actionable / len(actionable) if actionable else 0.0,
        "avoid_picks": len(avoid),
        "low_confidence_avoids": sum(1 for row in avoid if row.get("low_confidence")),
        "directional_correct": directional_correct,
        "directional_accuracy": directional_correct / len(materialized) if materialized else 0.0,
        "avg_confidence_margin": _avg_numeric(materialized, "confidence_margin"),
    }


def _forecast_policy_diagnostics(
    match_reports: Iterable[Mapping[str, Any]],
    decision_policy: Mapping[str, Any] | None = None,
) -> Dict[str, object]:
    materialized = list(match_reports)
    current_actionable = [row for row in materialized if row.get("actionable")]
    current_policy = {
        "actionable_picks": len(current_actionable),
        "correct_actionable": sum(1 for row in current_actionable if row.get("correct")),
        "actionable_accuracy": (
            sum(1 for row in current_actionable if row.get("correct")) / len(current_actionable)
            if current_actionable
            else 0.0
        ),
        "coverage": len(current_actionable) / len(materialized) if materialized else 0.0,
    }
    threshold_candidates = [
        _forecast_threshold_candidate(materialized, threshold)
        for threshold in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15)
    ]
    player_form_policy_candidates = _player_form_policy_candidates(materialized)
    market_favorite_player_form_policy_candidates = _market_favorite_player_form_policy_candidates(materialized)
    player_status_policy_candidates = _player_status_policy_candidates(materialized)
    bo1_margin_policy_candidates = _bo1_margin_policy_candidates(materialized)
    recommended = _recommended_threshold_candidate(threshold_candidates)
    return {
        "current_policy": current_policy,
        "threshold_candidates": threshold_candidates,
        "bo1_margin_policy_candidates": bo1_margin_policy_candidates,
        "recommended_minimum_margin": recommended.get("minimum_margin"),
        "recommendation_basis": (
            "highest_accuracy_with_minimum_two_picks"
            if recommended.get("actionable_picks", 0) >= 2
            else "insufficient_actionable_sample"
        ),
        "player_form_counter_signal": _player_form_counter_signal_risk(materialized),
        "player_form_policy_candidates": player_form_policy_candidates,
        "market_favorite_player_form_policy_candidates": market_favorite_player_form_policy_candidates,
        "player_status_signal_risk": _player_status_signal_risk(materialized),
        "player_status_policy_candidates": player_status_policy_candidates,
        "policy_tradeoff_summary": _policy_tradeoff_summary(
            current_policy,
            threshold_candidates,
            player_form_policy_candidates,
            market_favorite_player_form_policy_candidates,
            player_status_policy_candidates,
            bo1_margin_policy_candidates,
            decision_policy=decision_policy,
        ),
    }


def _policy_tradeoff_summary(
    current_policy: Mapping[str, Any],
    threshold_candidates: Iterable[Mapping[str, Any]],
    player_form_policy_candidates: Iterable[Mapping[str, Any]],
    market_favorite_player_form_policy_candidates: Iterable[Mapping[str, Any]],
    player_status_policy_candidates: Iterable[Mapping[str, Any]],
    bo1_margin_policy_candidates: Iterable[Mapping[str, Any]],
    decision_policy: Mapping[str, Any] | None = None,
) -> Dict[str, object]:
    current = _policy_tradeoff_candidate("current_policy", current_policy, _current_policy_parameter(decision_policy))
    candidates = [current]
    candidates.extend(
        _policy_tradeoff_candidate(
            "threshold_candidates",
            row,
            {"minimum_margin": row.get("minimum_margin")},
        )
        for row in threshold_candidates
    )
    candidates.extend(
        _policy_tradeoff_candidate(
            "player_form_policy_candidates",
            row,
            {"player_form_counter_min_confidence": row.get("player_form_counter_min_confidence")},
        )
        for row in player_form_policy_candidates
    )
    candidates.extend(
        _policy_tradeoff_candidate(
            "market_favorite_player_form_policy_candidates",
            row,
            {"market_favorite_min_probability": row.get("market_favorite_min_probability")},
        )
        for row in market_favorite_player_form_policy_candidates
    )
    candidates.extend(
        _policy_tradeoff_candidate(
            "player_status_policy_candidates",
            row,
            {
                "player_status_min_confidence": row.get("player_status_min_confidence"),
                "player_status_min_margin": row.get("player_status_min_margin"),
            },
        )
        for row in player_status_policy_candidates
    )
    candidates.extend(
        _policy_tradeoff_candidate(
            "bo1_margin_policy_candidates",
            row,
            {
                "minimum_margin": row.get("minimum_margin"),
                "bo1_minimum_margin": row.get("bo1_minimum_margin"),
            },
        )
        for row in bo1_margin_policy_candidates
    )
    eligible = [row for row in candidates if int(row.get("actionable_picks", 0)) >= 2]
    if not eligible:
        eligible = candidates
    highest_accuracy = max(
        eligible,
        key=lambda row: (
            _float(row.get("actionable_accuracy"), 0.0),
            int(row.get("correct_actionable", 0)),
            int(row.get("actionable_picks", 0)),
        ),
    )
    highest_correct = max(
        eligible,
        key=lambda row: (
            int(row.get("correct_actionable", 0)),
            _float(row.get("actionable_accuracy"), 0.0),
            int(row.get("actionable_picks", 0)),
        ),
    )
    current_correct = int(current.get("correct_actionable", 0))
    current_actionable = int(current.get("actionable_picks", 0))
    highest_accuracy_correct = int(highest_accuracy.get("correct_actionable", 0))
    highest_accuracy_actionable = int(highest_accuracy.get("actionable_picks", 0))
    accuracy_gain = _float(highest_accuracy.get("actionable_accuracy"), 0.0) - _float(
        current.get("actionable_accuracy"),
        0.0,
    )
    correct_delta = highest_accuracy_correct - current_correct
    actionable_delta = highest_accuracy_actionable - current_actionable
    recommendation, recommendation_basis = _policy_tradeoff_recommendation(
        current,
        highest_accuracy,
        highest_correct,
    )
    recommended_candidate = _policy_tradeoff_recommended_candidate(
        recommendation,
        current,
        highest_accuracy,
        highest_correct,
    )
    return {
        "minimum_actionable_picks": 2,
        "current_policy": current,
        "highest_accuracy_candidate": highest_accuracy,
        "highest_correct_candidate": highest_correct,
        "accuracy_gain_over_current": accuracy_gain,
        "correct_pick_delta_vs_current": correct_delta,
        "actionable_pick_delta_vs_current": actionable_delta,
        "coverage_delta_vs_current": actionable_delta / current_actionable if current_actionable else 0.0,
        "recommendation": recommendation,
        "recommendation_basis": recommendation_basis,
        "recommended_policy_update": _recommended_policy_update(
            recommendation,
            recommendation_basis,
            recommended_candidate,
        ),
    }


def _policy_tradeoff_candidate(
    source: str,
    row: Mapping[str, Any],
    parameter: Mapping[str, Any],
) -> Dict[str, object]:
    actionable_picks = int(row.get("actionable_picks", 0))
    correct_actionable = int(row.get("correct_actionable", 0))
    return {
        "source": source,
        "parameter": dict(parameter),
        "actionable_picks": actionable_picks,
        "correct_actionable": correct_actionable,
        "missed_actionable": int(row.get("missed_actionable", max(actionable_picks - correct_actionable, 0))),
        "actionable_accuracy": _float(row.get("actionable_accuracy"), 0.0),
        "coverage": _optional_float(row.get("coverage")),
        "avoided_wins": int(row.get("avoided_wins", 0)),
        "avoided_losses": int(row.get("avoided_losses", 0)),
    }


def _policy_tradeoff_recommendation(
    current: Mapping[str, Any],
    highest_accuracy: Mapping[str, Any],
    highest_correct: Mapping[str, Any],
) -> tuple[str, str]:
    if highest_accuracy.get("source") == "current_policy":
        return "keep_current_policy", "current_policy_highest_accuracy"
    if int(highest_accuracy.get("correct_actionable", 0)) < int(current.get("correct_actionable", 0)):
        return "keep_current_policy", "accuracy_gain_reduces_total_correct"
    if _float(highest_accuracy.get("actionable_accuracy"), 0.0) > _float(
        current.get("actionable_accuracy"),
        0.0,
    ):
        return "promote_highest_accuracy_candidate", "accuracy_gain_without_losing_correct_picks"
    if int(highest_correct.get("correct_actionable", 0)) > int(current.get("correct_actionable", 0)):
        return "review_highest_correct_candidate", "higher_total_correct_available"
    return "keep_current_policy", "no_candidate_improves_current_policy"


def _policy_tradeoff_recommended_candidate(
    recommendation: str,
    current: Mapping[str, Any],
    highest_accuracy: Mapping[str, Any],
    highest_correct: Mapping[str, Any],
) -> Mapping[str, Any]:
    if recommendation == "promote_highest_accuracy_candidate":
        return highest_accuracy
    if recommendation == "review_highest_correct_candidate":
        return highest_correct
    return current


def _recommended_policy_update(
    action: str,
    basis: str,
    candidate: Mapping[str, Any],
) -> Dict[str, object]:
    apply_args, cli_flags = _policy_candidate_apply_args(candidate)
    parameter = candidate.get("parameter", {})
    return {
        "action": action,
        "basis": basis,
        "source": candidate.get("source"),
        "parameter": dict(parameter) if isinstance(parameter, Mapping) else {},
        "apply_forecast_policy_args": apply_args,
        "cli_flags": cli_flags,
        "candidate": {
            "actionable_picks": int(candidate.get("actionable_picks", 0)),
            "correct_actionable": int(candidate.get("correct_actionable", 0)),
            "actionable_accuracy": _float(candidate.get("actionable_accuracy"), 0.0),
            "coverage": _optional_float(candidate.get("coverage")),
        },
    }


def _policy_candidate_apply_args(candidate: Mapping[str, Any]) -> tuple[Dict[str, object], List[str]]:
    source = candidate.get("source")
    parameter = candidate.get("parameter", {})
    if not isinstance(parameter, Mapping):
        parameter = {}
    args: Dict[str, object] = {}
    flags: List[str] = []
    if source == "current_policy":
        return _decision_policy_apply_args(parameter)
    if source == "threshold_candidates":
        minimum_margin = _optional_float(parameter.get("minimum_margin"))
        if minimum_margin is not None:
            args["minimum_margin"] = minimum_margin
            flags.extend(["--minimum-margin", _format_cli_float(minimum_margin)])
    elif source == "bo1_margin_policy_candidates":
        minimum_margin = _optional_float(parameter.get("minimum_margin"))
        bo1_minimum_margin = _optional_float(parameter.get("bo1_minimum_margin"))
        if minimum_margin is not None:
            args["minimum_margin"] = minimum_margin
            flags.extend(["--minimum-margin", _format_cli_float(minimum_margin)])
        if bo1_minimum_margin is not None:
            args["bo1_minimum_margin"] = bo1_minimum_margin
            flags.extend(["--bo1-minimum-margin", _format_cli_float(bo1_minimum_margin)])
    elif source == "player_form_policy_candidates":
        min_confidence = _optional_float(parameter.get("player_form_counter_min_confidence"))
        args["avoid_player_form_counter_signal"] = True
        flags.append("--avoid-player-form-counter-signal")
        if min_confidence is not None:
            args["player_form_counter_min_confidence"] = min_confidence
            flags.extend(["--player-form-counter-min-confidence", _format_cli_float(min_confidence)])
    elif source == "market_favorite_player_form_policy_candidates":
        min_probability = _optional_float(parameter.get("market_favorite_min_probability"))
        args["avoid_market_favorite_player_form_counter_signal"] = True
        flags.append("--avoid-market-favorite-player-form-counter-signal")
        if min_probability is not None:
            args["market_favorite_counter_min_probability"] = min_probability
            flags.extend(["--market-favorite-counter-min-probability", _format_cli_float(min_probability)])
    elif source == "player_status_policy_candidates":
        min_confidence = _optional_float(parameter.get("player_status_min_confidence"))
        min_margin = _optional_float(parameter.get("player_status_min_margin"))
        args["avoid_player_status_risk"] = True
        flags.append("--avoid-player-status-risk")
        if min_confidence is not None:
            args["player_status_min_confidence"] = min_confidence
            flags.extend(["--player-status-min-confidence", _format_cli_float(min_confidence)])
        if min_margin is not None:
            args["player_status_min_margin"] = min_margin
            flags.extend(["--player-status-min-margin", _format_cli_float(min_margin)])
    return args, flags


def _current_policy_parameter(decision_policy: Mapping[str, Any] | None) -> Dict[str, object]:
    if not isinstance(decision_policy, Mapping):
        return {}
    parameter: Dict[str, object] = {}
    for key in (
        "minimum_margin",
        "bo1_minimum_margin",
        "player_form_counter_min_confidence",
        "market_favorite_counter_min_probability",
        "player_status_min_confidence",
        "player_status_min_margin",
    ):
        value = _optional_float(decision_policy.get(key))
        if value is not None:
            parameter[key] = value
    for key in (
        "avoid_player_form_counter_signal",
        "avoid_market_favorite_player_form_counter_signal",
        "avoid_player_status_risk",
    ):
        if key in decision_policy:
            parameter[key] = _truthy(decision_policy.get(key))
    return parameter


def _decision_policy_apply_args(parameter: Mapping[str, Any]) -> tuple[Dict[str, object], List[str]]:
    args: Dict[str, object] = {}
    flags: List[str] = []
    minimum_margin = _optional_float(parameter.get("minimum_margin"))
    bo1_minimum_margin = _optional_float(parameter.get("bo1_minimum_margin"))
    if minimum_margin is not None:
        args["minimum_margin"] = minimum_margin
        flags.extend(["--minimum-margin", _format_cli_float(minimum_margin)])
    if bo1_minimum_margin is not None:
        args["bo1_minimum_margin"] = bo1_minimum_margin
        flags.extend(["--bo1-minimum-margin", _format_cli_float(bo1_minimum_margin)])
    if _truthy(parameter.get("avoid_player_form_counter_signal")):
        args["avoid_player_form_counter_signal"] = True
        flags.append("--avoid-player-form-counter-signal")
        min_confidence = _optional_float(parameter.get("player_form_counter_min_confidence"))
        if min_confidence is not None:
            args["player_form_counter_min_confidence"] = min_confidence
            flags.extend(["--player-form-counter-min-confidence", _format_cli_float(min_confidence)])
    if _truthy(parameter.get("avoid_market_favorite_player_form_counter_signal")):
        args["avoid_market_favorite_player_form_counter_signal"] = True
        flags.append("--avoid-market-favorite-player-form-counter-signal")
        min_probability = _optional_float(parameter.get("market_favorite_counter_min_probability"))
        if min_probability is not None:
            args["market_favorite_counter_min_probability"] = min_probability
            flags.extend(["--market-favorite-counter-min-probability", _format_cli_float(min_probability)])
    if _truthy(parameter.get("avoid_player_status_risk")):
        args["avoid_player_status_risk"] = True
        flags.append("--avoid-player-status-risk")
        min_confidence = _optional_float(parameter.get("player_status_min_confidence"))
        min_margin = _optional_float(parameter.get("player_status_min_margin"))
        if min_confidence is not None:
            args["player_status_min_confidence"] = min_confidence
            flags.extend(["--player-status-min-confidence", _format_cli_float(min_confidence)])
        if min_margin is not None:
            args["player_status_min_margin"] = min_margin
            flags.extend(["--player-status-min-margin", _format_cli_float(min_margin)])
    return args, flags


def _format_cli_float(value: float) -> str:
    return f"{value:g}"


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


def _bo1_margin_policy_candidates(
    rows: Iterable[Mapping[str, Any]],
    minimum_margin: float = 0.02,
) -> List[Dict[str, object]]:
    materialized = list(rows)
    return [
        _bo1_margin_policy_candidate(materialized, minimum_margin, bo1_minimum_margin)
        for bo1_minimum_margin in BO1_MINIMUM_MARGIN_CANDIDATES
    ]


def _bo1_margin_policy_candidate(
    rows: Iterable[Mapping[str, Any]],
    minimum_margin: float,
    bo1_minimum_margin: float,
) -> Dict[str, object]:
    materialized = list(rows)
    selected_indexes = set()
    bo1_matches = 0
    bo1_avoids = 0
    for index, row in enumerate(materialized):
        is_bo1 = _best_of(row) == 1
        required_margin = bo1_minimum_margin if is_bo1 else minimum_margin
        if is_bo1:
            bo1_matches += 1
        if _float(row.get("confidence_margin"), 0.0) >= required_margin:
            selected_indexes.add(index)
        elif is_bo1:
            bo1_avoids += 1
    selected = [row for index, row in enumerate(materialized) if index in selected_indexes]
    avoided = [row for index, row in enumerate(materialized) if index not in selected_indexes]
    correct = sum(1 for row in selected if row.get("directional_correct"))
    return {
        "minimum_margin": minimum_margin,
        "bo1_minimum_margin": bo1_minimum_margin,
        "bo1_matches": bo1_matches,
        "bo1_avoids": bo1_avoids,
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


def _market_favorite_player_form_policy_candidates(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, object]]:
    materialized = list(rows)
    return [
        _market_favorite_player_form_policy_candidate(materialized, min_probability)
        for min_probability in MARKET_FAVORITE_FORM_COUNTER_PROBABILITY_CANDIDATES
    ]


def _market_favorite_player_form_policy_candidate(
    rows: Iterable[Mapping[str, Any]],
    min_probability: float,
) -> Dict[str, object]:
    materialized = list(rows)
    avoided_indexes = set()
    counter_signal_matches = 0
    current_actionable = [row for row in materialized if row.get("actionable")]
    for index, row in enumerate(materialized):
        if not row.get("actionable"):
            continue
        if _float(row.get("market_favorite_probability"), 0.0) < min_probability:
            continue
        if _team_key(row.get("directional_pick")) != _team_key(row.get("market_favorite")):
            continue
        if row.get("player_form_directional_score") is None:
            continue
        if _float(row.get("player_form_directional_score"), 0.0) < 0:
            counter_signal_matches += 1
            avoided_indexes.add(index)
    selected = [
        row
        for index, row in enumerate(materialized)
        if row.get("actionable") and index not in avoided_indexes
    ]
    avoided = [row for index, row in enumerate(materialized) if index in avoided_indexes]
    correct = sum(1 for row in selected if row.get("directional_correct"))
    return {
        "market_favorite_min_probability": min_probability,
        "counter_signal_matches": counter_signal_matches,
        "actionable_picks": len(selected),
        "correct_actionable": correct,
        "missed_actionable": len(selected) - correct,
        "actionable_accuracy": correct / len(selected) if selected else 0.0,
        "coverage": len(selected) / len(current_actionable) if current_actionable else 0.0,
        "avoided_wins": sum(1 for row in avoided if row.get("directional_correct")),
        "avoided_losses": sum(1 for row in avoided if not row.get("directional_correct")),
    }


def _player_status_policy_candidates(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, object]]:
    materialized = list(rows)
    return [
        _player_status_policy_candidate(materialized, min_confidence, min_margin)
        for min_confidence in PLAYER_STATUS_CONFIDENCE_CANDIDATES
        for min_margin in PLAYER_STATUS_MARGIN_CANDIDATES
    ]


def _player_status_policy_candidate(
    rows: Iterable[Mapping[str, Any]],
    min_confidence: float,
    min_margin: float,
) -> Dict[str, object]:
    materialized = list(rows)
    current_actionable = [row for row in materialized if row.get("actionable")]
    avoided_indexes = set()
    status_risk_matches = 0
    substitute_risk_matches = 0
    low_sample_risk_matches = 0
    for index, row in enumerate(materialized):
        if not row.get("actionable"):
            continue
        if _float(row.get("confidence_margin"), 0.0) > min_margin:
            continue
        low_sample = _player_status_low_sample(row, min_confidence)
        substitute = _player_status_substitute(row)
        if low_sample or substitute:
            status_risk_matches += 1
            if substitute:
                substitute_risk_matches += 1
            if low_sample:
                low_sample_risk_matches += 1
            avoided_indexes.add(index)
    selected = [
        row
        for index, row in enumerate(materialized)
        if row.get("actionable") and index not in avoided_indexes
    ]
    avoided = [row for index, row in enumerate(materialized) if index in avoided_indexes]
    correct = sum(1 for row in selected if row.get("directional_correct"))
    return {
        "player_status_min_confidence": min_confidence,
        "player_status_min_margin": min_margin,
        "status_risk_matches": status_risk_matches,
        "substitute_risk_matches": substitute_risk_matches,
        "low_sample_risk_matches": low_sample_risk_matches,
        "actionable_picks": len(selected),
        "correct_actionable": correct,
        "missed_actionable": len(selected) - correct,
        "actionable_accuracy": correct / len(selected) if selected else 0.0,
        "coverage": len(selected) / len(current_actionable) if current_actionable else 0.0,
        "avoided_wins": sum(1 for row in avoided if row.get("directional_correct")),
        "avoided_losses": sum(1 for row in avoided if not row.get("directional_correct")),
    }


def _player_status_signal_risk(
    rows: Iterable[Mapping[str, Any]],
    min_confidence: float = 0.4,
) -> Dict[str, object]:
    materialized = list(rows)
    available = [row for row in materialized if _has_picked_player_status(row)]
    missing = [row for row in materialized if not _has_picked_player_status(row)]
    actionable = [row for row in available if row.get("actionable")]
    avoid = [row for row in available if not row.get("actionable")]
    status_risk_actionable = [row for row in actionable if _player_status_risk_row(row, min_confidence)]
    non_status_risk_actionable = [row for row in actionable if not _player_status_risk_row(row, min_confidence)]
    status_risk_avoid = [row for row in avoid if _player_status_risk_row(row, min_confidence)]
    status_risk_correct = sum(1 for row in status_risk_actionable if row.get("correct"))
    non_status_risk_correct = sum(1 for row in non_status_risk_actionable if row.get("correct"))
    return {
        "sample_confidence_threshold": min_confidence,
        "available_matches": len(available),
        "missing_status_matches": len(missing),
        "available_actionable_matches": len(actionable),
        "missing_status_actionable_matches": sum(1 for row in missing if row.get("actionable")),
        "status_risk_actionable_matches": len(status_risk_actionable),
        "low_sample_risk_actionable_matches": sum(
            1 for row in status_risk_actionable if _player_status_low_sample(row, min_confidence)
        ),
        "substitute_risk_actionable_matches": sum(
            1 for row in status_risk_actionable if _player_status_substitute(row)
        ),
        "status_risk_correct_actionable": status_risk_correct,
        "status_risk_missed_actionable": len(status_risk_actionable) - status_risk_correct,
        "status_risk_actionable_accuracy": (
            status_risk_correct / len(status_risk_actionable)
            if status_risk_actionable
            else None
        ),
        "non_status_risk_actionable_matches": len(non_status_risk_actionable),
        "non_status_risk_correct_actionable": non_status_risk_correct,
        "non_status_risk_missed_actionable": len(non_status_risk_actionable) - non_status_risk_correct,
        "non_status_risk_actionable_accuracy": (
            non_status_risk_correct / len(non_status_risk_actionable)
            if non_status_risk_actionable
            else None
        ),
        "status_risk_avoid_picks": len(status_risk_avoid),
        "status_risk_avoided_wins": sum(1 for row in status_risk_avoid if row.get("directional_correct")),
        "status_risk_avoided_losses": sum(1 for row in status_risk_avoid if not row.get("directional_correct")),
    }


def _has_picked_player_status(row: Mapping[str, Any]) -> bool:
    return row.get("picked_player_sample_confidence") is not None or row.get("picked_substitute_flag") is not None


def _player_status_risk_row(row: Mapping[str, Any], min_confidence: float) -> bool:
    return _player_status_low_sample(row, min_confidence) or _player_status_substitute(row)


def _player_status_low_sample(row: Mapping[str, Any], min_confidence: float) -> bool:
    sample_confidence = row.get("picked_player_sample_confidence")
    if sample_confidence is None:
        return False
    return _float(sample_confidence, 1.0) < min_confidence


def _player_status_substitute(row: Mapping[str, Any]) -> bool:
    return _float(row.get("picked_substitute_flag"), 0.0) >= 1.0


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


def _avg_numeric(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [
        _float(row.get(key), 0.0)
        for row in rows
        if row.get(key) not in (None, "")
    ]
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


def _picked_player_status(
    prediction: Mapping[str, Any],
    directional_pick: str,
    team1: str,
    team2: str,
) -> Dict[str, object]:
    summary = prediction.get("player_form_summary")
    if not isinstance(summary, Mapping):
        return {"sample_confidence": None, "substitute_flag": None}
    side = summary.get("team1")
    if isinstance(side, Mapping) and _player_status_side_matches(side, directional_pick, team1):
        return {
            "sample_confidence": _float(side.get("sample_confidence"), 0.0),
            "substitute_flag": _int(side.get("substitute_flag")),
        }
    side = summary.get("team2")
    if isinstance(side, Mapping) and _player_status_side_matches(side, directional_pick, team2):
        return {
            "sample_confidence": _float(side.get("sample_confidence"), 0.0),
            "substitute_flag": _int(side.get("substitute_flag")),
        }
    return {"sample_confidence": None, "substitute_flag": None}


def _player_status_side_matches(side: Mapping[str, Any], directional_pick: str, fallback_team: str) -> bool:
    side_team = side.get("team")
    expected_team = side_team if side_team not in (None, "") else fallback_team
    return _team_key(expected_team) == _team_key(directional_pick)


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


def _market_probability_team1(prediction: Mapping[str, Any]) -> float | None:
    market_signal = prediction.get("market_signal")
    if isinstance(market_signal, Mapping):
        probability = _optional_probability(market_signal.get("probability_team1"))
        if probability is not None:
            return probability
    return _optional_probability(prediction.get("market_probability_team1"))


def _favorite_from_probability(
    probability_team1: float | None,
    team1: str,
    team2: str,
) -> Dict[str, object] | None:
    if probability_team1 is None:
        return None
    if probability_team1 >= 0.5:
        return {"team": team1, "probability": probability_team1}
    return {"team": team2, "probability": 1.0 - probability_team1}


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


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _best_of(row: Mapping[str, Any]) -> int:
    parsed = _int(row.get("best_of") or 1)
    return parsed if parsed > 0 else 1


def _optional_probability(value: Any) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= probability <= 1.0:
        return probability
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
