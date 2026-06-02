from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping

from .data import read_matches_csv, read_teams_csv, write_json
from .pickem import model_driven_pickems


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

    for category in ("3-0", "advance", "0-3"):
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
    pickems = payload.get("pickems", payload)
    report = evaluate_pickem_result(pickems, read_matches_csv(results_path), pass_threshold=pass_threshold)
    report["pickems_path"] = pickems_path
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
        output["pickems"] = payload.get("pickems", payload)
        output["pickems_path"] = pickems_path
    else:
        output["pickems"] = case.get("pickems", {})
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


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
