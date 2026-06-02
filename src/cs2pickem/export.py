from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from .data import write_json
from .readiness import DEFAULT_PICKEM_SLOTS


CATEGORY_ORDER = ("3-0", "advance", "0-3")
OUTCOME_RECORDS = ("3-0", "3-1", "3-2", "0-3", "1-3", "2-3")


def build_pickem_answer_sheet(
    pickem_report: Mapping[str, Any],
    readiness_report: Optional[Mapping[str, Any]] = None,
    minimum_selection_margin: float = 0.04,
    required_slots: Optional[Mapping[str, int]] = DEFAULT_PICKEM_SLOTS,
) -> Dict[str, object]:
    pickems = _normalized_pickems(pickem_report.get("pickems", {}))
    picks = _flatten_picks(pickem_report.get("pickem_details", {}), pickems)
    readiness = dict(readiness_report or {})
    market_summary = dict(pickem_report.get("market_adjustment_summary", {}) or {})

    margins = [_num_or_none(pick.get("selection_margin")) for pick in picks]
    known_margins = [margin for margin in margins if margin is not None]
    minimum_margin = min(known_margins) if known_margins else None
    warnings = _warnings(
        pickems=pickems,
        picks=picks,
        readiness=readiness,
        market_summary=market_summary,
        minimum_selection_margin=minimum_selection_margin,
        required_slots=required_slots,
    )

    return {
        "pickems": pickems,
        "picks": picks,
        "team_outcomes": _team_outcomes(pickem_report.get("team_probabilities", {})),
        "ready": readiness.get("ready") if readiness_report is not None else None,
        "failed_checks": list(readiness.get("failed_checks", [])) if isinstance(readiness.get("failed_checks", []), list) else [],
        "stage": _stage(pickem_report),
        "simulations": pickem_report.get("simulations"),
        "confidence": {
            "minimum_selection_margin": minimum_margin,
            "market_adjusted_matchups": int(_num_or_none(market_summary.get("adjusted_matchups")) or 0),
            "cached_matchups": int(_num_or_none(market_summary.get("cached_matchups")) or 0),
        },
        "warnings": warnings,
    }


def build_pickem_answer_sheet_file(
    pickem_report_path: str,
    readiness_report_path: Optional[str] = None,
    output_path: Optional[str] = None,
    minimum_selection_margin: float = 0.04,
) -> Dict[str, object]:
    with open(pickem_report_path, encoding="utf-8") as handle:
        pickem_report = json.load(handle)
    readiness_report = None
    if readiness_report_path:
        with open(readiness_report_path, encoding="utf-8") as handle:
            readiness_report = json.load(handle)
    sheet = build_pickem_answer_sheet(
        pickem_report,
        readiness_report=readiness_report,
        minimum_selection_margin=minimum_selection_margin,
    )
    if output_path:
        write_json(output_path, sheet)
    return sheet


def _normalized_pickems(raw_pickems: Any) -> Dict[str, List[str]]:
    pickems: Dict[str, List[str]] = {}
    source = raw_pickems if isinstance(raw_pickems, Mapping) else {}
    for category in CATEGORY_ORDER:
        values = source.get(category, [])
        pickems[category] = [str(value) for value in values] if isinstance(values, list) else []
    return pickems


def _flatten_picks(raw_details: Any, pickems: Mapping[str, List[str]]) -> List[Dict[str, object]]:
    details = raw_details if isinstance(raw_details, Mapping) else {}
    flattened: List[Dict[str, object]] = []
    for category in CATEGORY_ORDER:
        category_details = details.get(category, [])
        if isinstance(category_details, list) and category_details:
            flattened.extend(_pick_from_detail(category, detail) for detail in category_details if isinstance(detail, Mapping))
            continue
        flattened.extend({"team": team, "category": category} for team in pickems.get(category, []))
    return flattened


def _pick_from_detail(category: str, detail: Mapping[str, Any]) -> Dict[str, object]:
    return {
        "category": str(detail.get("category", category)),
        "team": str(detail.get("team", "")),
        "probability": _num_or_none(detail.get("probability")),
        "rank": detail.get("rank"),
        "selection_score": _num_or_none(detail.get("selection_score")),
        "next_best_score": _num_or_none(detail.get("next_best_score")),
        "selection_margin": _num_or_none(detail.get("selection_margin")),
    }


def _team_outcomes(raw_probabilities: Any) -> List[Dict[str, object]]:
    probabilities = raw_probabilities if isinstance(raw_probabilities, Mapping) else {}
    outcomes: List[Dict[str, object]] = []
    for team, values in probabilities.items():
        if not isinstance(values, Mapping):
            continue
        record_probabilities = {record: _num_or_none(values.get(record)) for record in OUTCOME_RECORDS}
        known_records = {record: value for record, value in record_probabilities.items() if value is not None}
        most_likely_record = max(known_records.items(), key=lambda item: (item[1], item[0]))[0] if known_records else None
        outcomes.append(
            {
                "team": str(team),
                **record_probabilities,
                "advance": _num_or_none(values.get("advance")),
                "eliminate": _num_or_none(values.get("eliminate")),
                "most_likely_record": most_likely_record,
            }
        )
    return sorted(
        outcomes,
        key=lambda row: (
            -(_num_or_none(row.get("advance")) or 0.0),
            _num_or_none(row.get("eliminate")) or 0.0,
            str(row.get("team")),
        ),
    )


def _warnings(
    pickems: Mapping[str, List[str]],
    picks: List[Mapping[str, object]],
    readiness: Mapping[str, Any],
    market_summary: Mapping[str, Any],
    minimum_selection_margin: float,
    required_slots: Optional[Mapping[str, int]],
) -> List[Dict[str, object]]:
    warnings: List[Dict[str, object]] = []
    if readiness.get("ready") is False:
        warnings.append(
            {
                "code": "readiness_not_ready",
                "message": "Readiness gates are not all passing; treat this answer sheet as draft only.",
                "failed_checks": list(readiness.get("failed_checks", [])) if isinstance(readiness.get("failed_checks", []), list) else [],
            }
        )
    if required_slots:
        slot_gaps = {
            category: {"actual": len(pickems.get(category, [])), "required": int(required)}
            for category, required in required_slots.items()
            if len(pickems.get(category, [])) != int(required)
        }
        if slot_gaps:
            warnings.append(
                {
                    "code": "missing_pickem_slots",
                    "message": "Pick'em answer slots do not match the required submission shape.",
                    "slots": slot_gaps,
                }
            )
    low_margin = [
        {"team": pick.get("team"), "category": pick.get("category"), "selection_margin": margin}
        for pick in picks
        for margin in [_num_or_none(pick.get("selection_margin"))]
        if margin is not None and margin < minimum_selection_margin
    ]
    if low_margin:
        warnings.append(
            {
                "code": "low_selection_margin",
                "message": "At least one selected Pick'em has a narrow edge over the next candidate.",
                "minimum": minimum_selection_margin,
                "picks": low_margin,
            }
        )
    cached_matchups = int(_num_or_none(market_summary.get("cached_matchups")) or 0)
    adjusted_matchups = int(_num_or_none(market_summary.get("adjusted_matchups")) or 0)
    if cached_matchups > 0 and adjusted_matchups == 0:
        warnings.append(
            {
                "code": "no_market_adjusted_matchups",
                "message": "Fixture odds were available but no Swiss matchup probability used a market adjustment.",
                "cached_matchups": cached_matchups,
            }
        )
    return warnings


def _stage(report: Mapping[str, Any]) -> object:
    strategy = report.get("stage_strategy")
    if isinstance(strategy, Mapping):
        return strategy.get("stage")
    return None


def _num_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
