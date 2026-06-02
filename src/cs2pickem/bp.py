from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .data import read_matches_csv, write_matches_csv


TEAM_SCOPED_FIELDS = {
    "team1_bans": "team2_bans",
    "team2_bans": "team1_bans",
    "team1_pick": "team2_pick",
    "team2_pick": "team1_pick",
    "team1_picks": "team2_picks",
    "team2_picks": "team1_picks",
}


def merge_bp_into_fixtures(
    fixture_rows: Iterable[Mapping[str, Any]],
    bp_rows: Iterable[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    bp_by_key = {_fixture_key(row): dict(row) for row in bp_rows}
    merged = []
    matched = 0
    map_overrides = 0
    for fixture in fixture_rows:
        copied = dict(fixture)
        bp_row = bp_by_key.get(_fixture_key(fixture))
        if not bp_row:
            copied.setdefault("bp_applied", 0)
            merged.append(copied)
            continue
        matched += 1
        reversed_order = _normalize_team(copied.get("team1")) == _normalize_team(bp_row.get("team2"))
        map_name = _normalize_map(bp_row.get("confirmed_map") or bp_row.get("map") or bp_row.get("expected_map"))
        if map_name and map_name not in {"unknown", "tbd", "none"}:
            copied["map"] = map_name
            map_overrides += 1
        copied["bp_applied"] = 1
        copied["bp_source"] = bp_row.get("source", bp_row.get("provider", "unknown"))
        if bp_row.get("confidence") not in (None, ""):
            copied["bp_confidence"] = _num(bp_row.get("confidence"), 0.0)
        _copy_team_scoped_fields(copied, bp_row, reversed_order)
        merged.append(copied)
    report = {
        "fixtures": len(merged),
        "matched": matched,
        "unmatched": len(merged) - matched,
        "map_overrides": map_overrides,
    }
    return merged, report


def merge_bp_file(fixtures_path: str, bp_path: str, output_path: str) -> Dict[str, int]:
    merged, report = merge_bp_into_fixtures(read_matches_csv(fixtures_path), read_matches_csv(bp_path))
    write_matches_csv(output_path, merged)
    return report


def _copy_team_scoped_fields(target: Dict[str, Any], source: Mapping[str, Any], reversed_order: bool) -> None:
    for field_name in TEAM_SCOPED_FIELDS:
        if source.get(field_name) in (None, ""):
            continue
        target_name = TEAM_SCOPED_FIELDS[field_name] if reversed_order else field_name
        target[target_name] = source[field_name]


def _fixture_key(row: Mapping[str, Any]) -> str:
    date = str(row.get("date", "")).strip()
    teams = sorted([_normalize_team(row.get("team1")), _normalize_team(row.get("team2"))])
    return f"{date}__{teams[0]}__{teams[1]}"


def _normalize_team(value: Any) -> str:
    return str(value).strip().casefold()


def _normalize_map(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip().lower().replace("de_", "")


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
