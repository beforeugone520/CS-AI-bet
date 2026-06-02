from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping

from .data import read_matches_csv, write_matches_csv


def append_matches_dataset(
    dataset_path: str,
    incoming_rows: Iterable[Dict[str, Any]],
    manifest_path: str | None = None,
    source_name: str | None = None,
) -> Dict[str, object]:
    existing = read_matches_csv(dataset_path) if os.path.exists(dataset_path) else []
    incoming = [dict(row) for row in incoming_rows]
    by_identity = {match_identity(row): dict(row) for row in existing}
    added = 0
    for row in incoming:
        identity = match_identity(row)
        if identity not in by_identity:
            added += 1
        by_identity[identity] = row
    merged = sorted(by_identity.values(), key=lambda row: (str(row.get("date", "")), str(row.get("team1", "")), str(row.get("team2", ""))))
    write_matches_csv(dataset_path, merged)
    manifest = dataset_manifest(merged, source_name=source_name)
    if manifest_path:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return {
        "existing_rows": len(existing),
        "incoming_rows": len(incoming),
        "added_rows": added,
        "total_rows": len(merged),
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
    }


def dataset_manifest(rows: Iterable[Dict[str, Any]], source_name: str | None = None) -> Dict[str, object]:
    materialized = [dict(row) for row in rows]
    dates = sorted(str(row.get("date", ""))[:10] for row in materialized if row.get("date"))
    teams = {str(row.get("team1")) for row in materialized if row.get("team1")} | {str(row.get("team2")) for row in materialized if row.get("team2")}
    sources = sorted({str(row.get("source")) for row in materialized if row.get("source")})
    if source_name and source_name not in sources:
        sources.append(source_name)
        sources.sort()
    return {
        "rows": len(materialized),
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "teams": len(teams),
        "sources": sources,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def dataset_coverage_report(
    rows: Iterable[Mapping[str, Any]],
    minimum_rows: int = 8000,
    required_teams: int = 80,
    participant_teams: Iterable[str] | None = None,
    top_teams: Iterable[str] | None = None,
) -> Dict[str, object]:
    materialized = [dict(row) for row in rows]
    teams = _covered_teams(materialized)
    report: Dict[str, object] = {
        "rows": len(materialized),
        "minimum_rows": minimum_rows,
        "rows_remaining": max(0, minimum_rows - len(materialized)),
        "teams": len(teams),
        "required_teams": required_teams,
        "teams_remaining": max(0, required_teams - len(teams)),
    }
    if participant_teams is not None:
        report["participant_coverage"] = _team_coverage(teams, participant_teams)
    if top_teams is not None:
        report["top_team_coverage"] = _team_coverage(teams, top_teams)
    return report


def match_identity(row: Dict[str, Any]) -> str:
    source_url = str(row.get("source_match_url") or "").strip().lower()
    if source_url:
        return f"url::{source_url}"
    date = str(row.get("date", ""))[:10]
    teams = sorted([_key(row.get("team1")), _key(row.get("team2"))])
    map_name = _key(row.get("map"))
    best_of = str(row.get("best_of", ""))
    return f"match::{date}::{teams[0]}::{teams[1]}::{map_name}::{best_of}"


def _key(value: Any) -> str:
    return str(value or "").strip().lower()


def _covered_teams(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    teams = set()
    for row in rows:
        for key in ("team1", "team2"):
            value = _team_name(row.get(key))
            if value:
                teams.add(value)
    return teams


def _team_coverage(covered_teams: set[str], required_teams: Iterable[str]) -> Dict[str, object]:
    required = sorted({_team_name(team) for team in required_teams if _team_name(team)})
    covered_lookup = {_team_name(team) for team in covered_teams}
    missing = [team for team in required if _team_name(team) not in covered_lookup]
    return {
        "covered": len(required) - len(missing),
        "required": len(required),
        "missing": missing,
    }


def _team_name(value: Any) -> str:
    return str(value or "").strip()
