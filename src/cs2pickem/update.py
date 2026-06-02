from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from .dataset_store import append_matches_dataset, dataset_coverage_report
from .data import read_matches_csv, write_matches_csv
from .players import merge_player_stats_into_matches
from .sources import HltvEventParser, HltvPlayerStatsParser, HltvRankingParser, HltvResultParser, HttpCache, annotate_version_tags, parse_version_log


TEAM_METADATA_FIELDS = {
    "rank": "rank",
    "world_rank": "rank",
    "rmr_points": "rmr_points",
    "points": "rmr_points",
    "major_best_placement": "major_best_placement",
    "rating": "rating",
    "kd": "kd",
    "recent_winrate_10": "recent_winrate_10",
    "bo1_winrate_6m": "bo1_winrate_6m",
    "bo3_winrate_6m": "bo3_winrate_6m",
    "map_winrate": "map_winrate",
}

SWISS_DEFAULTS = {
    "swiss_round": 1,
    "team1_wins": 0,
    "team1_losses": 0,
    "team2_wins": 0,
    "team2_losses": 0,
}


def daily_update_from_config(
    config_path: str,
    output_dir: str | None = None,
    refresh: bool = False,
    fetcher: Callable[[str, Dict[str, str]], str] | None = None,
) -> Dict[str, object]:
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    output_root = os.path.abspath(output_dir or _resolve_path(config.get("output_dir", "daily-update"), base_dir))
    os.makedirs(output_root, exist_ok=True)

    dataset_path = _resolve_optional_path(config.get("dataset"), base_dir)
    dataset_manifest_path = _resolve_optional_path(config.get("dataset_manifest"), base_dir)
    participants_path = _resolve_optional_path(config.get("participants"), base_dir)
    top_teams_path = _resolve_optional_path(config.get("top_teams") or config.get("top-teams"), base_dir)
    minimum_rows = int(config.get("minimum_rows", 8000))
    required_teams = int(config.get("required_teams", 80))
    version_log_path = _resolve_optional_path(config.get("version_log"), base_dir)
    team_metadata_path = _resolve_optional_path(config.get("team_metadata") or config.get("teams"), base_dir)
    player_stats_path = _resolve_optional_path(config.get("player_stats") or config.get("players"), base_dir)
    player_window_days = int(config.get("player_window_days", 15))
    default_swiss_state = _as_bool(config.get("default_swiss_state", False))
    cache_dir = _resolve_path(config.get("cache_dir", os.path.join(output_root, "cache")), base_dir)
    job_reports = []
    for index, job in enumerate(config.get("jobs", []), start=1):
        report = _run_daily_job(
            job=dict(job),
            index=index,
            output_root=output_root,
            base_dir=base_dir,
            cache_dir=cache_dir,
            version_log_path=version_log_path,
            team_metadata_path=team_metadata_path,
            player_stats_path=player_stats_path,
            player_window_days=player_window_days,
            default_swiss_state=default_swiss_state,
            dataset_path=dataset_path,
            dataset_manifest_path=dataset_manifest_path,
            refresh=refresh,
            fetcher=fetcher,
        )
        job_reports.append(report)

    manifest = {
        "config_path": os.path.abspath(config_path),
        "output_dir": output_root,
        "jobs": len(job_reports),
        "total_rows": sum(int(report.get("rows", 0)) for report in job_reports),
        "total_added_rows": sum(int(dict(report.get("dataset") or {}).get("added_rows", 0)) for report in job_reports),
        "dataset_path": dataset_path,
        "dataset_manifest_path": dataset_manifest_path,
        "job_reports": job_reports,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if dataset_path and os.path.exists(dataset_path):
        manifest["coverage"] = dataset_coverage_report(
            read_matches_csv(dataset_path),
            minimum_rows=minimum_rows,
            required_teams=required_teams,
            participant_teams=_read_team_names(participants_path) if participants_path else None,
            top_teams=_read_team_names(top_teams_path) if top_teams_path else None,
        )
    manifest_path = os.path.join(output_root, "daily_update_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    manifest["manifest_path"] = manifest_path
    return manifest


def update_dataset_from_html(
    html_path: str,
    output_path: str,
    manifest_path: str,
    version_log_path: str | None = None,
    source_name: str = "hltv",
    dataset_path: str | None = None,
    dataset_manifest_path: str | None = None,
    team_metadata_path: str | None = None,
    player_stats_path: str | None = None,
    player_window_days: int = 15,
    default_swiss_state: bool = False,
) -> Dict[str, object]:
    with open(html_path, encoding="utf-8") as handle:
        html = handle.read()
    rows: List[dict] = HltvResultParser().parse_results_html(html)
    return _write_parsed_dataset(
        rows,
        output_path,
        manifest_path,
        version_log_path,
        source_name,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        team_metadata_path=team_metadata_path,
        player_stats_path=player_stats_path,
        player_window_days=player_window_days,
        default_swiss_state=default_swiss_state,
    )


def update_dataset_from_url(
    url: str,
    cache_dir: str,
    output_path: str,
    manifest_path: str,
    version_log_path: str | None = None,
    source_name: str = "hltv",
    refresh: bool = False,
    fetcher: Callable[[str, Dict[str, str]], str] | None = None,
    dataset_path: str | None = None,
    dataset_manifest_path: str | None = None,
    team_metadata_path: str | None = None,
    player_stats_path: str | None = None,
    player_window_days: int = 15,
    default_swiss_state: bool = False,
) -> Dict[str, object]:
    cache = HttpCache(cache_dir, fetcher=fetcher)
    html = cache.get(url, refresh=refresh)
    return _write_parsed_dataset(
        rows=HltvResultParser().parse_results_html(html),
        output_path=output_path,
        manifest_path=manifest_path,
        version_log_path=version_log_path,
        source_name=source_name,
        source_url=url,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        team_metadata_path=team_metadata_path,
        player_stats_path=player_stats_path,
        player_window_days=player_window_days,
        default_swiss_state=default_swiss_state,
    )


def update_event_teams_from_html(
    html_path: str,
    output_path: str,
    manifest_path: str,
    source_name: str = "hltv-event",
) -> Dict[str, object]:
    with open(html_path, encoding="utf-8") as handle:
        rows = HltvEventParser().parse_teams_html(handle.read())
    return _write_event_teams(rows, output_path, manifest_path, source_name)


def update_event_teams_from_url(
    url: str,
    cache_dir: str,
    output_path: str,
    manifest_path: str,
    source_name: str = "hltv-event",
    refresh: bool = False,
    fetcher: Callable[[str, Dict[str, str]], str] | None = None,
) -> Dict[str, object]:
    html = HttpCache(cache_dir, fetcher=fetcher).get(url, refresh=refresh)
    return _write_event_teams(
        rows=HltvEventParser().parse_teams_html(html),
        output_path=output_path,
        manifest_path=manifest_path,
        source_name=source_name,
        source_url=url,
    )


def update_rankings_from_html(
    html_path: str,
    output_path: str,
    manifest_path: str,
    limit: int = 80,
    source_name: str = "hltv-rankings",
) -> Dict[str, object]:
    with open(html_path, encoding="utf-8") as handle:
        rows = HltvRankingParser().parse_rankings_html(handle.read(), limit=limit)
    return _write_team_rows(rows, output_path, manifest_path, source_name, limit=limit)


def update_rankings_from_url(
    url: str,
    cache_dir: str,
    output_path: str,
    manifest_path: str,
    limit: int = 80,
    source_name: str = "hltv-rankings",
    refresh: bool = False,
    fetcher: Callable[[str, Dict[str, str]], str] | None = None,
) -> Dict[str, object]:
    html = HttpCache(cache_dir, fetcher=fetcher).get(url, refresh=refresh)
    return _write_team_rows(
        rows=HltvRankingParser().parse_rankings_html(html, limit=limit),
        output_path=output_path,
        manifest_path=manifest_path,
        source_name=source_name,
        source_url=url,
        limit=limit,
    )


def update_player_stats_from_html(
    html_path: str,
    output_path: str,
    manifest_path: str,
    default_date: str,
    source_name: str = "hltv-player-stats",
) -> Dict[str, object]:
    with open(html_path, encoding="utf-8") as handle:
        rows = HltvPlayerStatsParser().parse_player_stats_html(handle.read(), default_date=default_date)
    return _write_player_rows(rows, output_path, manifest_path, source_name, default_date=default_date)


def update_player_stats_from_url(
    url: str,
    cache_dir: str,
    output_path: str,
    manifest_path: str,
    default_date: str,
    source_name: str = "hltv-player-stats",
    refresh: bool = False,
    fetcher: Callable[[str, Dict[str, str]], str] | None = None,
) -> Dict[str, object]:
    html = HttpCache(cache_dir, fetcher=fetcher).get(url, refresh=refresh)
    return _write_player_rows(
        rows=HltvPlayerStatsParser().parse_player_stats_html(html, default_date=default_date),
        output_path=output_path,
        manifest_path=manifest_path,
        source_name=source_name,
        source_url=url,
        default_date=default_date,
    )


def _run_daily_job(
    job: Dict[str, Any],
    index: int,
    output_root: str,
    base_dir: str,
    cache_dir: str,
    version_log_path: str | None,
    team_metadata_path: str | None,
    player_stats_path: str | None,
    player_window_days: int,
    default_swiss_state: bool,
    dataset_path: str | None,
    dataset_manifest_path: str | None,
    refresh: bool,
    fetcher: Callable[[str, Dict[str, str]], str] | None,
) -> Dict[str, object]:
    kind = str(job.get("kind", "results")).strip().lower()
    if kind != "results":
        raise ValueError(f"unsupported daily update job kind: {kind}")
    name = _safe_name(str(job.get("name") or job.get("source_name") or f"job-{index}"))
    output_path = os.path.join(output_root, f"{index:02d}-{name}-matches.json")
    manifest_path = os.path.join(output_root, f"{index:02d}-{name}-manifest.json")
    source_name = str(job.get("source_name") or name)
    job_version_log = _resolve_optional_path(job.get("version_log"), base_dir) or version_log_path
    job_team_metadata = _resolve_optional_path(job.get("team_metadata") or job.get("teams"), base_dir) or team_metadata_path
    job_player_stats = _resolve_optional_path(job.get("player_stats") or job.get("players"), base_dir) or player_stats_path
    job_player_window_days = int(job.get("player_window_days", player_window_days))
    job_default_swiss_state = _as_bool(job.get("default_swiss_state", default_swiss_state))
    if job.get("html"):
        report = update_dataset_from_html(
            html_path=_resolve_path(job["html"], base_dir),
            output_path=output_path,
            manifest_path=manifest_path,
            version_log_path=job_version_log,
            source_name=source_name,
            dataset_path=dataset_path,
            dataset_manifest_path=dataset_manifest_path,
            team_metadata_path=job_team_metadata,
            player_stats_path=job_player_stats,
            player_window_days=job_player_window_days,
            default_swiss_state=job_default_swiss_state,
        )
    elif job.get("url"):
        report = update_dataset_from_url(
            url=str(job["url"]),
            cache_dir=_resolve_path(job.get("cache_dir", cache_dir), base_dir),
            output_path=output_path,
            manifest_path=manifest_path,
            version_log_path=job_version_log,
            source_name=source_name,
            refresh=_as_bool(job.get("refresh", refresh)),
            fetcher=fetcher,
            dataset_path=dataset_path,
            dataset_manifest_path=dataset_manifest_path,
            team_metadata_path=job_team_metadata,
            player_stats_path=job_player_stats,
            player_window_days=job_player_window_days,
            default_swiss_state=job_default_swiss_state,
        )
    else:
        raise ValueError(f"daily update job {name} requires html or url")
    return {**report, "name": name, "kind": kind, "output_path": output_path, "manifest_path": manifest_path}


def _write_parsed_dataset(
    rows: List[dict],
    output_path: str,
    manifest_path: str,
    version_log_path: str | None,
    source_name: str,
    source_url: str | None = None,
    dataset_path: str | None = None,
    dataset_manifest_path: str | None = None,
    team_metadata_path: str | None = None,
    player_stats_path: str | None = None,
    player_window_days: int = 15,
    default_swiss_state: bool = False,
) -> Dict[str, object]:
    if version_log_path:
        with open(version_log_path, encoding="utf-8") as handle:
            rows = annotate_version_tags(rows, parse_version_log(handle.read()))
    rows, augmentation = _augment_parsed_rows(
        rows,
        team_metadata_path=team_metadata_path,
        player_stats_path=player_stats_path,
        player_window_days=player_window_days,
        default_swiss_state=default_swiss_state,
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2, sort_keys=True)

    manifest: Dict[str, object] = {
        "source": source_name,
        "rows": len(rows),
        "output_path": output_path,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_url:
        manifest["source_url"] = source_url
    if augmentation:
        manifest["augmentation"] = augmentation
    if dataset_path:
        manifest["dataset"] = append_matches_dataset(dataset_path, rows, manifest_path=dataset_manifest_path, source_name=source_name)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest


def _augment_parsed_rows(
    rows: List[dict],
    team_metadata_path: str | None,
    player_stats_path: str | None,
    player_window_days: int,
    default_swiss_state: bool,
) -> tuple[List[dict], Dict[str, object]]:
    augmented = [dict(row) for row in rows]
    report: Dict[str, object] = {}
    if team_metadata_path:
        team_rows = read_matches_csv(team_metadata_path)
        augmented = _merge_team_metadata(augmented, team_rows)
        report["team_metadata"] = {
            "applied": True,
            "path": team_metadata_path,
            "teams": len(team_rows),
        }
    if player_stats_path:
        player_rows = read_matches_csv(player_stats_path)
        augmented = merge_player_stats_into_matches(augmented, player_rows, window_days=player_window_days)
        report["player_stats"] = {
            "applied": True,
            "path": player_stats_path,
            "rows": len(player_rows),
            "window_days": player_window_days,
        }
    if default_swiss_state:
        augmented = _fill_swiss_defaults(augmented)
        report["default_swiss_state"] = {"applied": True}
    return augmented, report


def _merge_team_metadata(rows: List[dict], team_rows: List[dict]) -> List[dict]:
    lookup = {_team_key(row.get("team") or row.get("name") or row.get("team_name")): row for row in team_rows}
    output = []
    for row in rows:
        copied = dict(row)
        for prefix in ("team1", "team2"):
            metadata = lookup.get(_team_key(copied.get(prefix)))
            if not metadata:
                continue
            for source_field, target_suffix in TEAM_METADATA_FIELDS.items():
                _set_if_missing(copied, f"{prefix}_{target_suffix}", metadata.get(source_field))
        output.append(copied)
    return output


def _fill_swiss_defaults(rows: List[dict]) -> List[dict]:
    output = []
    for row in rows:
        copied = dict(row)
        for key, value in SWISS_DEFAULTS.items():
            _set_if_missing(copied, key, value)
        output.append(copied)
    return output


def _write_event_teams(
    rows: List[dict],
    output_path: str,
    manifest_path: str,
    source_name: str,
    source_url: str | None = None,
) -> Dict[str, object]:
    write_matches_csv(output_path, rows)
    manifest: Dict[str, object] = {
        "source": source_name,
        "rows": len(rows),
        "teams": [row.get("team") for row in rows],
        "output_path": output_path,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_url:
        manifest["source_url"] = source_url
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest


def _write_team_rows(
    rows: List[dict],
    output_path: str,
    manifest_path: str,
    source_name: str,
    source_url: str | None = None,
    limit: int | None = None,
) -> Dict[str, object]:
    write_matches_csv(output_path, rows)
    manifest: Dict[str, object] = {
        "source": source_name,
        "rows": len(rows),
        "teams": [row.get("team") for row in rows],
        "output_path": output_path,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_url:
        manifest["source_url"] = source_url
    if limit is not None:
        manifest["limit"] = limit
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest


def _write_player_rows(
    rows: List[dict],
    output_path: str,
    manifest_path: str,
    source_name: str,
    default_date: str,
    source_url: str | None = None,
) -> Dict[str, object]:
    write_matches_csv(output_path, rows)
    manifest: Dict[str, object] = {
        "source": source_name,
        "rows": len(rows),
        "date": default_date,
        "teams": sorted({row.get("team") for row in rows if row.get("team")}),
        "players": [row.get("player") for row in rows],
        "output_path": output_path,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_url:
        manifest["source_url"] = source_url
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest


def _resolve_optional_path(value: Any, base_dir: str) -> str | None:
    if value in (None, ""):
        return None
    return _resolve_path(value, base_dir)


def _resolve_path(value: Any, base_dir: str) -> str:
    path = str(value)
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))


def _read_team_names(path: str) -> List[str]:
    names = []
    for row in read_matches_csv(path):
        value = row.get("team") or row.get("name") or row.get("team_name")
        if value not in (None, ""):
            names.append(str(value))
    return names


def _set_if_missing(row: Dict[str, Any], key: str, value: Any) -> None:
    if row.get(key) not in (None, "") or value in (None, ""):
        return
    row[key] = value


def _team_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.strip().lower()).strip("-") or "job"
