from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html import unescape
from typing import Any, Callable, Dict, Iterable, List
from urllib.parse import quote, urlencode, urljoin, urlparse

from .sources import HttpCache


FIVEE_BASE_URL = "https://csgo.5eplay.com"
FIVEE_EVENT_BASE_URL = "https://event.5eplay.com"
FIVEE_RESULT_API_URL = "https://app.5eplay.com/api/tournament/session_result_list"
FIVEE_DISPLAY_TZ = timezone(timedelta(hours=8))

FIVEE_RESULT_MATCH_FIELDS = [
    "source",
    "date",
    "start_time",
    "match_id",
    "event",
    "event_id",
    "event_grade",
    "event_grade_label",
    "event_status",
    "stage",
    "stage_detail",
    "round",
    "best_of",
    "match_status",
    "team1",
    "team1_id",
    "team1_rank",
    "team2",
    "team2_id",
    "team2_rank",
    "team1_match_score",
    "team2_match_score",
    "winner",
    "team1_odds",
    "team2_odds",
    "map_count",
    "source_match_url",
    "fetched_at",
]
FIVEE_RESULT_MAP_FIELDS = [
    "source",
    "date",
    "match_id",
    "bout_num",
    "map",
    "team1",
    "team2",
    "team1_score",
    "team2_score",
    "winner",
    "map_status",
    "source_match_url",
    "fetched_at",
]


@dataclass
class FiveETeamPage:
    team: Dict[str, Any]
    players: List[Dict[str, Any]]
    maps: List[Dict[str, Any]]
    matches: List[Dict[str, Any]]


class FiveETeamParser:
    """Parse public 5E team data pages into CSV-friendly rows."""

    def parse_team_html(self, html: str, source_url: str) -> FiveETeamPage:
        if _looks_like_waf_challenge(html):
            return FiveETeamPage(team=_base_team_row(source_url, status="blocked"), players=[], maps=[], matches=[])
        if "datas-team-page" not in html:
            return FiveETeamPage(team=_base_team_row(source_url, status="unparsed"), players=[], maps=[], matches=[])

        team_name = _extract_team_name(html, source_url)
        slug = _slug_from_url(source_url)
        team = {
            "source": "5e",
            "status": "ok" if team_name else "unparsed",
            "team": team_name,
            "slug": slug,
            "region": _extract_flag_region(html),
            "world_rank": _extract_labeled_int(html, "世界排名"),
            "regional_rank": _extract_regional_rank(html),
            "regional_label": _extract_regional_label(html),
            "winrate": _extract_percent_value(html, "胜率"),
            "wins": _extract_record_value(html, "win"),
            "draws": _extract_record_value(html, "flat"),
            "losses": _extract_record_value(html, "lose"),
            "rating": _extract_stat_value(html, "Rating"),
            "kd": _extract_stat_value(html, "K/D"),
            "maps": _extract_stat_value(html, "地图数"),
            "prize": _extract_stat_value(html, "总奖金"),
            "source_team_url": source_url,
        }
        players = _extract_players(html, team_name, source_url)
        maps = _extract_maps(html, team_name, source_url)
        matches = _extract_matches(html, team_name, source_url)
        return FiveETeamPage(team=team, players=players, maps=maps, matches=matches)


def collect_fivee_team_pages(
    urls: Iterable[str],
    cache_dir: str,
    output_dir: str,
    refresh: bool = False,
    delay_seconds: float = 3.0,
    start_date: str | None = None,
    end_date: str | None = None,
    fetcher: Callable[[str, Dict[str, str]], str] | None = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    cache = HttpCache(cache_dir, fetcher=fetcher)
    parser = FiveETeamParser()
    team_rows: List[Dict[str, Any]] = []
    player_rows: List[Dict[str, Any]] = []
    map_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    page_reports: List[Dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    start = _parse_iso_date(start_date) if start_date else None
    end = _parse_iso_date(end_date) if end_date else None

    normalized_urls = [_normalize_fivee_url(url) for url in urls if url and url.strip()]
    for index, url in enumerate(normalized_urls):
        if index and delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            html = cache.get(url, refresh=refresh)
            page = parser.parse_team_html(html, source_url=url)
            page.team["fetched_at"] = fetched_at
            team_rows.append(page.team)
            for row in page.players:
                row["fetched_at"] = fetched_at
            for row in page.maps:
                row["fetched_at"] = fetched_at
            selected_matches = [row for row in page.matches if _date_in_range(row.get("date"), start, end)]
            for row in selected_matches:
                row["fetched_at"] = fetched_at
            player_rows.extend(page.players)
            map_rows.extend(page.maps)
            match_rows.extend(selected_matches)
            page_reports.append(
                {
                    "url": url,
                    "status": page.team.get("status"),
                    "team": page.team.get("team"),
                    "players": len(page.players),
                    "maps": len(page.maps),
                    "matches": len(selected_matches),
                    "all_matches": len(page.matches),
                    "cache_path": cache.path_for(url),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive for live collection.
            team_rows.append({**_base_team_row(url, status="error"), "error": str(exc), "fetched_at": fetched_at})
            page_reports.append({"url": url, "status": "error", "error": str(exc), "players": 0, "maps": 0, "matches": 0, "all_matches": 0})

    teams_path = os.path.join(output_dir, "fivee_teams.csv")
    players_path = os.path.join(output_dir, "fivee_players.csv")
    maps_path = os.path.join(output_dir, "fivee_maps.csv")
    matches_path = os.path.join(output_dir, "fivee_matches.csv")
    manifest_path = os.path.join(output_dir, "fivee_manifest.json")
    _write_csv(teams_path, team_rows)
    _write_csv(players_path, player_rows)
    _write_csv(maps_path, map_rows)
    _write_csv(matches_path, match_rows)
    manifest = {
        "source": "5e",
        "kind": "team-pages",
        "urls": len(normalized_urls),
        "teams": len(team_rows),
        "players": len(player_rows),
        "maps": len(map_rows),
        "matches": len(match_rows),
        "match_window_start": start_date,
        "match_window_end": end_date,
        "ok_pages": sum(1 for report in page_reports if report.get("status") == "ok"),
        "blocked_pages": sum(1 for report in page_reports if report.get("status") == "blocked"),
        "error_pages": sum(1 for report in page_reports if report.get("status") == "error"),
        "cache_dir": os.path.abspath(cache_dir),
        "output_dir": os.path.abspath(output_dir),
        "teams_path": teams_path,
        "players_path": players_path,
        "maps_path": maps_path,
        "matches_path": matches_path,
        "page_reports": page_reports,
        "updated_at": fetched_at,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    manifest["manifest_path"] = manifest_path
    return manifest


def parse_fivee_match_results(payload: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Flatten 5E event result API JSON into match and map rows."""
    matches: List[Dict[str, Any]] = []
    maps: List[Dict[str, Any]] = []
    for item in _fivee_result_items(payload):
        match = _parse_fivee_result_match(item)
        if not match:
            continue
        matches.append(match)
        maps.extend(_parse_fivee_result_maps(item, match))
    return matches, maps


def collect_fivee_match_results(
    cache_dir: str,
    output_dir: str,
    start_date: str,
    end_date: str,
    refresh: bool = False,
    delay_seconds: float = 0.5,
    page_size: int = 100,
    max_pages: int = 1000,
    grades: str = "",
    fetcher: Callable[[str, Dict[str, str]], str] | None = None,
) -> Dict[str, Any]:
    """Collect completed/public 5E match-result pages backwards from end_date."""
    os.makedirs(output_dir, exist_ok=True)
    cache = HttpCache(cache_dir, fetcher=fetcher)
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    if start is None or end is None:
        raise ValueError("start_date and end_date must use YYYY-MM-DD")
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    fetched_at = datetime.now(timezone.utc).isoformat()
    page_token = _initial_fivee_result_page_token(end)
    match_rows: List[Dict[str, Any]] = []
    map_rows: List[Dict[str, Any]] = []
    page_reports: List[Dict[str, Any]] = []
    seen_match_ids: set[str] = set()
    pages_fetched = 0

    for page_index in range(max_pages):
        if page_index and delay_seconds > 0:
            time.sleep(delay_seconds)
        url = _fivee_result_page_url(page_token=page_token, page_size=page_size, grades=grades)
        text = cache.get(url, refresh=refresh)
        pages_fetched += 1
        payload = json.loads(text)
        raw_items = _fivee_result_items(payload)
        parsed_matches, parsed_maps = parse_fivee_match_results(payload)
        selected_ids: set[str] = set()
        page_dates = [row["date"] for row in parsed_matches if row.get("date")]

        for row in parsed_matches:
            match_id = row.get("match_id", "")
            if not _date_in_range(row.get("date"), start, end) or match_id in seen_match_ids:
                continue
            row["fetched_at"] = fetched_at
            match_rows.append(row)
            selected_ids.add(match_id)
            seen_match_ids.add(match_id)

        for row in parsed_maps:
            if row.get("match_id") not in selected_ids:
                continue
            row["fetched_at"] = fetched_at
            map_rows.append(row)

        next_token = _fivee_result_page_token_before(raw_items)
        page_reports.append(
            {
                "url": url,
                "raw_matches": len(raw_items),
                "selected_matches": len(selected_ids),
                "min_date": min(page_dates) if page_dates else "",
                "max_date": max(page_dates) if page_dates else "",
                "next_page_token": next_token,
                "cache_path": cache.path_for(url),
            }
        )
        if not raw_items or not next_token or next_token == page_token:
            break
        if page_dates and _parse_iso_date(max(page_dates)) < start:
            break
        page_token = next_token

    matches_path = os.path.join(output_dir, "fivee_match_results.csv")
    maps_path = os.path.join(output_dir, "fivee_match_maps.csv")
    manifest_path = os.path.join(output_dir, "fivee_match_results_manifest.json")
    _write_csv(matches_path, match_rows, fieldnames=FIVEE_RESULT_MATCH_FIELDS)
    _write_csv(maps_path, map_rows, fieldnames=FIVEE_RESULT_MAP_FIELDS)
    manifest = {
        "source": "5e",
        "kind": "match-results",
        "start_date": start_date,
        "end_date": end_date,
        "pages_fetched": pages_fetched,
        "page_size": page_size,
        "max_pages": max_pages,
        "matches": len(match_rows),
        "maps": len(map_rows),
        "cache_dir": os.path.abspath(cache_dir),
        "output_dir": os.path.abspath(output_dir),
        "matches_path": matches_path,
        "maps_path": maps_path,
        "page_reports": page_reports,
        "updated_at": fetched_at,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    manifest["manifest_path"] = manifest_path
    return manifest


def read_urls(path: str) -> List[str]:
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def _base_team_row(source_url: str, status: str) -> Dict[str, Any]:
    return {"source": "5e", "status": status, "team": "", "slug": _slug_from_url(source_url), "source_team_url": source_url}


def _normalize_fivee_url(url: str) -> str:
    stripped = url.strip()
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return stripped
    if stripped.startswith("/"):
        return urljoin(FIVEE_BASE_URL, stripped)
    return urljoin(f"{FIVEE_BASE_URL}/data/team/", stripped)


def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] if path else ""


def _looks_like_waf_challenge(html: str) -> bool:
    markers = ["aliyun_waf", "acw_sc__v2", "TJCaptcha", "captcha"]
    lower = html.lower()
    return any(marker.lower() in lower for marker in markers) and "datas-team-page" not in lower


def _extract_team_name(html: str, source_url: str) -> str:
    for pattern in (
        r'<span[^>]+class="[^"]*\bcur\b[^"]*"[^>]*>(.*?)</span>',
        r'<span[^>]+title="([^"]+)"[^>]*>\s*\1\s*</span>',
        r'<p[^>]+class="[^"]*\bname\b[^"]*"[^>]*>.*?<span[^>]*>(.*?)</span>',
    ):
        match = re.search(pattern, html, flags=re.I | re.S)
        if match:
            return _clean_text(match.group(1))
    return _slug_from_url(source_url).replace("-", " ").title()


def _extract_flag_region(html: str) -> str:
    match = re.search(r'<img[^>]+/flag-big/([a-z]{2})\.gif', html, flags=re.I)
    return match.group(1).lower() if match else ""


def _extract_labeled_int(html: str, label: str) -> int | None:
    match = re.search(
        r'<span[^>]+class="[^"]*\bval\b[^"]*"[^>]*>\s*([0-9,]+)\s*</span>\s*'
        r'<span[^>]+class="[^"]*\blabel\b[^"]*"[^>]*>\s*' + re.escape(label) + r"\s*</span>",
        html,
        flags=re.I | re.S,
    )
    return _to_int(match.group(1)) if match else None


def _extract_regional_rank(html: str) -> int | None:
    ranks = re.findall(
        r'<li>\s*<span[^>]+class="[^"]*\bval\b[^"]*"[^>]*>\s*([0-9,]+)\s*</span>\s*'
        r'<span[^>]+class="[^"]*\blabel\b[^"]*"[^>]*>\s*([^<]+)\s*</span>\s*</li>',
        html,
        flags=re.I | re.S,
    )
    for value, label in ranks:
        if _clean_text(label) != "世界排名":
            return _to_int(value)
    return None


def _extract_regional_label(html: str) -> str:
    ranks = re.findall(
        r'<li>\s*<span[^>]+class="[^"]*\bval\b[^"]*"[^>]*>\s*[0-9,]+\s*</span>\s*'
        r'<span[^>]+class="[^"]*\blabel\b[^"]*"[^>]*>\s*([^<]+)\s*</span>\s*</li>',
        html,
        flags=re.I | re.S,
    )
    for label in ranks:
        cleaned = _clean_text(label)
        if cleaned != "世界排名":
            return cleaned
    return ""


def _extract_percent_value(html: str, label: str) -> float | None:
    match = re.search(
        r'<span[^>]+class="[^"]*\bval\b[^"]*"[^>]*>\s*([0-9.]+)%\s*</span>\s*'
        r'<span[^>]+class="[^"]*\blabel\b[^"]*"[^>]*>\s*' + re.escape(label) + r"\s*</span>",
        html,
        flags=re.I | re.S,
    )
    return _percent_to_float(match.group(1)) if match else None


def _extract_record_value(html: str, class_name: str) -> int | None:
    match = re.search(r'<span[^>]+class="[^"]*\b' + re.escape(class_name) + r'\b[^"]*"[^>]*>\s*([0-9,]+)\s*</span>', html, flags=re.I | re.S)
    return _to_int(match.group(1)) if match else None


def _extract_stat_value(html: str, label: str) -> float | int | str | None:
    value = _value_before_label(html, label)
    if value is None:
        return None
    if value in {"$-", "-", "$ -"}:
        return ""
    return _to_number(value)


def _value_before_label(html: str, label: str) -> str | None:
    label_match = re.search(r'<span[^>]+class="[^"]*\blabel\b[^"]*"[^>]*>\s*' + re.escape(label) + r"\s*</span>", html, flags=re.I | re.S)
    if not label_match:
        return None
    window = html[max(0, label_match.start() - 500) : label_match.start()]
    values = re.findall(r'<span[^>]+class="[^"]*\bval\b[^"]*"[^>]*>\s*(.*?)\s*</span>', window, flags=re.I | re.S)
    return _clean_text(values[-1]) if values else None


def _extract_players(html: str, team: str, source_url: str) -> List[Dict[str, Any]]:
    fragment = _between(html, '<div class="floatl team-players">', '<div class="clearfix val-shows')
    rows: List[Dict[str, Any]] = []
    for href, body in re.findall(r'<a[^>]+href="([^"]*/data/player/[^"]+)"[^>]*>(.*?)</a>', fragment, flags=re.I | re.S):
        player = _clean_text(_last_match(body, r'<span[^>]*>(.*?)</span>') or _last_match(body, r'alt="([^"]+)"') or "")
        if not player:
            continue
        rows.append(
            {
                "source": "5e",
                "team": team,
                "player": player,
                "slug": _slug_from_url(href),
                "source_player_url": urljoin(FIVEE_BASE_URL, href),
                "source_team_url": source_url,
            }
        )
    return rows


def _extract_maps(html: str, team: str, source_url: str) -> List[Dict[str, Any]]:
    map_names = [_clean_text(name) for name in re.findall(r'<p[^>]+class="[^"]*\bmap-name\b[^"]*"[^>]*>(.*?)</p>', html, flags=re.I | re.S)]
    tables = re.findall(r'<table[^>]+class="[^"]*\btb-map\b[^"]*"[^>]*>(.*?)</table>', html, flags=re.I | re.S)
    rows: List[Dict[str, Any]] = []
    for index, table in enumerate(tables):
        row = {
            "source": "5e",
            "team": team,
            "map": map_names[index] if index < len(map_names) else "",
            "source_team_url": source_url,
        }
        row.update(_parse_map_table(table))
        rows.append(row)
    return rows


def _extract_matches(html: str, team: str, source_url: str) -> List[Dict[str, Any]]:
    raw_sessions = _extract_session_records(html)
    rows: List[Dict[str, Any]] = []
    for session in raw_sessions:
        opponent = session.get("opponent") or {}
        score1 = _to_int(session.get("team1_score")) or 0
        score2 = _to_int(session.get("team2_score")) or 0
        won = str(session.get("is_win")) == "1"
        draw = score1 == score2 and not won
        high, low = max(score1, score2), min(score1, score2)
        team_score = high if won else low
        opponent_score = low if won else high
        session_id = str(session.get("session_id") or "")
        rows.append(
            {
                "source": "5e",
                "team": team,
                "opponent": _clean_text(opponent.get("team_tag") or opponent.get("team_name") or ""),
                "opponent_slug": opponent.get("team_alias") or "",
                "date": _date_from_timestamp(session.get("session_start_time")),
                "event": _clean_text(session.get("event_name") or ""),
                "event_alias": session.get("event_alias") or "",
                "stage": _clean_text(session.get("session_section") or ""),
                "best_of": _best_of_from_format(session.get("format")),
                "team_score": team_score,
                "opponent_score": opponent_score,
                "won": 1 if won else 0,
                "draw": 1 if draw else 0,
                "session_id": session_id,
                "has_match_stats": 1 if str(session.get("is_match_stats")) == "1" else 0,
                "source_session_url": f"https://csgo.5eplay.com/session/{session_id}" if session_id else "",
                "source_team_url": source_url,
            }
        )
    return rows


def _fivee_result_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data") or {}
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    rows = data.get("matches") or data.get("list") or data.get("items") or []
    return [item for item in rows if isinstance(item, dict)]


def _parse_fivee_result_match(item: Dict[str, Any]) -> Dict[str, Any]:
    mc_info = item.get("mc_info") or {}
    state = item.get("state") or {}
    tt_info = item.get("tt_info") or {}
    if not isinstance(mc_info, dict) or not isinstance(state, dict) or not isinstance(tt_info, dict):
        return {}

    match_id = str(mc_info.get("id") or "")
    if not match_id:
        return {}
    team1 = _team_info(mc_info.get("t1_info"))
    team2 = _team_info(mc_info.get("t2_info"))
    if not team1.get("name") or not team2.get("name"):
        return {}

    team1_score = _to_int(state.get("t1_score"))
    team2_score = _to_int(state.get("t2_score"))
    winner = ""
    if team1_score is not None and team2_score is not None:
        if team1_score > team2_score:
            winner = team1["name"]
        elif team2_score > team1_score:
            winner = team2["name"]

    start_at = _display_datetime_from_timestamp(mc_info.get("plan_ts"))
    bout_states = item.get("state", {}).get("bout_states") or []
    if not isinstance(bout_states, list):
        bout_states = []
    source_match_url = f"{FIVEE_EVENT_BASE_URL}/csgo/match/{match_id}"
    return {
        "source": "5e",
        "date": start_at.date().isoformat() if start_at else "",
        "start_time": start_at.strftime("%Y-%m-%d %H:%M:%S%z") if start_at else "",
        "match_id": match_id,
        "event": _clean_text(tt_info.get("disp_name") or ""),
        "event_id": tt_info.get("id") or "",
        "event_grade": tt_info.get("grade") or mc_info.get("grade") or "",
        "event_grade_label": tt_info.get("grade_label") or tt_info.get("special_grade_label") or "",
        "event_status": tt_info.get("status") or "",
        "stage": _clean_text(mc_info.get("tt_stage") or ""),
        "stage_detail": _clean_text(mc_info.get("tt_stage_desc") or ""),
        "round": _clean_text(mc_info.get("round_name") or ""),
        "best_of": _best_of_from_api_format(mc_info.get("format"), len(bout_states)),
        "match_status": _status_label(state.get("status")),
        "team1": team1["name"],
        "team1_id": team1["id"],
        "team1_rank": team1["rank"],
        "team2": team2["name"],
        "team2_id": team2["id"],
        "team2_rank": team2["rank"],
        "team1_match_score": team1_score,
        "team2_match_score": team2_score,
        "winner": winner,
        "team1_odds": _to_number(state.get("t1_odds")),
        "team2_odds": _to_number(state.get("t2_odds")),
        "map_count": len(bout_states),
        "source_match_url": source_match_url,
    }


def _parse_fivee_result_maps(item: Dict[str, Any], match: Dict[str, Any]) -> List[Dict[str, Any]]:
    state = item.get("state") or {}
    bout_states = state.get("bout_states") or []
    if not isinstance(bout_states, list):
        return []
    rows: List[Dict[str, Any]] = []
    for bout in bout_states:
        if not isinstance(bout, dict):
            continue
        result = str(bout.get("result") or "")
        t1_score = _to_int(bout.get("t1_score"))
        t2_score = _to_int(bout.get("t2_score"))
        # 地图胜负以回合分为准(分高者胜)，与 match 层 (_parse_fivee_result_match)
        # 判定逻辑保持一致。5E 源数据的 result 字段偶尔标反(实测有 6 张图如此)，
        # 仅在回合分缺失或打平(如 forfeit/Default 占位图)时才回退到 result 字段。
        winner = ""
        if t1_score is not None and t2_score is not None and t1_score != t2_score:
            winner = match.get("team1", "") if t1_score > t2_score else match.get("team2", "")
        elif result == "t1":
            winner = match.get("team1", "")
        elif result == "t2":
            winner = match.get("team2", "")
        rows.append(
            {
                "source": "5e",
                "date": match.get("date", ""),
                "match_id": match.get("match_id", ""),
                "bout_num": _to_int(bout.get("bout_num")),
                "map": bout.get("map_name") or "",
                "team1": match.get("team1", ""),
                "team2": match.get("team2", ""),
                "team1_score": t1_score,
                "team2_score": t2_score,
                "winner": winner,
                "map_status": _status_label(bout.get("status")),
                "source_match_url": match.get("source_match_url", ""),
            }
        )
    return rows


def _team_info(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {"id": "", "name": "", "rank": ""}
    return {
        "id": str(value.get("id") or ""),
        "name": _clean_text(value.get("disp_name") or value.get("name") or value.get("team_name") or ""),
        "rank": str(value.get("rank") or value.get("v_rank") or ""),
    }


def _status_label(value: Any) -> str:
    raw = str(value)
    return {"0": "upcoming", "1": "live", "2": "completed", "3": "cancelled"}.get(raw, raw if raw else "")


def _best_of_from_api_format(value: Any, map_count: int) -> int | str:
    raw = _to_int(value)
    if raw in {1, 3, 5}:
        return raw
    legacy = {0: 1, 2: 3, 4: 5}
    if raw in legacy:
        return legacy[raw]
    return map_count if map_count else ""


def _display_datetime_from_timestamp(value: Any) -> datetime | None:
    timestamp = _to_int(value)
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=FIVEE_DISPLAY_TZ)


def _initial_fivee_result_page_token(end: date) -> str:
    return f"{(end + timedelta(days=1)).isoformat()} 00:00:00,"


def _fivee_result_page_url(page_token: str, page_size: int, grades: str = "") -> str:
    return FIVEE_RESULT_API_URL + "?" + urlencode(
        {
            "game_type": "1",
            "order_by": "asc",
            "grades": grades,
            "page_size": str(page_size),
            "page_token": page_token,
        },
        quote_via=quote,
    )


def _fivee_result_page_token_before(items: List[Dict[str, Any]]) -> str:
    timestamps = [_to_int((item.get("mc_info") or {}).get("plan_ts")) for item in items if isinstance(item.get("mc_info"), dict)]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if not timestamps:
        return ""
    earliest = min(timestamps)
    ids = [
        str((item.get("mc_info") or {}).get("id") or "")
        for item in items
        if _to_int((item.get("mc_info") or {}).get("plan_ts")) == earliest
    ]
    ids = [match_id for match_id in ids if match_id]
    start_at = datetime.fromtimestamp(earliest, tz=FIVEE_DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")
    return start_at + "," + ",".join(ids)


def _extract_session_records(html: str) -> List[Dict[str, Any]]:
    array_text = _extract_js_array(html, "'session'")
    if not array_text:
        array_text = _extract_js_array(html, '"session"')
    if not array_text:
        return []
    try:
        payload = json.loads(array_text)
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)]


def _extract_js_array(html: str, key: str) -> str:
    key_index = html.find(key)
    if key_index == -1:
        return ""
    start = html.find("[", key_index)
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    quote = ""
    for index in range(start, len(html)):
        char = html[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {"'", '"'}:
            in_string = True
            quote = char
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return html[start : index + 1]
    return ""


def _date_from_timestamp(value: Any) -> str:
    timestamp = _to_int(value)
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def _best_of_from_format(value: Any) -> int | str:
    raw = str(value)
    if raw == "0":
        return 1
    if raw == "2":
        return 3
    if raw == "4":
        return 5
    return ""


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_in_range(value: Any, start: date | None, end: date | None) -> bool:
    if not value:
        return False
    current = _parse_iso_date(str(value))
    if current is None:
        return False
    if start and current < start:
        return False
    if end and current > end:
        return False
    return True


def _parse_map_table(table: str) -> Dict[str, Any]:
    values = _table_label_values(table)
    record = values.get("胜 / 负 / 平", "")
    wins, losses, draws = _parse_record(record)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": _percent_to_float(values.get("胜率")),
        "rounds": _to_int(values.get("总回合数")),
        "opening_winrate": _percent_to_float(values.get("取得首杀后回合胜率")),
        "opening_denied_winrate": _percent_to_float(values.get("被首杀后回合胜率")),
        "last_big_win_score": _first_score_for_label(table, "最近一场大胜"),
        "last_big_win_opponent": _first_opponent_for_label(table, "最近一场大胜"),
        "last_big_loss_score": _first_score_for_label(table, "最近一场惨败"),
        "last_big_loss_opponent": _first_opponent_for_label(table, "最近一场惨败"),
    }


def _table_label_values(table: str) -> Dict[str, str]:
    rows = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.I | re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.I | re.S)
        if len(cells) >= 2:
            rows[_clean_text(cells[0])] = _clean_text(cells[-1])
    return rows


def _first_score_for_label(table: str, label: str) -> str:
    row = _row_for_label(table, label)
    return _clean_text(_last_match(row, r'<td[^>]+class="[^"]*(?:win|lose)[^"]*"[^>]*>(.*?)</td>') or "")


def _first_opponent_for_label(table: str, label: str) -> str:
    row = _row_for_label(table, label)
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)
    return _clean_text(cells[-1]) if len(cells) >= 3 else ""


def _row_for_label(table: str, label: str) -> str:
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.I | re.S):
        if label in _clean_text(row):
            return row
    return ""


def _parse_record(value: str) -> tuple[int | None, int | None, int | None]:
    numbers = [_to_int(number) for number in re.findall(r"\d+", value)]
    while len(numbers) < 3:
        numbers.append(None)
    return numbers[0], numbers[1], numbers[2]


def _between(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    if start_index == -1:
        return ""
    end_index = text.find(end, start_index)
    return text[start_index:] if end_index == -1 else text[start_index:end_index]


def _last_match(text: str, pattern: str) -> str:
    matches = re.findall(pattern, text, flags=re.I | re.S)
    return matches[-1] if matches else ""


def _clean_text(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9-]", "", str(value))
    if not cleaned:
        return None
    return int(cleaned)


def _to_number(value: Any) -> float | int | str | None:
    cleaned = _clean_text(value).replace(",", "")
    if not cleaned:
        return None
    try:
        number = float(cleaned.replace("$", ""))
    except ValueError:
        return cleaned
    return int(number) if number.is_integer() else number


def _percent_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(str(value).replace("%", "").strip()) / 100.0, 4)
    except ValueError:
        return None


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    selected_fieldnames: List[str] = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in selected_fieldnames:
                selected_fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
