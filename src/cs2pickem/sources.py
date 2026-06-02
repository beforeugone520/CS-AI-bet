from __future__ import annotations

import hashlib
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import date
from html import unescape
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from .cleaning import parse_date


DEFAULT_USER_AGENT = "cs2pickem-offline-research/0.1 (+personal analytics)"


class HttpCache:
    """Small disk cache for polite, repeatable public-page collection."""

    def __init__(self, cache_dir: str, fetcher: Callable[[str, Dict[str, str]], str] | None = None) -> None:
        self.cache_dir = cache_dir
        self.fetcher = fetcher or _resilient_fetch
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, url: str, refresh: bool = False) -> str:
        path = self.path_for(url)
        if not refresh and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                return handle.read()
        payload = self.fetcher(url, {"User-Agent": DEFAULT_USER_AGENT})
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return payload

    def path_for(self, url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{digest}.html")


@dataclass
class HltvResultParser:
    """Parse a conservative subset of HLTV result-card HTML into match rows."""

    def parse_results_html(self, html: str) -> List[Dict[str, Any]]:
        cards = _extract_result_cards(html)
        rows = []
        for card in cards:
            team1 = _first_text(card, ["team1", "team-cell", "team"])
            team2 = _last_text(card, ["team2", "team-cell", "team"])
            if not team1 or not team2 or team1 == team2:
                continue
            scores = _extract_scores(card)
            if len(scores) < 2:
                continue
            winner = team1 if scores[0][0] >= scores[1][0] else team2
            rows.append(
                {
                    "date": _extract_date(card),
                    "event": _first_text(card, ["event-name", "event"]) or "HLTV",
                    "event_tier": _infer_event_tier(card),
                    "status": "completed",
                    "team1": team1,
                    "team2": team2,
                    "winner": winner,
                    "best_of": _infer_best_of(card),
                    "map": (_first_text(card, ["map-text", "map"]) or "unknown").lower(),
                    "source": "hltv",
                    "source_match_url": _extract_href(card),
                }
            )
        return rows


@dataclass
class HltvEventParser:
    """Parse HLTV-like event pages into participant/team rows."""

    def parse_teams_html(self, html: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for anchor_body, href, fragment in _extract_team_anchor_fragments(html):
            team = _extract_team_name(anchor_body)
            normalized = team.casefold()
            if not team or normalized in seen:
                continue
            seen.add(normalized)
            ranks = _extract_hash_numbers(fragment)
            seed = _first_int_by_classes(fragment, ["event-seed", "seed"])
            world_rank = _first_int_by_classes(fragment, ["world-rank", "worldRanking", "ranking"])
            rows.append(
                {
                    "team": team,
                    "seed": seed if seed is not None else (ranks[0] if ranks else len(rows) + 1),
                    "world_rank": world_rank if world_rank is not None else (ranks[1] if len(ranks) > 1 else ""),
                    "qualification": _extract_qualification(fragment),
                    "source_team_url": _normalize_hltv_url(href),
                }
            )
        return rows


@dataclass
class HltvRankingParser:
    """Parse HLTV-like team ranking pages into Top-N team rows."""

    def parse_rankings_html(self, html: str, limit: int = 80) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for anchor_body, href, fragment in _extract_team_anchor_fragments(html):
            team = _extract_team_name(anchor_body)
            normalized = team.casefold()
            if not team or normalized in seen:
                continue
            rank = _first_int_by_classes(fragment, ["position", "rank", "ranking"])
            if rank is None:
                ranks = _extract_hash_numbers(fragment)
                rank = ranks[0] if ranks else len(rows) + 1
            rows.append(
                {
                    "team": team,
                    "rank": rank,
                    "points": _extract_points(fragment),
                    "region": _first_text(fragment, ["country", "region"]),
                    "source_team_url": _normalize_hltv_url(href),
                }
            )
            seen.add(normalized)
            if len(rows) >= limit:
                break
        return rows


@dataclass
class HltvPlayerStatsParser:
    """Parse HLTV-like player stat tables into merge-ready player rows."""

    def parse_player_stats_html(self, html: str, default_date: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for fragment in _extract_player_rows(html):
            player_body, player_href = _extract_player_anchor(fragment)
            player = _extract_team_name(player_body)
            team = _first_text(fragment, ["team", "team-name"])
            if not player or not team:
                continue
            rows.append(
                {
                    "date": _first_text(fragment, ["date", "snapshot-date"]) or default_date,
                    "team": team,
                    "player": player,
                    "rating": _extract_float_by_classes(fragment, ["rating", "rating2", "rating-20"], 1.0),
                    "kd": _extract_float_by_classes(fragment, ["kd", "k-d", "kd-ratio"], 1.0),
                    "opening_success": _extract_ratio_by_classes(fragment, ["opening-success", "opening", "fk-success"], 0.5),
                    "clutch_winrate": _extract_ratio_by_classes(fragment, ["clutch-winrate", "clutch", "clutch-success"], 0.5),
                    "is_substitute": 1 if _is_substitute_fragment(fragment) else 0,
                    "source_player_url": _normalize_hltv_url(player_href),
                }
            )
        return rows


def parse_version_log(text: str) -> List[Tuple[date, str]]:
    versions: List[Tuple[date, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "," in stripped:
            raw_date, tag = stripped.split(",", 1)
        else:
            parts = stripped.split(None, 1)
            if len(parts) != 2:
                continue
            raw_date, tag = parts
        versions.append((parse_date(raw_date.strip()), tag.strip()))
    return sorted(versions, key=lambda item: item[0])


def annotate_version_tags(rows: Iterable[Dict[str, Any]], version_log: Sequence[Tuple[date, str]]) -> List[Dict[str, Any]]:
    annotated = []
    for row in rows:
        copied = dict(row)
        played_at = parse_date(copied["date"])
        copied["version_tag"] = _latest_version_tag(played_at, version_log)
        annotated.append(copied)
    return annotated


def _urllib_fetch(url: str, headers: Dict[str, str]) -> str:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _resilient_fetch(url: str, headers: Dict[str, str]) -> str:
    errors = []
    for name, fetcher in (
        ("urllib", _urllib_fetch),
        ("requests", _requests_fetch),
        ("curl", _curl_fetch),
    ):
        try:
            return fetcher(url, headers)
        except Exception as exc:  # pragma: no cover - exercised via monkeypatched tests.
            errors.append(f"{name}: {exc}")
    raise RuntimeError(f"all fetchers failed for {url}: {'; '.join(errors)}")


def _requests_fetch(url: str, headers: Dict[str, str]) -> str:
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError("requests is not installed; install cs2pickem[scrape] for this fallback") from exc
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def _curl_fetch(url: str, headers: Dict[str, str]) -> str:
    command = ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "30"]
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])
    command.append(url)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def _extract_result_cards(html: str) -> List[str]:
    cards = re.findall(r'<div[^>]+class="[^"]*result-con[^"]*"[^>]*>(.*?)</div>\s*</div>|<div[^>]+class="[^"]*result-con[^"]*"[^>]*>(.*?)</div>', html, flags=re.I | re.S)
    flattened = [first or second for first, second in cards if first or second]
    if flattened:
        return flattened
    return re.findall(r"<a[^>]+href=\"/matches/.*?</a>", html, flags=re.I | re.S)


def _extract_team_anchor_fragments(html: str) -> List[Tuple[str, str, str]]:
    anchors: List[Tuple[str, str, str]] = []
    pattern = r'<a[^>]+href="(?P<href>(?:https://www\.hltv\.org)?/team/[^"]+)"[^>]*>(?P<body>.*?)</a>'
    matches = list(re.finditer(pattern, html, flags=re.I | re.S))
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(html)
        anchors.append((match.group("body"), match.group("href"), _team_card_fragment(html, match.start(), match.end(), next_start)))
    return anchors


def _extract_player_rows(html: str) -> List[str]:
    rows = re.findall(r"<tr[^>]*(?:class=\"[^\"]*player[^\"]*\"[^>]*)?>(?:(?!</tr>).)*?/player/.*?</tr>", html, flags=re.I | re.S)
    if rows:
        return rows
    return re.findall(r"<div[^>]+class=\"[^\"]*player[^\"]*\"[^>]*>.*?</div>", html, flags=re.I | re.S)


def _extract_player_anchor(fragment: str) -> Tuple[str, str]:
    match = re.search(r'<a[^>]+href="(?P<href>(?:https://www\.hltv\.org)?/player/[^"]+)"[^>]*>(?P<body>.*?)</a>', fragment, flags=re.I | re.S)
    if not match:
        return "", ""
    return match.group("body"), match.group("href")


def _team_card_fragment(html: str, anchor_start: int, anchor_end: int, next_anchor_start: int) -> str:
    prefix = html[:anchor_start]
    div_pattern = r'<div[^>]+class="[^"]*\bteam(?:-[a-z0-9]+)?\b[^"]*"[^>]*>'
    divs = list(re.finditer(div_pattern, prefix, flags=re.I | re.S))
    if divs:
        start = divs[-1].start()
        close = html.find("</div>", anchor_end, next_anchor_start)
        if close != -1:
            return html[start : close + len("</div>")]
    return html[anchor_start:next_anchor_start]


def _first_text(fragment: str, class_names: Sequence[str]) -> str:
    matches = _texts_by_classes(fragment, class_names)
    return matches[0] if matches else ""


def _last_text(fragment: str, class_names: Sequence[str]) -> str:
    matches = _texts_by_classes(fragment, class_names)
    return matches[-1] if matches else ""


def _texts_by_classes(fragment: str, class_names: Sequence[str]) -> List[str]:
    found: List[str] = []
    for class_name in class_names:
        pattern = rf'<[^>]+class="[^"]*\b{re.escape(class_name)}\b[^"]*"[^>]*>(.*?)</[^>]+>'
        for match in re.findall(pattern, fragment, flags=re.I | re.S):
            text = _strip_tags(match)
            if text and text not in found:
                found.append(text)
    return found


def _extract_team_name(anchor_body: str) -> str:
    text = _first_text(anchor_body, ["team-name", "name", "text-ellipsis"]) or _strip_tags(anchor_body)
    return re.sub(r"\s+#\d+\b.*$", "", text).strip()


def _first_int_by_classes(fragment: str, class_names: Sequence[str]) -> int | None:
    for text in _texts_by_classes(fragment, class_names):
        match = re.search(r"#?\s*(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def _extract_hash_numbers(fragment: str) -> List[int]:
    return [int(value) for value in re.findall(r"#\s*(\d+)", _strip_tags(fragment))]


def _extract_float_by_classes(fragment: str, class_names: Sequence[str], default: float) -> float:
    for text in _texts_by_classes(fragment, class_names):
        value = _parse_float(text)
        if value is not None:
            return value
    return default


def _extract_ratio_by_classes(fragment: str, class_names: Sequence[str], default: float) -> float:
    for text in _texts_by_classes(fragment, class_names):
        value = _parse_ratio(text)
        if value is not None:
            return value
    return default


def _parse_float(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def _parse_ratio(text: str) -> float | None:
    value = _parse_float(text)
    if value is None:
        return None
    return value / 100.0 if "%" in text or value > 1.0 else value


def _is_substitute_fragment(fragment: str) -> bool:
    role = " ".join(_texts_by_classes(fragment, ["role", "status", "note"])).casefold()
    return any(marker in role for marker in ("substitute", "stand-in", "stand in", "sub"))


def _extract_qualification(fragment: str) -> str:
    explicit = _first_text(fragment, ["qualification", "qualifier", "vrs-region", "region"])
    if explicit:
        return explicit
    match = re.search(r"\b(VRS\s*\([^)]+\)|Invite|Qualifier|Regional Standings)\b", _strip_tags(fragment), flags=re.I)
    return match.group(1).strip() if match else ""


def _extract_points(fragment: str) -> int:
    match = re.search(r"([\d,]+)\s*points?", _strip_tags(fragment), flags=re.I)
    return int(match.group(1).replace(",", "")) if match else 0


def _normalize_hltv_url(href: str) -> str:
    return f"https://www.hltv.org{href}" if href.startswith("/") else href


def _extract_scores(fragment: str) -> List[Tuple[int, str]]:
    scores = []
    pattern = r'<[^>]+class="[^"]*(score-won|score-lost|score)[^"]*"[^>]*>(\d+)</[^>]+>'
    for class_name, value in re.findall(pattern, fragment, flags=re.I | re.S):
        scores.append((int(value), class_name.lower()))
    return scores


def _extract_date(fragment: str) -> str:
    text_date = _first_text(fragment, ["date", "match-date", "time"])
    if re.match(r"\d{4}-\d{2}-\d{2}", text_date):
        return text_date[:10]
    data_unix = re.search(r'data-unix="(\d+)"', fragment, flags=re.I)
    if data_unix:
        from datetime import datetime

        return datetime.utcfromtimestamp(int(data_unix.group(1)) / 1000).date().isoformat()
    return "1970-01-01"


def _extract_href(fragment: str) -> str:
    match = re.search(r'href="([^"]+)"', fragment)
    return f"https://www.hltv.org{match.group(1)}" if match and match.group(1).startswith("/") else (match.group(1) if match else "")


def _infer_event_tier(fragment: str) -> str:
    event = (_first_text(fragment, ["event-name", "event"]) or fragment).lower()
    if any(marker in event for marker in ("major", "iem", "blast", "esl pro league", "rmr")):
        return "S"
    if any(marker in event for marker in ("cct", "challenger", "epl", "thunderpick")):
        return "A"
    return "B"


def _infer_best_of(fragment: str) -> int:
    text = _strip_tags(fragment).lower()
    bo_match = re.search(r"\bbo([135])\b", text)
    return int(bo_match.group(1)) if bo_match else 1


def _latest_version_tag(played_at: date, version_log: Sequence[Tuple[date, str]]) -> str:
    latest = "unknown"
    for version_date, tag in version_log:
        if version_date <= played_at:
            latest = tag
        else:
            break
    return latest


def _strip_tags(value: str) -> str:
    return unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value))).strip()
