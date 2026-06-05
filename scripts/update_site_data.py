#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cs2pickem.backtest import standings_from_results
from cs2pickem.data import write_json, write_matches_csv
from cs2pickem.fivee import collect_fivee_match_results
from cs2pickem.sources import HltvResultParser, HttpCache


EVENT_NAME = "IEM Cologne 2026"


@dataclass(frozen=True)
class SourceCandidate:
    name: str
    url: str
    kind: str = "public-results"


def update_site_sources(
    repo_root: Path,
    output_dir: Path,
    source_candidates: list[SourceCandidate] | None = None,
    fivee_candidate: SourceCandidate | None = SourceCandidate("5E", "5e://match-results", "fivee-results"),
    fetcher: Callable[[str, dict[str, str]], str] | None = None,
    start_date: str = "2026-06-01",
    end_date: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_source = ""
    selected_status = "failed"

    primary_candidates = default_primary_sources() if source_candidates is None else source_candidates
    for candidate in list(primary_candidates):
        rows = _attempt_candidate(candidate, output_dir, fetcher, attempts, start_date, end_date)
        if rows:
            selected_rows = rows
            selected_source = candidate.name
            selected_status = "primary_success"
            break

    if not selected_rows and fivee_candidate is not None:
        rows = _attempt_candidate(fivee_candidate, output_dir, fetcher, attempts, start_date, end_date)
        if rows:
            selected_rows = rows
            selected_source = fivee_candidate.name
            selected_status = "fallback_success"

    if not selected_rows:
        cached = _cached_manifest(output_dir)
        if cached:
            cached["status"] = "cached"
            cached["attempts"] = attempts
            return cached
        report = {
            "status": "failed",
            "selected_source": "",
            "completed_results": 0,
            "attempts": attempts,
            "updated_at": _now(),
        }
        write_json(str(output_dir / "failed-source-status.json"), report)
        return report

    completed = _deduplicate_completed(selected_rows)
    standings = standings_from_results(completed, source_label=selected_source)
    results_path = output_dir / "auto_results.csv"
    standings_path = output_dir / "auto_standings.csv"
    write_matches_csv(str(results_path), completed)
    write_matches_csv(str(standings_path), standings)
    report = {
        "status": selected_status,
        "selected_source": selected_source,
        "completed_results": len(completed),
        "results_path": _relative_to_repo(results_path, repo_root),
        "standings_path": _relative_to_repo(standings_path, repo_root),
        "attempts": attempts,
        "updated_at": _now(),
    }
    write_json(str(output_dir / "latest.json"), report)
    return report


def default_primary_sources() -> list[SourceCandidate]:
    configured = os.environ.get("SITE_PRIMARY_SOURCE_URLS", "").strip()
    if configured:
        return [
            SourceCandidate(f"primary-{index}", url.strip())
            for index, url in enumerate(configured.split(","), start=1)
            if url.strip()
        ]
    return [SourceCandidate("hltv-major", "https://www.hltv.org/major")]


def _attempt_candidate(
    candidate: SourceCandidate,
    output_dir: Path,
    fetcher: Callable[[str, dict[str, str]], str] | None,
    attempts: list[dict[str, Any]],
    start_date: str,
    end_date: str | None,
) -> list[dict[str, Any]]:
    try:
        rows = _fetch_candidate_rows(candidate, output_dir, fetcher, start_date, end_date)
        completed = _deduplicate_completed(rows)
        attempts.append({"source": candidate.name, "status": "success", "rows": len(completed)})
        return completed
    except Exception as exc:
        attempts.append({"source": candidate.name, "status": "error", "error": str(exc)})
        return []


def _fetch_candidate_rows(
    candidate: SourceCandidate,
    output_dir: Path,
    fetcher: Callable[[str, dict[str, str]], str] | None,
    start_date: str,
    end_date: str | None,
) -> list[dict[str, Any]]:
    if candidate.kind == "fivee-results":
        fivee_dir = output_dir / "fivee"
        report = collect_fivee_match_results(
            cache_dir=str(output_dir / "fivee-cache"),
            output_dir=str(fivee_dir),
            start_date=start_date,
            end_date=end_date or date.today().isoformat(),
            refresh=True,
            delay_seconds=0.0,
            max_pages=3,
            fetcher=fetcher,
        )
        return _read_csv_dicts(Path(report["matches_path"]))

    text = _read_source_text(candidate.url, output_dir, fetcher)
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        payload = json.loads(text)
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        return [dict(row) for row in rows]
    return HltvResultParser().parse_results_html(text)


def _read_source_text(candidate_url: str, output_dir: Path, fetcher: Callable[[str, dict[str, str]], str] | None) -> str:
    if candidate_url.startswith("file://"):
        return Path(candidate_url[7:]).read_text(encoding="utf-8")
    return HttpCache(str(output_dir / "cache"), fetcher=fetcher).get(candidate_url, refresh=True)


def _deduplicate_completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        copied = dict(row)
        if str(copied.get("status", "completed")).lower() != "completed":
            continue
        team1 = str(copied.get("team1", "")).strip()
        team2 = str(copied.get("team2", "")).strip()
        winner = str(copied.get("winner", "")).strip()
        if not team1 or not team2 or winner not in {team1, team2}:
            continue
        copied["event"] = copied.get("event") or EVENT_NAME
        copied["best_of"] = copied.get("best_of") or 3
        copied["source"] = copied.get("source") or "public"
        ordered = tuple(sorted([team1.casefold(), team2.casefold()]))
        key = (str(copied.get("date", "")), ordered[0], ordered[1], str(copied.get("event", "")))
        by_key[key] = copied
    return list(by_key.values())


def _cached_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh completed match data for the static site.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data/cologne2026/site_updates"))
    parser.add_argument("--primary-source", action="append", default=[])
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--disable-primary", action="store_true")
    parser.add_argument("--disable-fivee", action="store_true")
    args = parser.parse_args()
    candidates = (
        []
        if args.disable_primary
        else [SourceCandidate(f"primary-{index}", url) for index, url in enumerate(args.primary_source, start=1)] or None
    )
    report = update_site_sources(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        source_candidates=candidates,
        fivee_candidate=None if args.disable_fivee else SourceCandidate("5E", "5e://match-results", "fivee-results"),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
