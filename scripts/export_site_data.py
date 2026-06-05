#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cs2pickem.data import read_matches_csv, write_json


EVENT_ID = "iem-cologne-2026"
PREDICTION_DIR = Path("data/cologne2026/predictions/fivee_6m_stage1_2026-06-01")
SOURCE_DIR = Path("data/cologne2026/source_inputs")
PROCESSED_DIR = Path("data/cologne2026/processed")
AUTO_UPDATE_MANIFEST = Path("data/cologne2026/site_updates/latest.json")


def export_site_data(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1 = _stage1_payload(repo_root)
    stage2 = _empty_swiss_stage("stage-2", "Stage 2 尚未开赛，等待 Stage 1 最终晋级队伍。")
    stage3 = _empty_playoff_stage("stage-3", "Stage 3 bracket 尚未生成，等待 Stage 2 完赛与淘汰赛抽签。")
    pickem = _pickem_payload(repo_root)
    source_status = _source_status_payload(repo_root)
    latest = {
        "event_id": EVENT_ID,
        "current_stage": "stage-1",
        "current_view": "swiss",
        "last_updated": source_status["generated_at"],
        "source_status": source_status["primary"]["status"],
        "ai_status": "pending",
        "data_version": "2026-06-05-r4",
    }
    event = {
        "event_id": EVENT_ID,
        "name": "IEM Cologne 2026",
        "game": "Counter-Strike 2",
        "stages": ["stage-1", "stage-2", "stage-3"],
        "current_stage": "stage-1",
    }

    payloads = {
        "latest.json": latest,
        "events/iem-cologne-2026.json": event,
        "stages/stage-1.json": stage1,
        "stages/stage-2.json": stage2,
        "stages/stage-3.json": stage3,
        "pickem/current.json": pickem,
        "system/source-status.json": source_status,
    }
    for relative_path, payload in payloads.items():
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(str(path), payload)
    return {
        "event_id": EVENT_ID,
        "current_stage": latest["current_stage"],
        "files_written": len(payloads),
        "output_dir": str(output_dir),
    }


def _stage1_payload(repo_root: Path) -> dict[str, Any]:
    paths = _site_input_paths(repo_root)
    standings = read_matches_csv(str(paths["standings"]))
    fixtures = read_matches_csv(str(paths["fixtures"]))
    results = read_matches_csv(str(paths["results"]))
    return {
        "stage_id": "stage-1",
        "name": "Stage 1",
        "format": "swiss",
        "status": "live",
        "teams": sorted({row["team"] for row in standings}),
        "standings": standings,
        "rounds": _rounds_from_results(results, fixtures),
        "fixtures": fixtures,
        "results": results,
        "pickem_impact": _pickem_payload(repo_root),
    }


def _site_input_paths(repo_root: Path) -> dict[str, Path]:
    paths = {
        "standings": repo_root / SOURCE_DIR / "stage1_round4_standings_2026-06-05.csv",
        "fixtures": repo_root / PROCESSED_DIR / "stage1_round5_fixtures_with_standings_2026-06-05.csv",
        "results": repo_root / SOURCE_DIR / "stage1_round1_4_results_2026-06-05.csv",
    }
    manifest_path = repo_root / AUTO_UPDATE_MANIFEST
    if not manifest_path.exists():
        return paths
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, manifest_key in (("results", "results_path"), ("standings", "standings_path")):
        candidate = repo_root / str(manifest.get(manifest_key, ""))
        if candidate.exists():
            paths[key] = candidate
    return paths


def _rounds_from_results(results: list[dict[str, Any]], fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounds: dict[str, dict[str, Any]] = {}
    for row in results:
        key = str(row.get("round", "unknown"))
        rounds.setdefault(key, {"round": key, "results": [], "fixtures": []})
        rounds[key]["results"].append(row)
    for row in fixtures:
        key = str(row.get("swiss_round", "unknown"))
        rounds.setdefault(key, {"round": key, "results": [], "fixtures": []})
        rounds[key]["fixtures"].append(row)
    return [rounds[key] for key in sorted(rounds, key=lambda value: int(value) if value.isdigit() else 999)]


def _pickem_payload(repo_root: Path) -> dict[str, Any]:
    path = repo_root / PREDICTION_DIR / "final_fused_pickem_checkpoint_round4_2026-06-05.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    picks = data["picks"]
    return {
        "summary": data["summary"],
        "picks": picks,
        "locked_picks": [row["team"] for row in picks if row["status"] == "locked"],
        "alive_picks": [row["team"] for row in picks if row["status"] == "alive"],
        "broken_picks": [row["team"] for row in picks if row["status"] == "broken"],
        "source": str(path.relative_to(repo_root)),
    }


def _empty_swiss_stage(stage_id: str, message: str) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "format": "swiss",
        "status": "upcoming",
        "teams": [],
        "standings": [],
        "rounds": [],
        "fixtures": [],
        "results": [],
        "pickem_impact": {},
        "empty_state": {
            "title": "等待赛程生成",
            "message": message,
            "next_update": "每天 02:00 BJT 自动检查更新",
        },
    }


def _empty_playoff_stage(stage_id: str, message: str) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "format": "playoff",
        "status": "upcoming",
        "bracket": {"quarterfinals": [], "semifinals": [], "final": []},
        "champion_path": {},
        "empty_state": {
            "title": "等待淘汰赛签表",
            "message": message,
            "next_update": "每天 02:00 BJT 自动检查更新",
        },
    }


def _source_status_payload(repo_root: Path) -> dict[str, Any]:
    manifest_path = repo_root / AUTO_UPDATE_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "generated_at": manifest.get("updated_at", _now()),
            "primary": {
                "name": manifest.get("selected_source", "unknown"),
                "status": manifest.get("status", "unknown"),
            },
            "cross_checks": manifest.get("attempts", []),
            "fallback": {
                "name": "5E",
                "status": "used" if manifest.get("status") == "fallback_success" else "not_used",
            },
            "visible_status": _visible_source_status(str(manifest.get("status", "unknown"))),
        }
    return {
        "generated_at": _now(),
        "primary": {"name": "repository-source-inputs", "status": "primary_success"},
        "cross_checks": [{"name": "existing-reviewed-csv", "status": "success"}],
        "fallback": {"name": "5E", "status": "not_used"},
        "visible_status": "主来源已更新",
    }


def _visible_source_status(status: str) -> str:
    labels = {
        "primary_success": "主来源已更新",
        "fallback_success": "主来源失败，已使用 5E fallback",
        "cached": "数据源失败，保留上一次有效数据",
        "failed": "数据源更新失败",
    }
    return labels.get(status, "数据状态未知")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static site JSON for GitHub Pages.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("site/data"))
    args = parser.parse_args()
    report = export_site_data(args.repo_root, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
