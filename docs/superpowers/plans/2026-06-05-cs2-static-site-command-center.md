# CS2 Static Site Command Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first GitHub Pages compatible static command center for CS2 Major stage prediction, AI analysis, and daily data refresh.

**Architecture:** Keep runtime frontend static: `site/` contains HTML, CSS, browser JavaScript, and generated JSON. Python scripts convert existing project data into `site/data/*.json` and generate AI articles during GitHub Actions; browser JavaScript handles Swiss and playoff user simulations without a backend.

**Tech Stack:** Python standard library, existing `cs2pickem` package, vanilla HTML/CSS/ES modules, Node built-in test runner for pure JavaScript logic, GitHub Actions, GitHub Pages.

---

## File Structure

Create these files:

- `scripts/export_site_data.py`: Converts existing Cologne data files into the static site JSON contract.
- `scripts/update_site_data.py`: Fetches completed match data from public sources, falls back to 5E, and preserves the last valid update manifest on failure.
- `scripts/generate_ai_articles.py`: Generates AI Desk articles from site JSON using an OpenAI-compatible API when `AI_API_KEY` exists; otherwise writes deterministic fallback articles.
- `tests/test_site_export.py`: Python unit tests for static data export.
- `tests/test_site_update.py`: Python unit tests for source priority, fallback/cached behavior, and exporter source selection.
- `tests/test_ai_articles.py`: Python unit tests for article fallback and API request construction.
- `site/index.html`: Single-page static app shell.
- `site/styles.css`: Dark command-center visual system.
- `site/src/data.js`: JSON loading helpers and static route helpers.
- `site/src/swiss.js`: Browser Swiss snapshot simulation logic.
- `site/src/pickem.js`: Browser Pick'em state classification and summaries.
- `site/src/bracket.js`: Browser playoff bracket simulation logic.
- `site/src/render.js`: DOM rendering helpers for stage switcher, status bar, predictor, and AI Desk.
- `site/src/main.js`: App bootstrap, state wiring, and hash routing.
- `site/tests/swiss.test.mjs`: Node tests for Swiss simulation.
- `site/tests/pickem.test.mjs`: Node tests for Pick'em impact logic.
- `site/tests/bracket.test.mjs`: Node tests for playoff simulation.
- `site/data/.gitkeep`: Keeps generated data directory present before export.
- `.github/workflows/pages.yml`: Daily data generation and GitHub Pages deployment workflow.

Modify these files:

- `.github/workflows/ci.yml`: Add Node logic tests and static exporter tests to CI.
- `README.md`: Add a short website/deployment section after implementation is complete.
- `docs/data-processing.md`: Link the new static site export commands after implementation is complete.

Do not move or rewrite existing model modules. Data generation should import stable helpers from `src/cs2pickem/data.py` and `src/cs2pickem/backtest.py`.

## Task 1: Static Site Data Exporter

**Files:**

- Create: `scripts/export_site_data.py`
- Create: `tests/test_site_export.py`
- Create: `site/data/.gitkeep`

- [ ] **Step 1: Write failing exporter tests**

Create `tests/test_site_export.py`:

```python
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SiteExportTests(unittest.TestCase):
    def test_export_site_data_writes_static_contract(self):
        from scripts.export_site_data import export_site_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = export_site_data(ROOT, Path(tmpdir))

            self.assertEqual(report["event_id"], "iem-cologne-2026")
            self.assertEqual(report["current_stage"], "stage-1")
            self.assertEqual(report["files_written"] >= 7, True)

            latest = json.loads((Path(tmpdir) / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["event_id"], "iem-cologne-2026")
            self.assertEqual(latest["current_view"], "swiss")
            self.assertIn(latest["source_status"], {"primary_success", "fallback_success", "cached"})

            stage1 = json.loads((Path(tmpdir) / "stages" / "stage-1.json").read_text(encoding="utf-8"))
            self.assertEqual(stage1["format"], "swiss")
            self.assertEqual(stage1["status"], "live")
            self.assertGreaterEqual(len(stage1["standings"]), 16)
            self.assertGreaterEqual(len(stage1["fixtures"]), 1)

            stage2 = json.loads((Path(tmpdir) / "stages" / "stage-2.json").read_text(encoding="utf-8"))
            self.assertEqual(stage2["format"], "swiss")
            self.assertEqual(stage2["status"], "upcoming")
            self.assertIn("empty_state", stage2)

            stage3 = json.loads((Path(tmpdir) / "stages" / "stage-3.json").read_text(encoding="utf-8"))
            self.assertEqual(stage3["format"], "playoff")
            self.assertEqual(stage3["status"], "upcoming")
            self.assertIn("bracket", stage3)

    def test_export_site_data_preserves_pickem_summary(self):
        from scripts.export_site_data import export_site_data

        with tempfile.TemporaryDirectory() as tmpdir:
            export_site_data(ROOT, Path(tmpdir))
            pickem = json.loads((Path(tmpdir) / "pickem" / "current.json").read_text(encoding="utf-8"))

            self.assertEqual(pickem["summary"], {"alive": 2, "broken": 4, "locked": 4, "missing": 0})
            self.assertEqual(sorted(pickem["alive_picks"]), ["BIG", "TYLOO"])
            self.assertIn("Gaimin Gladiators", pickem["locked_picks"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_site_export -v
```

Expected: fail with `ModuleNotFoundError: No module named 'scripts.export_site_data'`.

- [ ] **Step 3: Implement exporter**

Create `scripts/export_site_data.py`:

```python
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


def export_site_data(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1 = _stage1_payload(repo_root)
    stage2 = _empty_swiss_stage("stage-2", "Stage 2 尚未开赛，等待 Stage 1 最终晋级队伍。")
    stage3 = _empty_playoff_stage("stage-3", "Stage 3 bracket 尚未生成，等待 Stage 2 完赛与淘汰赛抽签。")
    pickem = _pickem_payload(repo_root)
    source_status = _source_status_payload()
    latest = {
        "event_id": EVENT_ID,
        "current_stage": "stage-1",
        "current_view": "swiss",
        "last_updated": source_status["generated_at"],
        "source_status": "primary_success",
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
    standings = read_matches_csv(str(repo_root / SOURCE_DIR / "stage1_round4_standings_2026-06-05.csv"))
    fixtures = read_matches_csv(str(repo_root / PROCESSED_DIR / "stage1_round5_fixtures_with_standings_2026-06-05.csv"))
    results = read_matches_csv(str(repo_root / SOURCE_DIR / "stage1_round1_4_results_2026-06-05.csv"))
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


def _source_status_payload() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "primary": {"name": "repository-source-inputs", "status": "success"},
        "cross_checks": [{"name": "existing-reviewed-csv", "status": "success"}],
        "fallback": {"name": "5E", "status": "not_used"},
        "visible_status": "主来源已更新",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static site JSON for GitHub Pages.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("site/data"))
    args = parser.parse_args()
    report = export_site_data(args.repo_root, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add data directory marker**

Create `site/data/.gitkeep` as an empty file.

- [ ] **Step 5: Run exporter tests and verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_site_export -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Generate local site data**

Run:

```bash
PYTHONPATH=src python3 scripts/export_site_data.py --repo-root . --output-dir site/data
```

Expected: JSON report with `"files_written": 7` or higher.

- [ ] **Step 7: Commit**

Run:

```bash
git add scripts/export_site_data.py tests/test_site_export.py site/data
git commit -m "feat: export static site data"
```

## Task 2: Scheduled Source Refresh Orchestrator

**Files:**

- Create: `scripts/update_site_data.py`
- Create: `tests/test_site_update.py`
- Modify: `scripts/export_site_data.py`

- [ ] **Step 1: Write failing source update tests**

Create `tests/test_site_update.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path


class SiteUpdateTests(unittest.TestCase):
    def test_update_uses_primary_source_and_writes_manifest(self):
        from scripts.update_site_data import SourceCandidate, update_site_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            output_dir = repo_root / "data" / "cologne2026" / "site_updates"

            def fetcher(url, headers):
                self.assertEqual(url, "https://primary.example/results")
                return json.dumps(
                    {
                        "results": [
                            {
                                "date": "2026-06-05",
                                "event": "IEM Cologne 2026",
                                "status": "completed",
                                "team1": "BIG",
                                "team2": "NRG",
                                "winner": "BIG",
                                "best_of": 3,
                                "source_match_url": url,
                            },
                            {
                                "date": "2026-06-05",
                                "event": "IEM Cologne 2026",
                                "status": "completed",
                                "team1": "TYLOO",
                                "team2": "Lynn Vision",
                                "winner": "TYLOO",
                                "best_of": 3,
                                "source_match_url": url,
                            },
                        ]
                    }
                )

            report = update_site_sources(
                repo_root=repo_root,
                output_dir=output_dir,
                source_candidates=[SourceCandidate("primary-test", "https://primary.example/results")],
                fivee_candidate=None,
                fetcher=fetcher,
            )

            self.assertEqual(report["status"], "primary_success")
            self.assertEqual(report["selected_source"], "primary-test")
            self.assertEqual(report["completed_results"], 2)
            self.assertTrue((output_dir / "latest.json").exists())
            self.assertTrue((output_dir / "auto_results.csv").exists())
            self.assertTrue((output_dir / "auto_standings.csv").exists())

    def test_update_returns_cached_manifest_when_sources_fail(self):
        from scripts.update_site_data import SourceCandidate, update_site_sources

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            output_dir = repo_root / "data" / "cologne2026" / "site_updates"
            output_dir.mkdir(parents=True)
            cached = {
                "status": "primary_success",
                "selected_source": "previous-valid",
                "completed_results": 1,
                "results_path": "data/cologne2026/site_updates/auto_results.csv",
                "standings_path": "data/cologne2026/site_updates/auto_standings.csv",
            }
            (output_dir / "latest.json").write_text(json.dumps(cached), encoding="utf-8")

            def failing_fetcher(url, headers):
                raise RuntimeError("source down")

            report = update_site_sources(
                repo_root=repo_root,
                output_dir=output_dir,
                source_candidates=[SourceCandidate("primary-test", "https://primary.example/results")],
                fivee_candidate=None,
                fetcher=failing_fetcher,
            )

            self.assertEqual(report["status"], "cached")
            self.assertEqual(report["selected_source"], "previous-valid")
            self.assertEqual(report["completed_results"], 1)
            self.assertEqual(report["attempts"][0]["status"], "error")

    def test_exporter_prefers_auto_update_manifest_paths(self):
        from scripts.export_site_data import _site_input_paths

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            update_dir = repo_root / "data" / "cologne2026" / "site_updates"
            update_dir.mkdir(parents=True)
            (update_dir / "auto_results.csv").write_text("team1,team2,winner\nBIG,NRG,BIG\n", encoding="utf-8")
            (update_dir / "auto_standings.csv").write_text("team,wins,losses,status\nBIG,3,2,advanced\n", encoding="utf-8")
            (update_dir / "latest.json").write_text(
                json.dumps(
                    {
                        "results_path": "data/cologne2026/site_updates/auto_results.csv",
                        "standings_path": "data/cologne2026/site_updates/auto_standings.csv",
                    }
                ),
                encoding="utf-8",
            )

            paths = _site_input_paths(repo_root)

            self.assertEqual(paths["results"].name, "auto_results.csv")
            self.assertEqual(paths["standings"].name, "auto_standings.csv")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_site_update -v
```

Expected: fail with `ModuleNotFoundError: No module named 'scripts.update_site_data'`.

- [ ] **Step 3: Implement source update script**

Create `scripts/update_site_data.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
        return [SourceCandidate(f"primary-{index}", url.strip()) for index, url in enumerate(configured.split(","), start=1) if url.strip()]
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
    import csv

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
    candidates = [] if args.disable_primary else [SourceCandidate(f"primary-{index}", url) for index, url in enumerate(args.primary_source, start=1)] or None
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
```

- [ ] **Step 4: Modify exporter to prefer auto update inputs**

In `scripts/export_site_data.py`, add this constant near the existing path constants:

```python
AUTO_UPDATE_MANIFEST = Path("data/cologne2026/site_updates/latest.json")
```

Replace `_stage1_payload` with:

```python
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
```

Add this helper after `_stage1_payload`:

```python
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
```

In `export_site_data`, change the source status call from:

```python
    source_status = _source_status_payload()
```

to:

```python
    source_status = _source_status_payload(repo_root)
```

Replace `_source_status_payload` with:

```python
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
        "primary": {"name": "repository-source-inputs", "status": "success"},
        "cross_checks": [{"name": "existing-reviewed-csv", "status": "success"}],
        "fallback": {"name": "5E", "status": "not_used"},
        "visible_status": "主来源已更新",
    }
```

Add these helpers below `_source_status_payload`:

```python
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
```

- [ ] **Step 5: Run source update tests and exporter tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_site_update tests.test_site_export -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Run a local update with 5E disabled**

Run:

```bash
PYTHONPATH=src python3 scripts/update_site_data.py --repo-root . --output-dir data/cologne2026/site_updates --disable-primary --disable-fivee
```

Expected: report status is either `cached` or `failed`; no network source, API key, or authorization header is printed.

- [ ] **Step 7: Commit**

Run:

```bash
git add scripts/update_site_data.py scripts/export_site_data.py tests/test_site_update.py
git commit -m "feat: refresh static site match sources"
```

## Task 3: AI Desk Article Generator

**Files:**

- Create: `scripts/generate_ai_articles.py`
- Create: `tests/test_ai_articles.py`

- [ ] **Step 1: Write failing AI article tests**

Create `tests/test_ai_articles.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path


class AiArticleTests(unittest.TestCase):
    def test_template_fallback_generates_article_without_key(self):
        from scripts.generate_ai_articles import generate_ai_articles

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_fixture_data(data_dir)

            report = generate_ai_articles(data_dir=data_dir, output_dir=data_dir / "ai", api_key=None)

            self.assertTrue(report["fallback_used"])
            articles = json.loads((data_dir / "ai" / "articles.json").read_text(encoding="utf-8"))
            self.assertTrue(articles["fallback_used"])
            self.assertEqual(articles["articles"][0]["type"], "round_preview")
            self.assertIn("BIG", articles["articles"][0]["body"])

    def test_build_ai_request_uses_openai_compatible_shape(self):
        from scripts.generate_ai_articles import build_ai_request

        request = build_ai_request(
            model="gpt-5.5",
            data_summary={"alive_picks": ["BIG", "TYLOO"], "locked": 4},
        )

        self.assertEqual(request["model"], "gpt-5.5")
        self.assertEqual(request["temperature"], 0.4)
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(request["messages"][1]["role"], "user")
        self.assertIn("BIG", request["messages"][1]["content"])


def _write_fixture_data(data_dir: Path) -> None:
    (data_dir / "pickem").mkdir(parents=True)
    (data_dir / "system").mkdir(parents=True)
    (data_dir / "pickem" / "current.json").write_text(
        json.dumps(
            {
                "summary": {"locked": 4, "alive": 2, "broken": 4, "missing": 0},
                "locked_picks": ["BetBoom", "B8", "M80", "Gaimin Gladiators"],
                "alive_picks": ["BIG", "TYLOO"],
                "broken_picks": ["MIBR", "GamerLegion", "HEROIC", "NRG"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (data_dir / "latest.json").write_text(
        json.dumps({"event_id": "iem-cologne-2026", "current_stage": "stage-1", "data_version": "fixture"}),
        encoding="utf-8",
    )
    (data_dir / "system" / "source-status.json").write_text(
        json.dumps({"visible_status": "主来源已更新"}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_ai_articles -v
```

Expected: fail with `ModuleNotFoundError: No module named 'scripts.generate_ai_articles'`.

- [ ] **Step 3: Implement AI generator**

Create `scripts/generate_ai_articles.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cs2pickem.data import write_json


DEFAULT_BASE_URL = "https://zhengdatech.com/openai/v1"
DEFAULT_MODEL = "gpt-5.5"


def generate_ai_articles(
    data_dir: Path,
    output_dir: Path,
    api_key: str | None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(data_dir)
    fallback_used = True
    articles = template_articles(summary)
    if api_key:
        try:
            generated = call_ai_articles(base_url=base_url, model=model, api_key=api_key, data_summary=summary)
            if generated:
                articles = generated
                fallback_used = False
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
            fallback_used = True
            articles = template_articles(summary)
    payload = {
        "generated_at": _now(),
        "model": model if not fallback_used else "template-fallback",
        "fallback_used": fallback_used,
        "source_data_version": summary["data_version"],
        "articles": articles,
    }
    headlines = {
        "generated_at": payload["generated_at"],
        "fallback_used": fallback_used,
        "items": [
            {
                "id": article["id"],
                "title": article["title"],
                "summary": article["summary"],
                "stage": article["stage"],
                "type": article["type"],
            }
            for article in articles
        ],
    }
    write_json(str(output_dir / "articles.json"), payload)
    write_json(str(output_dir / "headlines.json"), headlines)
    return {"articles": len(articles), "fallback_used": fallback_used, "output_dir": str(output_dir)}


def build_ai_request(model: str, data_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {
                "role": "system",
                "content": "你是 CS2 电竞数据编辑。只根据用户提供的数据写简短中文分析，不夸大投注价值。",
            },
            {
                "role": "user",
                "content": json.dumps(data_summary, ensure_ascii=False, sort_keys=True),
            },
        ],
    }


def call_ai_articles(base_url: str, model: str, api_key: str, data_summary: dict[str, Any]) -> list[dict[str, Any]]:
    request_payload = build_ai_request(model=model, data_summary=data_summary)
    url = base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    articles = parsed["articles"] if isinstance(parsed, dict) else parsed
    if not isinstance(articles, list):
        raise ValueError("AI response must contain an articles list")
    return [_normalize_article(article, data_summary) for article in articles]


def template_articles(data_summary: dict[str, Any]) -> list[dict[str, Any]]:
    alive = data_summary["alive_picks"]
    locked = data_summary["summary"]["locked"]
    broken = data_summary["summary"]["broken"]
    alive_text = "、".join(alive) if alive else "没有仍可兑现的 Pick'em 槽位"
    return [
        {
            "id": "template-current-stage-watch",
            "stage": data_summary["current_stage"],
            "type": "round_preview",
            "title": alive_text + " 是当前补分重点",
            "summary": f"当前 Pick'em 状态为 {locked} locked / {len(alive)} alive / {broken} broken。",
            "body": f"当前仍可变化的 Pick'em 槽位集中在 {alive_text}。如果这些队伍赢下后续关键比赛，advance 槽位会继续补成 locked；如果失利，对应槽位会变成 broken。该结论来自最新静态 standings 和 checkpoint 数据。",
            "source_data_version": data_summary["data_version"],
        }
    ]


def _normalize_article(article: dict[str, Any], data_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(article.get("id") or "ai-" + data_summary["data_version"]),
        "stage": str(article.get("stage") or data_summary["current_stage"]),
        "type": str(article.get("type") or "round_preview"),
        "title": str(article.get("title") or "当前阶段 AI 分析"),
        "summary": str(article.get("summary") or "基于最新赛程、赛果和 Pick'em 状态生成。"),
        "body": str(article.get("body") or article.get("summary") or "暂无正文。"),
        "source_data_version": str(article.get("source_data_version") or data_summary["data_version"]),
    }


def _load_summary(data_dir: Path) -> dict[str, Any]:
    latest = json.loads((data_dir / "latest.json").read_text(encoding="utf-8"))
    pickem = json.loads((data_dir / "pickem" / "current.json").read_text(encoding="utf-8"))
    source_status = json.loads((data_dir / "system" / "source-status.json").read_text(encoding="utf-8"))
    return {
        "event_id": latest["event_id"],
        "current_stage": latest["current_stage"],
        "data_version": latest["data_version"],
        "source_status": source_status.get("visible_status", "数据状态未知"),
        "summary": pickem["summary"],
        "locked_picks": pickem["locked_picks"],
        "alive_picks": pickem["alive_picks"],
        "broken_picks": pickem["broken_picks"],
    }


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static AI Desk article JSON.")
    parser.add_argument("--data-dir", type=Path, default=Path("site/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("site/data/ai"))
    parser.add_argument("--base-url", default=os.environ.get("AI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("AI_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()
    report = generate_ai_articles(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        api_key=os.environ.get("AI_API_KEY"),
        base_url=args.base_url,
        model=args.model,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run AI tests and verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_ai_articles -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Generate fallback articles locally**

Run:

```bash
PYTHONPATH=src python3 scripts/generate_ai_articles.py --data-dir site/data --output-dir site/data/ai
```

Expected: JSON report with `"fallback_used": true` when `AI_API_KEY` is unset.

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/generate_ai_articles.py tests/test_ai_articles.py site/data/ai
git commit -m "feat: generate static AI desk articles"
```

## Task 4: Swiss Simulation Logic

**Files:**

- Create: `site/src/swiss.js`
- Create: `site/tests/swiss.test.mjs`

- [ ] **Step 1: Write failing Swiss tests**

Create `site/tests/swiss.test.mjs`:

```javascript
import assert from "node:assert/strict";
import test from "node:test";
import { applySwissWinner, resetSwissState, undoSwiss } from "../src/swiss.js";

const standings = [
  { team: "BIG", wins: 2, losses: 2, status: "alive" },
  { team: "NRG", wins: 2, losses: 2, status: "alive" },
  { team: "M80", wins: 3, losses: 1, status: "advanced" }
];

test("applySwissWinner advances winner and eliminates loser at 2-2", () => {
  const state = resetSwissState(standings);
  const next = applySwissWinner(state, { team1: "NRG", team2: "BIG" }, "BIG");

  assert.equal(next.records.BIG.wins, 3);
  assert.equal(next.records.BIG.status, "advanced");
  assert.equal(next.records.NRG.losses, 3);
  assert.equal(next.records.NRG.status, "eliminated");
  assert.equal(next.history.length, 1);
});

test("applySwissWinner does not mutate previous state", () => {
  const state = resetSwissState(standings);
  const next = applySwissWinner(state, { team1: "NRG", team2: "BIG" }, "BIG");

  assert.equal(state.records.BIG.wins, 2);
  assert.equal(next.records.BIG.wins, 3);
});

test("undoSwiss removes only the latest simulated result", () => {
  const state = resetSwissState(standings);
  const first = applySwissWinner(state, { team1: "NRG", team2: "BIG" }, "BIG");
  const second = applySwissWinner(first, { team1: "M80", team2: "BIG" }, "M80");
  const undone = undoSwiss(second);

  assert.equal(undone.history.length, 1);
  assert.equal(undone.records.BIG.wins, 3);
  assert.equal(undone.records.NRG.status, "eliminated");
  assert.equal(undone.records.M80.wins, 3);
});
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
node --test site/tests/swiss.test.mjs
```

Expected: fail because `site/src/swiss.js` does not exist.

- [ ] **Step 3: Implement Swiss logic**

Create `site/src/swiss.js`:

```javascript
export function resetSwissState(standings) {
  const records = recordsFromStandings(standings);
  return {
    initialRecords: cloneRecords(records),
    records,
    history: []
  };
}

export function applySwissWinner(state, fixture, winner) {
  if (winner !== fixture.team1 && winner !== fixture.team2) {
    throw new Error("winner must be one of the fixture teams");
  }
  const loser = winner === fixture.team1 ? fixture.team2 : fixture.team1;
  const records = cloneRecords(state.records);
  records[winner] = bump(records[winner], 1, 0);
  records[loser] = bump(records[loser], 0, 1);
  return {
    initialRecords: cloneRecords(state.initialRecords || state.records),
    records,
    history: state.history.concat([{ fixture, winner, loser }])
  };
}

export function undoSwiss(state) {
  if (state.history.length === 0) {
    return state;
  }
  const history = state.history.slice(0, -1);
  let replay = {
    initialRecords: cloneRecords(state.initialRecords),
    records: cloneRecords(state.initialRecords),
    history: []
  };
  for (const entry of history) {
    replay = applySwissWinner(replay, entry.fixture, entry.winner);
  }
  return replay;
}

export function recordStatus(wins, losses) {
  if (wins >= 3) return "advanced";
  if (losses >= 3) return "eliminated";
  return "alive";
}

function bump(record, winDelta, lossDelta) {
  if (!record) {
    throw new Error("record missing for team");
  }
  const wins = Number(record.wins) + winDelta;
  const losses = Number(record.losses) + lossDelta;
  return { ...record, wins, losses, status: recordStatus(wins, losses) };
}

function cloneRecords(records) {
  const cloned = {};
  for (const [team, record] of Object.entries(records)) {
    cloned[team] = { ...record };
  }
  return cloned;
}

function recordsFromStandings(rows) {
  const records = {};
  for (const row of rows) {
    const wins = Number(row.wins || 0);
    const losses = Number(row.losses || 0);
    records[row.team] = {
      team: row.team,
      wins,
      losses,
      status: row.status || recordStatus(wins, losses)
    };
  }
  return records;
}
```

- [ ] **Step 4: Run Swiss tests and verify they pass**

Run:

```bash
node --test site/tests/swiss.test.mjs
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add site/src/swiss.js site/tests/swiss.test.mjs
git commit -m "feat: add browser Swiss simulation logic"
```

## Task 5: Pick'em Impact Logic

**Files:**

- Create: `site/src/pickem.js`
- Create: `site/tests/pickem.test.mjs`

- [ ] **Step 1: Write failing Pick'em tests**

Create `site/tests/pickem.test.mjs`:

```javascript
import assert from "node:assert/strict";
import test from "node:test";
import { classifyPickem, summarizePickem } from "../src/pickem.js";

const pickems = {
  picks: [
    { category: "advance", team: "BIG" },
    { category: "advance", team: "TYLOO" },
    { category: "0-3", team: "Gaimin Gladiators" },
    { category: "3-0", team: "MIBR" }
  ]
};

test("classifyPickem tracks locked alive and broken states", () => {
  const records = {
    BIG: { wins: 3, losses: 2, status: "advanced" },
    TYLOO: { wins: 2, losses: 2, status: "alive" },
    "Gaimin Gladiators": { wins: 0, losses: 3, status: "eliminated" },
    MIBR: { wins: 3, losses: 1, status: "advanced" }
  };

  const rows = classifyPickem(pickems, records);
  assert.equal(rows.find((row) => row.team === "BIG").status, "locked");
  assert.equal(rows.find((row) => row.team === "TYLOO").status, "alive");
  assert.equal(rows.find((row) => row.team === "Gaimin Gladiators").status, "locked");
  assert.equal(rows.find((row) => row.team === "MIBR").status, "broken");
});

test("summarizePickem counts statuses", () => {
  const rows = [
    { status: "locked" },
    { status: "alive" },
    { status: "broken" },
    { status: "locked" }
  ];

  assert.deepEqual(summarizePickem(rows), { locked: 2, alive: 1, broken: 1, missing: 0 });
});
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
node --test site/tests/pickem.test.mjs
```

Expected: fail because `site/src/pickem.js` does not exist.

- [ ] **Step 3: Implement Pick'em logic**

Create `site/src/pickem.js`:

```javascript
export function classifyPickem(pickemPayload, records) {
  return pickemPayload.picks.map((pick) => {
    const record = records[pick.team];
    return {
      ...pick,
      wins: record ? Number(record.wins) : null,
      losses: record ? Number(record.losses) : null,
      status: pickStatus(pick.category, record)
    };
  });
}

export function summarizePickem(rows) {
  const summary = { locked: 0, alive: 0, broken: 0, missing: 0 };
  for (const row of rows) {
    summary[row.status] = (summary[row.status] || 0) + 1;
  }
  return summary;
}

export function pickStatus(category, record) {
  if (!record) return "missing";
  const wins = Number(record.wins);
  const losses = Number(record.losses);
  if (category === "3-0") {
    if (wins >= 3 && losses === 0) return "locked";
    if (losses > 0 || (wins >= 3 && losses > 0)) return "broken";
    return "alive";
  }
  if (category === "advance") {
    if (wins >= 3) return "locked";
    if (losses >= 3) return "broken";
    return "alive";
  }
  if (category === "0-3") {
    if (losses >= 3 && wins === 0) return "locked";
    if (wins > 0 || (losses >= 3 && wins > 0)) return "broken";
    return "alive";
  }
  return "missing";
}
```

- [ ] **Step 4: Run Pick'em tests and verify they pass**

Run:

```bash
node --test site/tests/pickem.test.mjs
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add site/src/pickem.js site/tests/pickem.test.mjs
git commit -m "feat: add browser pickem impact logic"
```

## Task 6: Playoff Bracket Logic

**Files:**

- Create: `site/src/bracket.js`
- Create: `site/tests/bracket.test.mjs`

- [ ] **Step 1: Write failing bracket tests**

Create `site/tests/bracket.test.mjs`:

```javascript
import assert from "node:assert/strict";
import test from "node:test";
import { applyBracketWinner, emptyBracketState, resetBracket, undoBracket } from "../src/bracket.js";

const bracket = {
  quarterfinals: [
    { id: "qf-1", team1: "Alpha", team2: "Bravo", nextMatchId: "sf-1", nextSlot: "team1" }
  ],
  semifinals: [{ id: "sf-1", team1: null, team2: "Charlie", nextMatchId: "final", nextSlot: "team1" }],
  final: [{ id: "final", team1: null, team2: null, nextMatchId: null, nextSlot: null }]
};

test("applyBracketWinner advances quarterfinal winner to semifinal slot", () => {
  const state = emptyBracketState(bracket);

  const next = applyBracketWinner(state, "qf-1", "Alpha");
  assert.equal(next.matches["qf-1"].winner, "Alpha");
  assert.equal(next.matches["sf-1"].team1, "Alpha");
});

test("applyBracketWinner sets champion when final winner is chosen", () => {
  const state = emptyBracketState({
    quarterfinals: [],
    semifinals: [],
    final: [{ id: "final", team1: "Alpha", team2: "Charlie", nextMatchId: null, nextSlot: null }]
  });

  const next = applyBracketWinner(state, "final", "Charlie");
  assert.equal(next.champion, "Charlie");
});

test("undoBracket removes only the latest simulated winner", () => {
  const first = applyBracketWinner(emptyBracketState(bracket), "qf-1", "Alpha");
  const second = applyBracketWinner(first, "sf-1", "Alpha");
  const undone = undoBracket(second);

  assert.equal(undone.history.length, 1);
  assert.equal(undone.matches["qf-1"].winner, "Alpha");
  assert.equal(undone.matches["sf-1"].winner, null);
  assert.equal(undone.matches["final"].team1, null);
});

test("resetBracket clears all simulated winners", () => {
  const selected = applyBracketWinner(emptyBracketState(bracket), "qf-1", "Alpha");
  const reset = resetBracket(selected.originalBracket);

  assert.equal(reset.history.length, 0);
  assert.equal(reset.matches["qf-1"].winner, null);
  assert.equal(reset.matches["sf-1"].team1, null);
});
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
node --test site/tests/bracket.test.mjs
```

Expected: fail because `site/src/bracket.js` does not exist.

- [ ] **Step 3: Implement bracket logic**

Create `site/src/bracket.js`:

```javascript
export function emptyBracketState(bracket) {
  const originalBracket = cloneBracket(bracket);
  const matches = {};
  for (const round of ["quarterfinals", "semifinals", "final"]) {
    for (const match of bracket[round] || []) {
      matches[match.id] = { ...match, round, winner: match.winner || null };
    }
  }
  return { originalBracket, matches, champion: null, history: [] };
}

export function applyBracketWinner(state, matchId, winner) {
  const matches = cloneMatches(state.matches);
  const match = matches[matchId];
  if (!match) {
    throw new Error("match not found");
  }
  if (winner !== match.team1 && winner !== match.team2) {
    throw new Error("winner must be one of the match teams");
  }
  match.winner = winner;
  let champion = state.champion;
  if (match.nextMatchId) {
    matches[match.nextMatchId] = {
      ...matches[match.nextMatchId],
      [match.nextSlot]: winner
    };
  } else {
    champion = winner;
  }
  return {
    originalBracket: cloneBracket(state.originalBracket),
    matches,
    champion,
    history: state.history.concat([{ matchId, winner }])
  };
}

export function undoBracket(state) {
  if (state.history.length === 0) {
    return state;
  }
  const history = state.history.slice(0, -1);
  let replay = emptyBracketState(state.originalBracket);
  for (const entry of history) {
    replay = applyBracketWinner(replay, entry.matchId, entry.winner);
  }
  return replay;
}

export function resetBracket(bracket) {
  return emptyBracketState(bracket);
}

function cloneBracket(bracket) {
  const cloned = {};
  for (const round of ["quarterfinals", "semifinals", "final"]) {
    cloned[round] = (bracket[round] || []).map((match) => ({ ...match }));
  }
  return cloned;
}

function cloneMatches(matches) {
  const cloned = {};
  for (const [id, match] of Object.entries(matches)) {
    cloned[id] = { ...match };
  }
  return cloned;
}
```

- [ ] **Step 4: Run bracket tests and verify they pass**

Run:

```bash
node --test site/tests/bracket.test.mjs
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add site/src/bracket.js site/tests/bracket.test.mjs
git commit -m "feat: add browser bracket simulation logic"
```

## Task 7: Static App Shell and Rendering

**Files:**

- Create: `site/index.html`
- Create: `site/styles.css`
- Create: `site/src/data.js`
- Create: `site/src/render.js`
- Create: `site/src/main.js`

- [ ] **Step 1: Create app shell**

Create `site/index.html`:

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CS2 Major Intel</title>
  <link rel="stylesheet" href="./styles.css">
</head>
<body>
  <a class="skip-link" href="#app">跳到主要内容</a>
  <header class="topbar">
    <div class="brand"><span class="brand-mark">AI</span> CS2 Major Intel</div>
    <nav aria-label="主导航">
      <a href="#/">Overview</a>
      <a href="#/stage/1">Stage 1</a>
      <a href="#/stage/2">Stage 2</a>
      <a href="#/stage/3">Stage 3</a>
      <a href="#/ai">AI Desk</a>
      <a href="#/model">Model Lab</a>
    </nav>
  </header>
  <main id="app" tabindex="-1">
    <section class="loading">Loading command center data...</section>
  </main>
  <script type="module" src="./src/main.js"></script>
</body>
</html>
```

- [ ] **Step 2: Add command-center styles**

Create `site/styles.css` with these tokens and core layout:

```css
:root {
  color-scheme: dark;
  --bg: #05070b;
  --surface: #0b101a;
  --surface-2: #101827;
  --line: rgba(169, 184, 211, 0.18);
  --text: #eef4ff;
  --muted: #93a3bb;
  --blue: #3b82f6;
  --gold: #d97706;
  --green: #22c55e;
  --red: #ef4444;
  --amber: #f59e0b;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at 14% 0%, rgba(59, 130, 246, 0.16), transparent 28rem), var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.skip-link {
  position: absolute;
  left: 12px;
  top: -48px;
  background: var(--gold);
  color: #120a02;
  padding: 10px 14px;
  z-index: 100;
}
.skip-link:focus { top: 12px; }

.topbar {
  min-height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 0 24px;
  border-bottom: 1px solid var(--line);
  background: rgba(5, 7, 11, 0.88);
  position: sticky;
  top: 0;
  z-index: 20;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 800;
}

.brand-mark {
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, var(--blue), var(--gold));
  color: #fff;
  font: 800 12px ui-monospace, SFMono-Regular, Menlo, monospace;
}

nav {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

nav a {
  color: var(--muted);
  text-decoration: none;
  font-size: 14px;
}

nav a:hover, nav a:focus { color: var(--text); }

#app {
  width: min(1280px, calc(100% - 32px));
  margin: 0 auto;
  padding: 24px 0 56px;
}

.command-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 18px;
  align-items: start;
}

.panel {
  border: 1px solid var(--line);
  background: rgba(11, 16, 26, 0.86);
}

.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 16px;
  border-bottom: 1px solid var(--line);
}

.muted { color: var(--muted); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.status-good { color: var(--green); }
.status-warn { color: var(--amber); }
.status-bad { color: var(--red); }

.team-row, .match-row, .article-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  padding: 12px 16px;
  border-top: 1px solid var(--line);
}

.winner-button {
  min-height: 40px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  padding: 0 12px;
  cursor: pointer;
}

.winner-button:hover, .winner-button:focus {
  border-color: var(--gold);
}

@media (max-width: 980px) {
  .command-grid { grid-template-columns: 1fr; }
  .topbar { align-items: flex-start; flex-direction: column; padding: 14px 16px; }
}
```

- [ ] **Step 3: Add data loader**

Create `site/src/data.js`:

```javascript
export async function loadSiteData(route = "#/") {
  const [latest, sourceStatus, pickem, articles] = await Promise.all([
    getJson("./data/latest.json"),
    getJson("./data/system/source-status.json"),
    getJson("./data/pickem/current.json"),
    getJson("./data/ai/articles.json").catch(() => ({ articles: [], fallback_used: true }))
  ]);
  const stage = await getJson(`./data/stages/${stageIdFromRoute(route, latest.current_stage)}.json`);
  return { latest, sourceStatus, pickem, articles, stage };
}

export async function loadStage(stageNumber) {
  return getJson(`./data/stages/stage-${stageNumber}.json`);
}

async function getJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

function stageIdFromRoute(route, currentStage) {
  const match = String(route || "").match(/^#\/stage\/([123])$/);
  if (match) {
    return `stage-${match[1]}`;
  }
  return currentStage;
}
```

- [ ] **Step 4: Add renderer**

Create `site/src/render.js`:

```javascript
export function renderApp(root, data, handlers) {
  root.innerHTML = `
    ${renderStatusBar(data)}
    <section class="command-grid">
      <div class="panel">
        ${renderStageHead(data.stage)}
        <div id="predictor"></div>
      </div>
      <aside class="panel">
        ${renderAiDesk(data)}
      </aside>
    </section>
  `;
  renderPredictor(document.querySelector("#predictor"), data.stage, handlers);
}

export function renderPredictor(root, stage, handlers) {
  if (stage.empty_state) {
    root.innerHTML = `<div class="panel-head"><div><h2>${escapeHtml(stage.empty_state.title)}</h2><p class="muted">${escapeHtml(stage.empty_state.message)}</p></div></div>`;
    return;
  }
  if (stage.format === "swiss") {
    root.innerHTML = `
      <div class="panel-head"><h2>Swiss Predictor</h2><span class="mono muted">${escapeHtml(stage.stage_id)}</span></div>
      ${stage.fixtures.map((fixture, index) => renderMatchRow(fixture, index)).join("")}
      <div id="standings">${stage.standings.map(renderStandingRow).join("")}</div>
    `;
    root.querySelectorAll("[data-winner]").forEach((button) => {
      button.addEventListener("click", () => handlers.onSwissWinner(Number(button.dataset.index), button.dataset.winner));
    });
    return;
  }
  if (stage.format === "playoff") {
    const bracket = stage.bracket || {};
    root.innerHTML = `
      <div class="panel-head"><h2>Playoff Bracket</h2><span class="mono muted">${escapeHtml(stage.stage_id)}</span></div>
      ${renderBracketRound("Quarterfinals", bracket.quarterfinals || [])}
      ${renderBracketRound("Semifinals", bracket.semifinals || [])}
      ${renderBracketRound("Final", bracket.final || [])}
    `;
    root.querySelectorAll("[data-bracket-winner]").forEach((button) => {
      button.addEventListener("click", () => handlers.onBracketWinner(button.dataset.matchId, button.dataset.bracketWinner));
    });
    return;
  }
  root.innerHTML = `<div class="panel-head"><h2>Stage data unavailable</h2><p class="muted">This stage format is not supported yet.</p></div>`;
}

export function renderStatusBar(data) {
  return `
    <section class="panel panel-head" aria-label="数据状态">
      <div>
        <strong>${escapeHtml(data.latest.event_id)}</strong>
        <div class="muted">Last updated: ${escapeHtml(data.latest.last_updated)}</div>
      </div>
      <div class="mono status-good">${escapeHtml(data.sourceStatus.visible_status || data.latest.source_status)}</div>
    </section>
  `;
}

export function renderAiDesk(data) {
  const articles = data.articles.articles || [];
  return `
    <div class="panel-head"><h2>AI Desk</h2><span class="mono muted">${articles.length} articles</span></div>
    ${articles.map((article) => `
      <article class="article-row">
        <div>
          <strong>${escapeHtml(article.title)}</strong>
          <p class="muted">${escapeHtml(article.summary)}</p>
        </div>
        <span class="mono muted">${escapeHtml(article.type)}</span>
      </article>
    `).join("")}
  `;
}

function renderStageHead(stage) {
  return `<div class="panel-head"><div><h1>${escapeHtml(stage.name || stage.stage_id)}</h1><p class="muted">${escapeHtml(stage.format)} · ${escapeHtml(stage.status)}</p></div></div>`;
}

function renderMatchRow(fixture, index) {
  return `
    <div class="match-row">
      <div><strong>${escapeHtml(fixture.team1)} vs ${escapeHtml(fixture.team2)}</strong><div class="muted">${escapeHtml(fixture.note || "")}</div></div>
      <div>
        <button class="winner-button" data-index="${index}" data-winner="${escapeHtml(fixture.team1)}">${escapeHtml(fixture.team1)}</button>
        <button class="winner-button" data-index="${index}" data-winner="${escapeHtml(fixture.team2)}">${escapeHtml(fixture.team2)}</button>
      </div>
    </div>
  `;
}

function renderBracketRound(label, matches) {
  return `
    <section class="bracket-round">
      <div class="panel-head"><h3>${escapeHtml(label)}</h3></div>
      ${matches.length ? matches.map(renderBracketMatch).join("") : `<div class="match-row"><span class="muted">Waiting for bracket draw.</span></div>`}
    </section>
  `;
}

function renderBracketMatch(match) {
  const team1 = match.team1 || "待定";
  const team2 = match.team2 || "待定";
  return `
    <div class="match-row">
      <div><strong>${escapeHtml(team1)} vs ${escapeHtml(team2)}</strong><div class="muted">${escapeHtml(match.winner ? `Winner: ${match.winner}` : match.id)}</div></div>
      <div>
        ${match.team1 ? `<button class="winner-button" data-match-id="${escapeHtml(match.id)}" data-bracket-winner="${escapeHtml(match.team1)}">${escapeHtml(match.team1)}</button>` : ""}
        ${match.team2 ? `<button class="winner-button" data-match-id="${escapeHtml(match.id)}" data-bracket-winner="${escapeHtml(match.team2)}">${escapeHtml(match.team2)}</button>` : ""}
      </div>
    </div>
  `;
}

function renderStandingRow(row) {
  return `<div class="team-row"><span>${escapeHtml(row.team)}</span><span class="mono">${row.wins}-${row.losses} ${escapeHtml(row.status)}</span></div>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}
```

- [ ] **Step 5: Add app bootstrap**

Create `site/src/main.js`:

```javascript
import { loadSiteData } from "./data.js";
import { resetSwissState, applySwissWinner } from "./swiss.js";
import { emptyBracketState, applyBracketWinner } from "./bracket.js";
import { renderApp } from "./render.js";

const root = document.querySelector("#app");
let appData = null;
let swissState = null;
let bracketState = null;

loadCurrentRoute();
window.addEventListener("hashchange", loadCurrentRoute);

function loadCurrentRoute() {
  loadSiteData(window.location.hash || "#/")
  .then((data) => {
    appData = data;
    swissState = null;
    bracketState = null;
    if (data.stage.format === "swiss" && !data.stage.empty_state) {
      swissState = resetSwissState(data.stage.standings);
    }
    if (data.stage.format === "playoff" && !data.stage.empty_state) {
      bracketState = emptyBracketState(data.stage.bracket);
    }
    renderApp(root, data, { onSwissWinner, onBracketWinner });
  })
  .catch((error) => {
    root.innerHTML = `<section class="panel panel-head"><h1>数据加载失败</h1><p class="muted">${error.message}</p></section>`;
  });
}

function onSwissWinner(fixtureIndex, winner) {
  const fixture = appData.stage.fixtures[fixtureIndex];
  swissState = applySwissWinner(swissState, fixture, winner);
  appData = {
    ...appData,
    stage: {
      ...appData.stage,
      standings: Object.values(swissState.records)
    }
  };
  renderApp(root, appData, { onSwissWinner, onBracketWinner });
}

function onBracketWinner(matchId, winner) {
  bracketState = applyBracketWinner(bracketState, matchId, winner);
  appData = {
    ...appData,
    stage: {
      ...appData.stage,
      bracket: bracketPayloadFromState(bracketState),
      champion_path: { champion: bracketState.champion }
    }
  };
  renderApp(root, appData, { onSwissWinner, onBracketWinner });
}

function bracketPayloadFromState(state) {
  const bracket = { quarterfinals: [], semifinals: [], final: [] };
  for (const match of Object.values(state.matches)) {
    bracket[match.round].push(match);
  }
  return bracket;
}
```

- [ ] **Step 6: Smoke check static files**

Run:

```bash
python3 -m http.server 8000 --directory site
```

Expected: server starts. Open `http://localhost:8000` and confirm the command center renders current stage data.

- [ ] **Step 7: Stop local server**

Press `Ctrl+C` in the terminal running the server.

- [ ] **Step 8: Commit**

Run:

```bash
git add site/index.html site/styles.css site/src/data.js site/src/render.js site/src/main.js
git commit -m "feat: add static command center app shell"
```

## Task 8: Wire Pick'em Impact Into Browser UI

**Files:**

- Modify: `site/src/main.js`
- Modify: `site/src/render.js`
- Test: `site/tests/pickem.test.mjs`

- [ ] **Step 1: Add Pick'em summary rendering expectation**

Extend `site/tests/pickem.test.mjs` with:

```javascript
test("summarizePickem includes missing status when record is absent", () => {
  const rows = classifyPickem({ picks: [{ category: "advance", team: "Unknown" }] }, {});
  assert.deepEqual(summarizePickem(rows), { locked: 0, alive: 0, broken: 0, missing: 1 });
});
```

- [ ] **Step 2: Run Pick'em tests**

Run:

```bash
node --test site/tests/pickem.test.mjs
```

Expected: 3 Pick'em tests pass.

- [ ] **Step 3: Update main app to classify Pick'em after winner selection**

Modify `site/src/main.js`:

```javascript
import { loadSiteData } from "./data.js";
import { resetSwissState, applySwissWinner } from "./swiss.js";
import { emptyBracketState, applyBracketWinner } from "./bracket.js";
import { classifyPickem, summarizePickem } from "./pickem.js";
import { renderApp } from "./render.js";

const root = document.querySelector("#app");
let appData = null;
let swissState = null;
let bracketState = null;

loadCurrentRoute();
window.addEventListener("hashchange", loadCurrentRoute);

function loadCurrentRoute() {
  loadSiteData(window.location.hash || "#/")
  .then((data) => {
    appData = enrichPickem(data);
    swissState = null;
    bracketState = null;
    if (data.stage.format === "swiss" && !data.stage.empty_state) {
      swissState = resetSwissState(data.stage.standings);
    }
    if (data.stage.format === "playoff" && !data.stage.empty_state) {
      bracketState = emptyBracketState(data.stage.bracket);
    }
    renderApp(root, appData, { onSwissWinner, onBracketWinner });
  })
  .catch((error) => {
    root.innerHTML = `<section class="panel panel-head"><h1>数据加载失败</h1><p class="muted">${error.message}</p></section>`;
  });
}

function onSwissWinner(fixtureIndex, winner) {
  const fixture = appData.stage.fixtures[fixtureIndex];
  swissState = applySwissWinner(swissState, fixture, winner);
  appData = enrichPickem({
    ...appData,
    stage: {
      ...appData.stage,
      standings: Object.values(swissState.records)
    }
  });
  renderApp(root, appData, { onSwissWinner, onBracketWinner });
}

function onBracketWinner(matchId, winner) {
  bracketState = applyBracketWinner(bracketState, matchId, winner);
  appData = enrichPickem({
    ...appData,
    stage: {
      ...appData.stage,
      bracket: bracketPayloadFromState(bracketState),
      champion_path: { champion: bracketState.champion }
    }
  });
  renderApp(root, appData, { onSwissWinner, onBracketWinner });
}

function enrichPickem(data) {
  if (!data.pickem || !data.pickem.picks || data.stage.empty_state) {
    return data;
  }
  const records = {};
  for (const row of data.stage.standings) {
    records[row.team] = row;
  }
  const rows = classifyPickem(data.pickem, records);
  return {
    ...data,
    pickemRuntime: {
      rows,
      summary: summarizePickem(rows)
    }
  };
}

function bracketPayloadFromState(state) {
  const bracket = { quarterfinals: [], semifinals: [], final: [] };
  for (const match of Object.values(state.matches)) {
    bracket[match.round].push(match);
  }
  return bracket;
}
```

- [ ] **Step 4: Update AI Desk renderer to show runtime Pick'em summary**

Modify `renderAiDesk` in `site/src/render.js` to include this block above article rows:

```javascript
const runtime = data.pickemRuntime;
const runtimeHtml = runtime ? `
  <div class="article-row">
    <div>
      <strong>Pick'em runtime</strong>
      <p class="muted">${runtime.summary.locked} locked / ${runtime.summary.alive} alive / ${runtime.summary.broken} broken</p>
    </div>
    <span class="mono muted">local</span>
  </div>
` : "";
```

Then return:

```javascript
return `
  <div class="panel-head"><h2>AI Desk</h2><span class="mono muted">${articles.length} articles</span></div>
  ${runtimeHtml}
  ${articles.map((article) => `
    <article class="article-row">
      <div>
        <strong>${escapeHtml(article.title)}</strong>
        <p class="muted">${escapeHtml(article.summary)}</p>
      </div>
      <span class="mono muted">${escapeHtml(article.type)}</span>
    </article>
  `).join("")}
`;
```

- [ ] **Step 5: Run JS tests**

Run:

```bash
node --test site/tests/*.test.mjs
```

Expected: all JS tests pass.

- [ ] **Step 6: Manual browser check**

Run:

```bash
python3 -m http.server 8000 --directory site
```

Open `http://localhost:8000`, click BIG over NRG, and confirm the AI Desk runtime summary changes.

- [ ] **Step 7: Stop local server**

Press `Ctrl+C`.

- [ ] **Step 8: Commit**

Run:

```bash
git add site/src/main.js site/src/render.js site/tests/pickem.test.mjs
git commit -m "feat: show live pickem impact in command center"
```

## Task 9: GitHub Pages Workflow

**Files:**

- Create: `.github/workflows/pages.yml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add Pages workflow**

Create `.github/workflows/pages.yml`:

```yaml
name: Deploy static site

on:
  schedule:
    - cron: "0 18 * * *"
  workflow_dispatch:
  push:
    branches: [ main ]
    paths:
      - "site/**"
      - "scripts/export_site_data.py"
      - "scripts/update_site_data.py"
      - "scripts/generate_ai_articles.py"
      - "data/cologne2026/**"
      - ".github/workflows/pages.yml"

permissions:
  contents: write
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
      - name: Run Python tests for site data
        run: PYTHONPATH=src python -m unittest tests.test_site_export tests.test_site_update tests.test_ai_articles -v
      - name: Run browser logic tests
        run: node --test site/tests/*.test.mjs
      - name: Refresh completed match sources
        run: PYTHONPATH=src python scripts/update_site_data.py --repo-root . --output-dir data/cologne2026/site_updates
      - name: Export static site data
        run: PYTHONPATH=src python scripts/export_site_data.py --repo-root . --output-dir site/data
      - name: Generate AI articles
        env:
          AI_API_KEY: ${{ secrets.AI_API_KEY }}
          AI_BASE_URL: https://zhengdatech.com/openai/v1
          AI_MODEL: gpt-5.5
        run: PYTHONPATH=src python scripts/generate_ai_articles.py --data-dir site/data --output-dir site/data/ai
      - name: Persist successful static data snapshot
        if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
        run: |
          STATUS=$(python - <<'PY'
          import json
          from pathlib import Path
          path = Path("data/cologne2026/site_updates/latest.json")
          if not path.exists():
              print("no-update")
          else:
              print(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
          PY
          )
          if ([ "$STATUS" = "primary_success" ] || [ "$STATUS" = "fallback_success" ] || [ "$STATUS" = "cached" ]) && [ -n "$(git status --porcelain data/cologne2026/site_updates site/data)" ]; then
            git config user.name "github-actions[bot]"
            git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
            git add data/cologne2026/site_updates site/data
            git commit -m "data: refresh static site snapshot [skip ci]"
            git push
          fi
      - name: Configure Pages
        uses: actions/configure-pages@v5
      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Extend CI workflow**

Modify `.github/workflows/ci.yml` by adding a second job:

```yaml
  site:
    name: Static site tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
      - name: Run site Python tests
        run: PYTHONPATH=src python -m unittest tests.test_site_export tests.test_site_update tests.test_ai_articles -v
      - name: Dry-run source refresh without network
        run: PYTHONPATH=src python scripts/update_site_data.py --repo-root . --output-dir data/cologne2026/site_updates --disable-primary --disable-fivee
      - name: Export static site data
        run: PYTHONPATH=src python scripts/export_site_data.py --repo-root . --output-dir site/data
      - name: Generate fallback AI articles
        run: PYTHONPATH=src python scripts/generate_ai_articles.py --data-dir site/data --output-dir site/data/ai
      - name: Run browser logic tests
        run: node --test site/tests/*.test.mjs
```

- [ ] **Step 3: Run local workflow-equivalent commands**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_site_export tests.test_site_update tests.test_ai_articles -v
PYTHONPATH=src python3 scripts/update_site_data.py --repo-root . --output-dir data/cologne2026/site_updates --disable-primary --disable-fivee
PYTHONPATH=src python3 scripts/export_site_data.py --repo-root . --output-dir site/data
PYTHONPATH=src python3 scripts/generate_ai_articles.py --data-dir site/data --output-dir site/data/ai
node --test site/tests/*.test.mjs
```

Expected: Python tests pass, exporters write JSON, JS tests pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add .github/workflows/pages.yml .github/workflows/ci.yml
git commit -m "ci: deploy static command center to pages"
```

## Task 10: Documentation and Final Verification

**Files:**

- Modify: `README.md`
- Modify: `docs/data-processing.md`

- [ ] **Step 1: Update README website section**

Add this exact Markdown block before `## 技术二级菜单` in `README.md`:

````markdown
## 静态网站部署

`site/` 是 GitHub Pages 静态站入口。它展示当前 Stage 指挥中心、Swiss/Bracket 推演、AI Desk 文章和数据来源状态。

本地预览：

```bash
PYTHONPATH=src python3 scripts/update_site_data.py --repo-root . --output-dir data/cologne2026/site_updates --disable-primary --disable-fivee
PYTHONPATH=src python3 scripts/export_site_data.py --repo-root . --output-dir site/data
PYTHONPATH=src python3 scripts/generate_ai_articles.py --data-dir site/data --output-dir site/data/ai
python3 -m http.server 8000 --directory site
```

然后打开 `http://localhost:8000`。

自动部署使用 `.github/workflows/pages.yml`，每天北京时间 02:00 更新一次，也可以在 GitHub Actions 手动触发。成功更新的数据快照会提交回仓库，用于下一次源失败时保留上一次有效页面。AI API key 必须放在 GitHub Secrets 的 `AI_API_KEY`，不要写入仓库。
````

- [ ] **Step 2: Update data-processing static export section**

Add this exact Markdown block after the IEM Cologne data assets table in `docs/data-processing.md`:

````markdown
## 静态网站数据导出

GitHub Pages 站点只读取 `site/data/*.json`。用下面命令从已复核的赛事数据生成静态 JSON：

```bash
PYTHONPATH=src python3 scripts/update_site_data.py \
  --repo-root . \
  --output-dir data/cologne2026/site_updates \
  --disable-primary \
  --disable-fivee

PYTHONPATH=src python3 scripts/export_site_data.py \
  --repo-root . \
  --output-dir site/data

PYTHONPATH=src python3 scripts/generate_ai_articles.py \
  --data-dir site/data \
  --output-dir site/data/ai
```

`update_site_data.py` 在 GitHub Actions 定时任务中会启用主来源和 5E fallback；成功生成的 `data/cologne2026/site_updates` 与 `site/data` 会提交回仓库，作为下一次失败时的有效缓存。本地文档命令使用 `--disable-primary --disable-fivee` 方便离线验证。`generate_ai_articles.py` 在 `AI_API_KEY` 存在时调用 OpenAI-compatible API；没有 key 或 API 失败时写入模板 fallback 文章。
````

- [ ] **Step 3: Run complete verification**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/update_site_data.py --repo-root . --output-dir data/cologne2026/site_updates --disable-primary --disable-fivee
PYTHONPATH=src python3 scripts/export_site_data.py --repo-root . --output-dir site/data
PYTHONPATH=src python3 scripts/generate_ai_articles.py --data-dir site/data --output-dir site/data/ai
node --test site/tests/*.test.mjs
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
HTMLParser().feed(Path("site/index.html").read_text(encoding="utf-8"))
for path in ["site/data/latest.json", "site/data/stages/stage-1.json", "site/data/ai/articles.json"]:
    assert Path(path).exists(), path
print("static site verification ok")
PY
```

Expected:

- All Python unit tests pass.
- Export scripts complete without printing secrets.
- Node tests pass.
- HTML parser script prints `static site verification ok`.

- [ ] **Step 4: Manual browser verification**

Run:

```bash
python3 -m http.server 8000 --directory site
```

Open `http://localhost:8000` and verify:

- Home page renders current Stage 1 data.
- Source status is visible.
- AI Desk shows at least one article.
- Clicking a Swiss match winner updates standings and Pick'em runtime summary.
- Stage 2 and Stage 3 links show complete future-state screens.

- [ ] **Step 5: Stop local server**

Press `Ctrl+C`.

- [ ] **Step 6: Commit docs**

Run:

```bash
git add README.md docs/data-processing.md
git commit -m "docs: document static site deployment"
```

- [ ] **Step 7: Final status check**

Run:

```bash
git status --short
```

Expected: no uncommitted changes from this implementation plan, except unrelated pre-existing work the executor intentionally did not touch.
