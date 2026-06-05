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
    update_notes: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(data_dir, update_notes=update_notes)
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
        "update_notes": summary["update_notes"],
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
                "content": (
                    "你是 CS2 电竞数据编辑。只根据用户提供的 IEM Cologne Major 2026 数据和更新备注"
                    "写简短中文分析，不引入其他赛事，不夸大投注价值。输出 JSON。"
                ),
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
            "body": (
                f"当前仍可变化的 Pick'em 槽位集中在 {alive_text}。如果这些队伍赢下后续关键比赛，"
                "advance 槽位会继续补成 locked；如果失利，对应槽位会变成 broken。"
                "该结论来自最新静态 standings 和 checkpoint 数据。"
            ),
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


def _load_summary(data_dir: Path, update_notes: str | None = None) -> dict[str, Any]:
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
        "update_notes": _normalize_update_notes(update_notes),
    }


def _normalize_update_notes(update_notes: str | None) -> str:
    normalized = " ".join((update_notes or "").split())
    return normalized[:1200]


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
        update_notes=os.environ.get("AI_UPDATE_NOTES"),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
