#!/usr/bin/env python3
"""Render the README Pick'em checkpoint board as a self-contained SVG."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


WIDTH = 1200
HEIGHT = 720
LANE_LAYOUTS = {
    0: {"line_y": 128, "title_y": 158, "subtitle_y": 181, "card_y": 148},
    1: {"line_y": 306, "title_y": 336, "subtitle_y": 359, "card_y": 318},
    2: {"line_y": 508, "title_y": 538, "subtitle_y": 561, "card_y": 580},
}
TONE_CLASSES = {
    "good": ("good", "good-fill"),
    "warn": ("warn", "warn-fill"),
    "bad": ("bad", "bad-fill"),
}


def text(value: Any) -> str:
    return escape(str(value), quote=True)


def card_svg(card: dict[str, Any], x: int, y: int, width: int, height: int) -> str:
    tone = str(card.get("tone", "good"))
    stroke_class, fill_class = TONE_CLASSES.get(tone, TONE_CLASSES["good"])
    band_width = 82 if width >= 300 else 66
    record_x = band_width / 2
    team_x = band_width + 22
    team_font = 20 if len(str(card["team"])) > 15 else 22
    score = card.get("score")
    score_text = ""
    if score:
        score_text = f'<text x="{width - 24}" y="37" text-anchor="end" class="record">{text(score)}</text>'
    return f'''  <g transform="translate({x} {y})">
    <rect class="card {stroke_class}" x="0" y="0" width="{width}" height="{height}" rx="8"/>
    <rect class="{fill_class}" x="0" y="0" width="{band_width}" height="{height}" rx="8"/>
    <text x="{record_x:.0f}" y="{31 if height < 80 else 35}" text-anchor="middle" class="tick">{text(card["record"])}</text>
    <text x="{record_x:.0f}" y="{51 if height < 80 else 55}" text-anchor="middle" class="label">{text(card["label"])}</text>
    <text x="{team_x}" y="{31 if height < 80 else 34}" class="team" font-size="{team_font}">{text(card["team"])}</text>
    <text x="{team_x}" y="{56 if height < 80 else 61}" class="status">{text(card["status"])}</text>
    {score_text}
  </g>'''


def lane_card_positions(index: int, cards: list[dict[str, Any]]) -> list[tuple[int, int, int, int]]:
    y = LANE_LAYOUTS[index]["card_y"]
    if index == 0:
        return [(272, y, 310, 92), (626, y, 310, 92)]
    if index == 1:
        top = [(272, y, 250, 82), (542, y, 250, 82), (812, y, 250, 82)]
        bottom = [(272, y + 98, 250, 82), (542, y + 98, 250, 82), (812, y + 98, 250, 82)]
        return (top + bottom)[: len(cards)]
    return [(272, y, 310, 72), (626, y, 310, 72)]


def render_svg(data: dict[str, Any]) -> str:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f'  <title id="title">{text(data["title"])} Pick&apos;em Day 1 checkpoint</title>',
        f'  <desc id="desc">A dark esports status board showing 3-0 picks, advance picks, and 0-3 picks after two Swiss rounds on {text(data["date"])}.</desc>',
        "  <defs>",
        '    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0" stop-color="#071020"/>',
        '      <stop offset="0.54" stop-color="#0b1326"/>',
        '      <stop offset="1" stop-color="#101827"/>',
        "    </linearGradient>",
        '    <linearGradient id="rail" x1="0" y1="0" x2="1" y2="0">',
        '      <stop offset="0" stop-color="#1b7cff"/>',
        '      <stop offset="1" stop-color="#59a7ff"/>',
        "    </linearGradient>",
        '    <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">',
        '      <feGaussianBlur stdDeviation="3" result="blur"/>',
        "      <feMerge>",
        '        <feMergeNode in="blur"/>',
        '        <feMergeNode in="SourceGraphic"/>',
        "      </feMerge>",
        "    </filter>",
        "    <style>",
        "      .bg { fill: url(#bg); }",
        "      .lane-title { fill: #dce9ff; font: 700 25px Arial, Helvetica, sans-serif; letter-spacing: 0; }",
        "      .lane-sub { fill: #8ea6ca; font: 500 14px Arial, Helvetica, sans-serif; letter-spacing: 0; }",
        "      .meta { fill: #8ea6ca; font: 500 15px Arial, Helvetica, sans-serif; letter-spacing: 0; }",
        "      .team { fill: #f4f8ff; font-family: Arial, Helvetica, sans-serif; font-weight: 700; letter-spacing: 0; }",
        "      .record { fill: #d5e3f8; font: 700 18px Arial, Helvetica, sans-serif; letter-spacing: 0; }",
        "      .status { fill: #d5e3f8; font: 700 13px Arial, Helvetica, sans-serif; letter-spacing: 0; }",
        "      .note { fill: #9db0cd; font: 500 13px Arial, Helvetica, sans-serif; letter-spacing: 0; }",
        "      .label { fill: #0b1326; font: 800 12px Arial, Helvetica, sans-serif; letter-spacing: 0; }",
        "      .lane-line { stroke: url(#rail); stroke-width: 2; opacity: 0.9; }",
        "      .lane-dot { fill: #43a5ff; filter: url(#softGlow); }",
        "      .card { fill: #172338; stroke-width: 2; }",
        "      .good { stroke: #38d56b; }",
        "      .warn { stroke: #f4c84a; }",
        "      .bad { stroke: #f06464; }",
        "      .good-fill { fill: #1e7e3a; }",
        "      .warn-fill { fill: #8f721e; }",
        "      .bad-fill { fill: #8d2d36; }",
        "      .tick { fill: #0d1728; font: 800 14px Arial, Helvetica, sans-serif; letter-spacing: 0; }",
        "    </style>",
        "  </defs>",
        "",
        f'  <rect class="bg" width="{WIDTH}" height="{HEIGHT}" rx="0"/>',
        '  <rect x="36" y="32" width="1128" height="676" rx="22" fill="none" stroke="#203a62" stroke-width="1.2"/>',
        '  <path d="M72 128H1128M72 306H1128M72 508H1128" class="lane-line"/>',
        "",
        f'  <text x="72" y="80" fill="#f4f8ff" font-family="Arial, Helvetica, sans-serif" font-size="34" font-weight="800">{text(data["title"])}</text>',
        f'  <text x="72" y="108" class="meta">{text(data["subtitle"])} · {text(data["date"])}</text>',
        f'  <text x="1128" y="86" text-anchor="end" fill="#43a5ff" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="800">{text(data["badge"])}</text>',
        f'  <text x="1128" y="110" text-anchor="end" class="meta">{text(data["source_note"])}</text>',
        "",
    ]
    for index, lane in enumerate(data["lanes"]):
        layout = LANE_LAYOUTS[index]
        lines.extend(
            [
                f'  <circle cx="72" cy="{layout["line_y"]}" r="5" class="lane-dot"/>',
                f'  <text x="88" y="{layout["title_y"]}" class="lane-title">{text(lane["title"])}</text>',
                f'  <text x="88" y="{layout["subtitle_y"]}" class="lane-sub">{text(lane["subtitle"])}</text>',
                "",
            ]
        )
        cards = lane["cards"]
        for card, position in zip(cards, lane_card_positions(index, cards)):
            lines.append(card_svg(card, *position))
            lines.append("")
    lines.append(f'  <text x="1128" y="696" text-anchor="end" class="note">{text(data["legend"])}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Checkpoint JSON data.")
    parser.add_argument("--output", required=True, type=Path, help="SVG output path.")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    svg = render_svg(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
