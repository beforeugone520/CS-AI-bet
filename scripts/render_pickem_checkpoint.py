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
COLUMN_MARGIN = 44
COLUMN_GAP = 16
COLUMN_Y = 188
COLUMN_H = 500
CARD_H = 62
CARD_GAP = 10
CARD_START_Y = 316
TONE = {
    "good": {
        "stroke": "#34d66d",
        "fill": "#1c7f3e",
        "soft": "#123323",
        "text": "#d9ffe6",
    },
    "warn": {
        "stroke": "#f4c84a",
        "fill": "#9a7b1b",
        "soft": "#342b13",
        "text": "#fff1bd",
    },
    "bad": {
        "stroke": "#ff5c68",
        "fill": "#96313b",
        "soft": "#351920",
        "text": "#ffd5d9",
    },
    "muted": {
        "stroke": "#50627e",
        "fill": "#39465d",
        "soft": "#1c2536",
        "text": "#d6deeb",
    },
}


def text(value: Any) -> str:
    return escape(str(value), quote=True)


def tone(name: str, key: str) -> str:
    return TONE.get(name, TONE["good"])[key]


def style_block() -> str:
    return """    <style>
      .bg { fill: url(#bg); }
      .grid { fill: url(#grid); opacity: 0.23; }
      .outer { fill: rgba(11, 19, 38, 0.58); stroke: #233f69; stroke-width: 1.3; }
      .column { fill: #0f1a2d; stroke-width: 1.5; }
      .column-glow { opacity: 0.18; }
      .title { fill: #f4f8ff; font: 800 35px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .subtitle { fill: #9fb4d2; font: 500 15px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .badge { fill: #43a5ff; font: 800 19px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .record { fill: #f4f8ff; font: 900 42px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .column-title { fill: #dce9ff; font: 800 18px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .column-sub { fill: #8ea6ca; font: 500 12px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .team { fill: #f4f8ff; font-family: Arial, Helvetica, sans-serif; font-weight: 800; letter-spacing: 0; }
      .status { fill: #aebfda; font: 600 11px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .slot { fill: #08111f; font: 900 11px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .metric-value { fill: #f4f8ff; font: 900 23px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .metric-label { fill: #aebfda; font: 700 12px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .footer { fill: #9fb4d2; font: 600 13px Arial, Helvetica, sans-serif; letter-spacing: 0; }
      .rail { stroke: #2d8cff; stroke-width: 2; opacity: 0.9; }
    </style>"""


def metric_svg(metric: dict[str, Any], x: int) -> str:
    tone_name = str(metric.get("tone", "good"))
    return f'''  <g transform="translate({x} 126)">
    <rect x="0" y="0" width="150" height="46" rx="8" fill="{tone(tone_name, "soft")}" stroke="{tone(tone_name, "stroke")}" stroke-width="1.4"/>
    <text x="22" y="30" class="metric-value">{text(metric["value"])}</text>
    <text x="56" y="28" class="metric-label">{text(metric["label"])}</text>
  </g>'''


def card_svg(card: dict[str, Any], x: int, y: int, width: int) -> str:
    tone_name = str(card.get("tone", "good"))
    team = str(card["team"])
    team_font = 15 if len(team) > 15 else 17
    tag_w = 42 if len(str(card["slot"])) <= 3 else 50
    return f'''  <g transform="translate({x} {y})">
    <rect x="0" y="0" width="{width}" height="{CARD_H}" rx="8" fill="#172338" stroke="{tone(tone_name, "stroke")}" stroke-width="1.8"/>
    <rect x="0" y="0" width="7" height="{CARD_H}" rx="3.5" fill="{tone(tone_name, "stroke")}"/>
    <rect x="16" y="15" width="{tag_w}" height="26" rx="6" fill="{tone(tone_name, "fill")}"/>
    <text x="{16 + tag_w / 2:.0f}" y="33" text-anchor="middle" class="slot">{text(card["slot"])}</text>
    <text x="{30 + tag_w}" y="28" class="team" font-size="{team_font}">{text(team)}</text>
    <text x="{30 + tag_w}" y="49" class="status">{text(card["status"])}</text>
  </g>'''


def column_width(total_columns: int) -> int:
    return int((WIDTH - 2 * COLUMN_MARGIN - (total_columns - 1) * COLUMN_GAP) / total_columns)


def column_x(index: int, total_columns: int) -> int:
    return COLUMN_MARGIN + index * (column_width(total_columns) + COLUMN_GAP)


def column_svg(column: dict[str, Any], index: int, total_columns: int) -> str:
    x = column_x(index, total_columns)
    column_w = column_width(total_columns)
    tone_name = str(column.get("tone", "good"))
    lines = [
        f'  <g transform="translate({x} {COLUMN_Y})">',
        f'    <rect class="column-glow" x="0" y="0" width="{column_w}" height="{COLUMN_H}" rx="14" fill="{tone(tone_name, "stroke")}"/>',
        f'    <rect class="column" x="0" y="0" width="{column_w}" height="{COLUMN_H}" rx="14" stroke="{tone(tone_name, "stroke")}"/>',
        f'    <rect x="18" y="22" width="78" height="66" rx="10" fill="{tone(tone_name, "fill")}"/>',
        f'    <text x="57" y="66" text-anchor="middle" class="record">{text(column["record"])}</text>',
        f'    <text x="112" y="45" class="column-title">{text(column["title"])}</text>',
        f'    <text x="112" y="70" class="column-sub">{text(column["subtitle"])}</text>',
        f'    <line x1="20" y1="116" x2="{column_w - 20}" y2="116" class="rail"/>',
        "  </g>",
        "",
    ]
    card_x = x + 18
    card_w = column_w - 36
    for card_index, card in enumerate(column["cards"]):
        y = CARD_START_Y + card_index * (CARD_H + CARD_GAP)
        lines.append(card_svg(card, card_x, y, card_w))
        lines.append("")
    return "\n".join(lines)


def render_svg(data: dict[str, Any]) -> str:
    metric_x = [72, 236, 400]
    columns = data["columns"]
    title = data["title"]
    subtitle = data.get("subtitle", "Pick'em checkpoint")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f'  <title id="title">{text(title)} Pick&apos;em checkpoint</title>',
        f'  <desc id="desc">A Swiss record board showing {text(subtitle)} on {text(data["date"])}.</desc>',
        "  <defs>",
        '    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '      <stop offset="0" stop-color="#071020"/>',
        '      <stop offset="0.58" stop-color="#0b1326"/>',
        '      <stop offset="1" stop-color="#111827"/>',
        "    </linearGradient>",
        '    <pattern id="grid" width="44" height="44" patternUnits="userSpaceOnUse">',
        '      <path d="M44 0H0V44" fill="none" stroke="#1b3558" stroke-width="0.8"/>',
        "    </pattern>",
        style_block(),
        "  </defs>",
        "",
        f'  <rect class="bg" width="{WIDTH}" height="{HEIGHT}"/>',
        f'  <rect class="grid" width="{WIDTH}" height="{HEIGHT}"/>',
        '  <rect class="outer" x="34" y="24" width="1132" height="692" rx="22"/>',
        "",
        f'  <text x="72" y="78" class="title">{text(data["title"])}</text>',
        f'  <text x="72" y="106" class="subtitle">{text(data["subtitle"])} · {text(data["date"])}</text>',
        f'  <text x="1128" y="84" text-anchor="end" class="badge">{text(data["badge"])}</text>',
        f'  <text x="1128" y="108" text-anchor="end" class="subtitle">{text(data["source_note"])}</text>',
        "",
    ]
    for metric, x in zip(data.get("summary", []), metric_x):
        lines.append(metric_svg(metric, x))
        lines.append("")
    for index, column in enumerate(columns):
        lines.append(column_svg(column, index, len(columns)))
    lines.append(f'  <text x="1128" y="700" text-anchor="end" class="footer">{text(data["footer"])}</text>')
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
