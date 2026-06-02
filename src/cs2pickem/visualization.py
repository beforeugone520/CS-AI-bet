from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .data import write_json


def write_training_visualizations(report: Mapping[str, Any], output_dir: str, prefix: str = "training") -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    importance = _importance_items(report)
    probabilities = _probabilities(report)

    renderer = "matplotlib" if _can_use_matplotlib() else "svg"
    extension = "png" if renderer == "matplotlib" else "svg"
    importance_path = os.path.abspath(os.path.join(output_dir, f"{prefix}_feature_importance.{extension}"))
    probabilities_path = os.path.abspath(os.path.join(output_dir, f"{prefix}_probability_distribution.{extension}"))

    if renderer == "matplotlib":
        _write_matplotlib_charts(importance, probabilities, importance_path, probabilities_path)
    else:
        _write_bar_svg(importance_path, "Feature importance", importance)
        _write_bar_svg(probabilities_path, "Prediction probability distribution", _probability_bins(probabilities))

    manifest = {
        "renderer": renderer,
        "feature_importance_path": importance_path,
        "probability_distribution_path": probabilities_path,
        "feature_count": len(importance),
        "probability_count": len(probabilities),
    }
    write_json(os.path.join(output_dir, f"{prefix}_visualization_manifest.json"), manifest)
    return manifest


def visualize_training_report_file(report_path: str, output_dir: str, prefix: str = "training") -> Dict[str, object]:
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    return write_training_visualizations(report, output_dir, prefix=prefix)


def _importance_items(report: Mapping[str, Any]) -> List[Tuple[str, float]]:
    raw = report.get("feature_importance", {})
    if isinstance(raw, Mapping):
        items = [(str(name), _num(score)) for name, score in raw.items()]
    else:
        items = []
    return sorted(items, key=lambda item: (-item[1], item[0]))[:25]


def _probabilities(report: Mapping[str, Any]) -> List[float]:
    output = []
    for row in report.get("probabilities", []) or []:
        if isinstance(row, Mapping):
            output.append(min(1.0, max(0.0, _num(row.get("winner_probability_team1"), 0.5))))
    return output


def _can_use_matplotlib() -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401

        return True
    except Exception:
        return False


def _write_matplotlib_charts(
    importance: List[Tuple[str, float]],
    probabilities: List[float],
    importance_path: str,
    probabilities_path: str,
) -> None:
    import matplotlib.pyplot as plt

    names = [name for name, _ in importance] or ["no features"]
    scores = [score for _, score in importance] or [0.0]
    height = max(3.0, len(names) * 0.35)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(names, scores, color="#2563eb")
    ax.invert_yaxis()
    ax.set_xlabel("importance")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(importance_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probabilities or [0.5], bins=[index / 10 for index in range(11)], color="#059669", edgecolor="#064e3b")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("team1 win probability")
    ax.set_ylabel("matches")
    ax.set_title("Prediction probability distribution")
    fig.tight_layout()
    fig.savefig(probabilities_path, dpi=140)
    plt.close(fig)


def _write_bar_svg(path: str, title: str, items: Iterable[Tuple[str, float]]) -> None:
    items = list(items)
    width = 900
    row_height = 30
    height = max(150, 70 + len(items) * row_height)
    max_value = max([value for _, value in items] or [1.0]) or 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="30" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
    ]
    for index, (name, value) in enumerate(items):
        y = 62 + index * row_height
        bar_width = int((value / max_value) * 520) if max_value else 0
        lines.append(f'<text x="30" y="{y + 17}" font-family="Arial" font-size="13">{html.escape(name)}</text>')
        lines.append(f'<rect x="260" y="{y}" width="{bar_width}" height="18" rx="3" fill="#2563eb"/>')
        lines.append(f'<text x="{270 + bar_width}" y="{y + 14}" font-family="Arial" font-size="12">{value:.3g}</text>')
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _probability_bins(probabilities: List[float]) -> List[Tuple[str, float]]:
    bins = [0] * 10
    for probability in probabilities:
        index = min(9, int(probability * 10))
        bins[index] += 1
    return [(f"{index / 10:.1f}-{(index + 1) / 10:.1f}", float(count)) for index, count in enumerate(bins)]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
