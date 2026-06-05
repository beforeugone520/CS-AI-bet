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
            self.assertEqual(articles["update_notes"], "")

    def test_build_ai_request_uses_openai_compatible_shape(self):
        from scripts.generate_ai_articles import build_ai_request

        request = build_ai_request(
            model="gpt-5.5",
            data_summary={"alive_picks": ["BIG", "TYLOO"], "locked": 4, "update_notes": "只看今晚赛果"},
        )

        self.assertEqual(request["model"], "gpt-5.5")
        self.assertEqual(request["temperature"], 0.4)
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(request["messages"][1]["role"], "user")
        self.assertIn("BIG", request["messages"][1]["content"])
        self.assertIn("只看今晚赛果", request["messages"][1]["content"])

    def test_manual_update_notes_are_normalized_into_summary(self):
        from scripts.generate_ai_articles import generate_ai_articles

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_fixture_data(data_dir)

            generate_ai_articles(
                data_dir=data_dir,
                output_dir=data_dir / "ai",
                api_key=None,
                update_notes="  Stage 1 已结束\n补充 TYLOO 赛果  ",
            )

            articles = json.loads((data_dir / "ai" / "articles.json").read_text(encoding="utf-8"))
            self.assertEqual(articles["update_notes"], "Stage 1 已结束 补充 TYLOO 赛果")


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
