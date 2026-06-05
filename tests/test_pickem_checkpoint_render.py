import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


class PickemCheckpointRenderTests(unittest.TestCase):
    def test_renderer_outputs_valid_svg_with_checkpoint_teams(self):
        repo = Path(__file__).resolve().parents[1]
        data_path = repo / "data/cologne2026/source_inputs/pickem_checkpoint_round4_2026-06-05.json"
        script_path = repo / "scripts/render_pickem_checkpoint.py"
        teams = [
            "GamerLegion",
            "MIBR",
            "BetBoom",
            "B8",
            "M80",
            "BIG",
            "HEROIC",
            "TYLOO",
            "Gaimin Gladiators",
            "NRG",
            "Liquid",
            "FlyQuest",
            "SINNERS",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "checkpoint.svg"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--input",
                    str(data_path),
                    "--output",
                    str(output_path),
                ],
                cwd=repo,
                check=True,
            )

            ET.parse(output_path)
            svg = output_path.read_text(encoding="utf-8")
            for team in teams:
                self.assertIn(team, svg)
            for required in ["3-x", "2-2", "1-3", "0-3", "Stage 2", "Round 5 BO3", "Eliminated"]:
                self.assertIn(required, svg)
            for required in ["locked / hit", "alive", "slot broken"]:
                self.assertIn(required, svg)
            self.assertIn("Round 4 complete", svg)
            self.assertNotIn("Day 1 checkpoint", svg)
            self.assertNotIn("after two Swiss rounds", svg)


if __name__ == "__main__":
    unittest.main()
