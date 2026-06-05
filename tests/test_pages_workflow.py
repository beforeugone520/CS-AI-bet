import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PagesWorkflowTests(unittest.TestCase):
    def test_pages_workflow_is_manual_ai_led_not_scheduled_scraping(self):
        workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")

        self.assertNotIn("schedule:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("ai_update_notes", workflow)
        self.assertIn("refresh_sources", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch' && inputs.refresh_sources == 'true'", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch' && inputs.ai_update_notes != ''", workflow)


if __name__ == "__main__":
    unittest.main()
