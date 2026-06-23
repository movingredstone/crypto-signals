import unittest


class ResearchProgressTests(unittest.TestCase):
    def test_format_progress_message_includes_percent_done_and_remaining(self):
        from src.research_engine import format_progress_message

        msg = format_progress_message("researched", 250, 1000, "experiments")

        self.assertIn("250/1000", msg)
        self.assertIn("25.0% done", msg)
        self.assertIn("75.0% remaining", msg)

    def test_format_progress_message_handles_zero_total(self):
        from src.research_engine import format_progress_message

        msg = format_progress_message("researched", 0, 0, "experiments")

        self.assertIn("0/0", msg)
        self.assertIn("100.0% done", msg)
        self.assertIn("0.0% remaining", msg)


if __name__ == "__main__":
    unittest.main()
