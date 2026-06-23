import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PaperTraderStateTests(unittest.TestCase):
    def _load_module(self):
        with patch.dict(os.environ, {"TELEGRAM_TOKEN": "dummy", "TELEGRAM_CHAT_ID": "dummy"}):
            import paper_trader_github
            return importlib.reload(paper_trader_github)

    def test_load_state_backfills_missing_equity_curve_for_committed_state(self):
        paper = self._load_module()
        old_state = {
            "version": 3,
            "balance": 1000.0,
            "initial_balance": 1000.0,
            "peak_balance": 1000.0,
            "positions": [],
            "closed_trades": [],
            "signal_log": [],
            "last_run": None,
        }
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / paper.STATE_FILE
            state_path.write_text(json.dumps(old_state))
            cwd = os.getcwd()
            try:
                os.chdir(td)
                state = paper.load_state()
            finally:
                os.chdir(cwd)

        self.assertIn("equity_curve", state)
        self.assertEqual(state["equity_curve"], [])
        self.assertEqual(state["balance"], 1000.0)

    def test_empty_state_uses_current_portfolio_allocation(self):
        paper = self._load_module()
        self.assertEqual(paper.empty_state()["initial_balance"], 1000.0)


if __name__ == "__main__":
    unittest.main()
