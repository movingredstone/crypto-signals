import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _fake_row(interval, family, score, params=None):
    params = params or {
        "symbol": "BTCUSDT",
        "interval": interval,
        "family": family,
        "direction_filter": "both",
        "lookback": 20,
        "volume_min": 1.0,
        "atr_stop_mult": 2.0,
        "take_profit_r": 2.0,
        "max_holding_bars": 48,
        "stop_rule": "atr",
        "adx_min": 20,
        "regime": "any",
        "trailing_atr_mult": None,
        "breakeven_r": None,
        "partial_tp_r": None,
    }
    return {
        "experiment_id": 1,
        "symbol": "BTCUSDT",
        "interval": interval,
        "family": family,
        "params_json": json.dumps(params),
        "train_return_pct": 20.0,
        "train_mdd_pct": -5.0,
        "train_pf": 1.5,
        "train_trades": 80,
        "val_return_pct": 12.0,
        "val_mdd_pct": -4.0,
        "val_pf": 1.4,
        "val_trades": 40,
        "test_return_pct": 10.0,
        "test_mdd_pct": -3.0,
        "test_pf": 1.3,
        "test_trades": 30,
        "test_win_rate_pct": 52.0,
        "test_fees": 10.0,
        "test_slippage_cost": 5.0,
        "wf_folds": 4,
        "wf_pos_folds": 3,
        "wf_mean_return": 4.0,
        "wf_min_return": -1.0,
        "wf_total_trades": 150,
        "robust_score": score,
    }


class OptimizerTests(unittest.TestCase):
    def test_broad_optimization_runs_all_intervals_and_writes_ranked_outputs(self):
        from src import optimizer

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            calls = []

            def fake_run_research(**kwargs):
                calls.append(kwargs)
                interval = kwargs["interval"]
                rows = [
                    _fake_row(interval, "slow", 5.0),
                    _fake_row(interval, "fast", 25.0),
                ]
                path = tmp_path / f"{interval}_research.csv"
                pd.DataFrame(rows).to_csv(path, index=False)
                return {"path": str(path), "top": rows, "total_experiments": len(rows)}

            with patch.object(optimizer, "run_research", fake_run_research):
                report = optimizer.run_optimization(
                    symbol="BTCUSDT",
                    intervals=["1h", "4h"],
                    experiments=123,
                    workers=7,
                    stage="broad",
                    output_dir=tmp_path / "optimization",
                    top_n=3,
                    seed=99,
                    wf_folds=5,
                )

            self.assertEqual([c["interval"] for c in calls], ["1h", "4h"])
            self.assertTrue(all(c["max_experiments"] == 123 for c in calls))
            self.assertTrue(all(c["max_workers"] == 7 for c in calls))
            self.assertEqual(report["stage"], "broad")
            self.assertEqual(report["top_candidates"][0]["family"], "fast")
            self.assertEqual(report["top_candidates"][0]["robust_score"], 25.0)
            self.assertTrue(Path(report["top_candidates_path"]).exists())
            self.assertTrue(Path(report["summary_path"]).exists())
            self.assertTrue(Path(report["markdown_path"]).exists())

            ranked = pd.read_csv(report["top_candidates_path"])
            self.assertEqual(len(ranked), 3)
            self.assertGreaterEqual(ranked.iloc[0]["robust_score"], ranked.iloc[-1]["robust_score"])

    def test_build_refine_focuses_from_top_candidate_parameters(self):
        from src.optimizer import build_refine_focuses

        row = _fake_row(
            "1h",
            "trend_pullback",
            42.0,
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "family": "trend_pullback",
                "direction_filter": "long_only",
                "lookback": 48,
                "volume_min": 1.5,
                "atr_stop_mult": 2.0,
                "take_profit_r": 2.5,
                "max_holding_bars": 72,
                "stop_rule": "trailing_atr",
                "adx_min": 25,
                "regime": "trend",
                "trailing_atr_mult": 2.5,
                "breakeven_r": 1.0,
                "partial_tp_r": 1.5,
                "partial_tp_frac": 0.5,
            },
        )

        focuses = build_refine_focuses([row], per_candidate=1)

        self.assertEqual(len(focuses), 1)
        focus = focuses[0]
        self.assertEqual(focus["families"], ["trend_pullback"])
        self.assertEqual(focus["direction_filters"], ["long_only"])
        self.assertIn(48, focus["lookbacks"])
        self.assertTrue({36, 48, 60}.issubset(set(focus["lookbacks"])))
        self.assertIn(2.0, focus["atr_stop_mults"])
        self.assertIn(2.5, focus["take_profit_rs"])
        self.assertEqual(focus["stop_rules"], ["trailing_atr"])

    def test_format_optimization_report_contains_artifacts_and_best_candidate(self):
        from src.optimizer import format_optimization_report

        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            report = {
                "symbol": "BTCUSDT",
                "stage": "broad",
                "intervals": ["1h"],
                "experiments": 100,
                "workers": 4,
                "top_candidates_path": str(tmp_path / "top.csv"),
                "summary_path": str(tmp_path / "summary.json"),
                "markdown_path": str(tmp_path / "report.md"),
                "top_candidates": [_fake_row("1h", "trend_pullback", 42.0)],
            }

            text = format_optimization_report(report)

            self.assertIn("[investmentsystem Optimization]", text)
            self.assertIn("BTCUSDT", text)
            self.assertIn("trend_pullback", text)
            self.assertIn("score=42.0", text)
            self.assertIn(str(tmp_path / "top.csv"), text)


if __name__ == "__main__":
    unittest.main()
