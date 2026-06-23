from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.research_engine import run_research, format_progress_message
from src.fold_evaluator import evaluate_folds, FOLDS, STRESS_OVERRIDES, BASELINE_OVERRIDES


NUMERIC_NEIGHBORS = {
    "lookback": ("lookbacks", "int", [0.75, 1.0, 1.25]),
    "volume_min": ("volume_mins", "float", [-0.3, -0.1, 0.0, 0.1, 0.3]),
    "atr_stop_mult": ("atr_stop_mults", "float", [-0.5, -0.25, 0.0, 0.25, 0.5]),
    "take_profit_r": ("take_profit_rs", "float", [-0.5, -0.25, 0.0, 0.25, 0.5]),
    "max_holding_bars": ("max_holding_bars", "int", [0.67, 1.0, 1.33]),
    "adx_min": ("adx_mins", "int", [-5, 0, 5]),
    "trailing_atr_mult": ("trailing_atr_mults", "float_or_none", [-0.5, 0.0, 0.5]),
    "breakeven_r": ("breakeven_rs", "float_or_none", [-0.25, 0.0, 0.25]),
    "partial_tp_r": ("partial_tp_rs", "float_or_none", [-0.25, 0.0, 0.25]),
}

CATEGORICAL_FOCUS = {
    "family": "families",
    "direction_filter": "direction_filters",
    "stop_rule": "stop_rules",
    "regime": "regimes",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _dedupe_sorted(values):
    cleaned = []
    for value in values:
        if value is None:
            cleaned.append(None)
            continue
        if isinstance(value, float):
            value = round(value, 4)
            if value < 0:
                continue
        if isinstance(value, int) and value < 0:
            continue
        cleaned.append(value)

    seen = set()
    out = []
    for value in cleaned:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return sorted(out, key=lambda x: (-1 if x is None else float(x) if isinstance(x, (int, float)) else str(x)))


def _load_rows(path: str | Path) -> list[dict]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _load_params(row: dict) -> dict:
    params = row.get("params_json") or "{}"
    if isinstance(params, dict):
        return params
    try:
        return json.loads(params)
    except Exception:
        return {}


def build_refine_focuses(top_rows: list[dict], per_candidate: int = 1) -> list[dict]:
    """Build focused search spaces around already-good candidate parameters."""
    focuses: list[dict] = []
    for row in top_rows[: max(0, per_candidate)]:
        params = _load_params(row)
        focus: dict = {}

        for param_key, focus_key in CATEGORICAL_FOCUS.items():
            value = params.get(param_key)
            if value is not None:
                focus[focus_key] = [value]

        for param_key, (focus_key, mode, offsets) in NUMERIC_NEIGHBORS.items():
            base = params.get(param_key)
            if base is None:
                if mode.endswith("or_none"):
                    focus[focus_key] = [None]
                continue
            try:
                base_num = float(base)
            except Exception:
                continue

            values = []
            if mode == "int":
                for offset in offsets:
                    if isinstance(offset, float) and abs(offset) <= 2 and offset != 0:
                        values.append(max(1, int(round(base_num * offset))))
                    else:
                        values.append(max(1, int(round(base_num + offset))))
            else:
                if mode.endswith("or_none"):
                    values.append(None)
                for offset in offsets:
                    values.append(max(0.0, base_num + float(offset)))
            focus[focus_key] = _dedupe_sorted(values)

        for scalar in [
            "rsi_low",
            "rsi_high",
            "vwap_dev_atr",
            "squeeze_pct",
            "atr_breakout_min",
            "tolerance_pct",
            "partial_tp_frac",
            "hours_allowed",
            "weekday_only",
            "funding_max_zs",
            "use_funding",
        ]:
            if scalar in params:
                focus[scalar] = params[scalar]

        focuses.append(focus)
    return focuses


def _rank_rows(rows: list[dict], top_n: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "robust_score" not in df.columns:
        df["robust_score"] = 0.0
    df["robust_score"] = pd.to_numeric(df["robust_score"], errors="coerce").fillna(-999999.0)
    sort_cols = ["robust_score"]
    ascending = [False]
    if "test_return_pct" in df.columns:
        df["test_return_pct"] = pd.to_numeric(df["test_return_pct"], errors="coerce").fillna(-999999.0)
        sort_cols.append("test_return_pct")
        ascending.append(False)
    if "test_mdd_pct" in df.columns:
        df["test_mdd_abs"] = pd.to_numeric(df["test_mdd_pct"], errors="coerce").abs().fillna(999999.0)
        sort_cols.append("test_mdd_abs")
        ascending.append(True)
    df = df.sort_values(sort_cols, ascending=ascending).head(top_n).reset_index(drop=True)
    if "test_mdd_abs" in df.columns:
        df = df.drop(columns=["test_mdd_abs"])
    return df


def _candidate_markdown(row: dict, rank: int) -> list[str]:
    params = _load_params(row)
    lines = [
        f"{rank}. {row.get('family')} / {row.get('interval')} | score={row.get('robust_score')}",
    ]

    # Fold-based format (from fold_evaluator)
    if row.get("total_folds") is not None:
        folds = f"{int(row.get('pos_folds', 0))}/{int(row.get('total_folds', 0))}"
        lines.append(
            f"   Folds: {folds} pos ({row.get('survival_pct')}%), "
            f"mean={row.get('mean_return')}%, median={row.get('median_return')}%, "
            f"min={row.get('min_return')}%, max={row.get('max_return')}%"
        )
        lines.append(
            f"   PF: mean={row.get('mean_pf')}, "
            f"MDD: mean={row.get('mean_mdd')}%, max={row.get('max_mdd')}%, "
            f"trades={row.get('total_trades')}"
        )
        # Per-fold returns for transparency
        fold_names = ["2023H1", "2023H2", "2024H1", "2024H2", "2025H1", "2025H2", "2026H1"]
        fold_keys = ["2023_h1_return_pct", "2023_h2_return_pct", "2024_h1_return_pct",
                     "2024_h2_return_pct", "2025_h1_return_pct", "2025_h2_return_pct", "2026_h1_return_pct"]

        # Try per-fold columns first, fall back to fold_returns JSON
        fold_vals = []
        for fk in fold_keys:
            v = row.get(fk)
            fold_vals.append(v)

        # If all None/NaN, try parsing fold_returns JSON
        if all(v is None or (isinstance(v, float) and pd.isna(v)) for v in fold_vals):
            fr_raw = row.get("fold_returns")
            if fr_raw and isinstance(fr_raw, str):
                try:
                    fold_vals = json.loads(fr_raw)
                except Exception:
                    pass

        fold_str = " | ".join(
            f"{n}: {v:.1f}%"
            if v is not None and isinstance(v, (int, float)) and not (isinstance(v, float) and pd.isna(v))
            else f"{n}: —"
            for n, v in zip(fold_names, fold_vals)
        )
        lines.append(f"   Per-fold: {fold_str}")
    else:
        # Legacy train/val/test format fallback
        lines.append(f"   Train: {row.get('train_return_pct')}%, PF={row.get('train_pf')}, MDD={row.get('train_mdd_pct')}%, trades={row.get('train_trades')}")
        lines.append(f"   Val:   {row.get('val_return_pct')}%, PF={row.get('val_pf')}, MDD={row.get('val_mdd_pct')}%, trades={row.get('val_trades')}")
        lines.append(f"   Test:  {row.get('test_return_pct')}%, PF={row.get('test_pf')}, MDD={row.get('test_mdd_pct')}%, trades={row.get('test_trades')}")
        lines.append(f"   WF:    pos={row.get('wf_pos_folds')}/{row.get('wf_folds')}, mean={row.get('wf_mean_return')}%, min={row.get('wf_min_return')}%")

    lines.append(
        "   Params: "
        + ", ".join(
            f"{k}={params.get(k)}"
            for k in [
                "direction_filter",
                "lookback",
                "volume_min",
                "atr_stop_mult",
                "take_profit_r",
                "max_holding_bars",
                "stop_rule",
                "adx_min",
                "regime",
                "trailing_atr_mult",
                "breakeven_r",
                "partial_tp_r",
            ]
        )
    )
    return lines


def _write_artifacts(
    *,
    symbol: str,
    stage: str,
    intervals: list[str],
    experiments: int,
    workers: int,
    output_dir: Path,
    top_df: pd.DataFrame,
    extra: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"{symbol}_{stage}_{stamp}"
    top_path = output_dir / f"{prefix}_top_candidates.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    markdown_path = output_dir / f"{prefix}_report.md"

    top_df.to_csv(top_path, index=False)
    top_candidates = top_df.to_dict(orient="records") if not top_df.empty else []

    report = {
        "symbol": symbol,
        "stage": stage,
        "intervals": intervals,
        "experiments": int(experiments),
        "workers": int(workers),
        "generated_at_utc": stamp,
        "top_candidates_path": str(top_path),
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "top_candidates": _json_safe(top_candidates),
    }
    if extra:
        report.update(_json_safe(extra))

    markdown_path.write_text(format_optimization_report(report), encoding="utf-8")
    summary_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _run_broad(
    *,
    symbol: str,
    intervals: list[str],
    experiments: int,
    workers: int,
    config_path: str,
    output_dir: Path,
    top_n: int,
    seed: int,
    wf_folds: int,
) -> dict:
    """Broad fold evaluation under baseline (normal) conditions."""
    result = evaluate_folds(
        symbol=symbol,
        intervals=intervals,
        experiments=experiments,
        workers=workers,
        top_n=top_n,
        seed=seed,
        config_path=config_path,
        output_dir=str(output_dir),
        mode="baseline",
    )
    return {
        "symbol": symbol,
        "stage": "broad",
        "intervals": intervals,
        "experiments": experiments,
        "workers": workers,
        "generated_at_utc": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "top_candidates_path": result.get("csv_path", ""),
        "summary_path": result.get("summary_path", ""),
        "markdown_path": result.get("report_path", ""),
        "survival_counts": _count_fold_survivors(result.get("df", pd.DataFrame())),
        "top_candidates": _fold_df_to_dicts(result.get("df", pd.DataFrame()), top_n),
    }


def _run_refine(
    *,
    symbol: str,
    intervals: list[str],
    experiments: int,
    workers: int,
    config_path: str,
    output_dir: Path,
    top_n: int,
    seed: int,
    wf_folds: int,
    source_top_path: str | Path,
    refine_from_top: int,
) -> dict:
    source_rows = _load_rows(source_top_path)
    focuses = build_refine_focuses(source_rows, per_candidate=refine_from_top)
    all_rows: list[dict] = []
    research_paths = []
    per_focus_experiments = max(20, experiments // max(1, len(focuses)))

    for idx, focus in enumerate(focuses):
        source_params = _load_params(source_rows[idx]) if idx < len(source_rows) else {}
        interval = source_params.get("interval") or source_rows[idx].get("interval") or intervals[0]
        if interval not in intervals:
            interval = intervals[0]
        report = run_research(
            symbol=symbol,
            interval=interval,
            max_experiments=per_focus_experiments,
            config_path=config_path,
            max_workers=workers,
            focus=focus,
            seed=seed + idx,
            wf_folds=wf_folds,
        )
        research_paths.append(report.get("path"))
        if report.get("path") and Path(report["path"]).exists():
            all_rows.extend(_load_rows(report["path"]))
        else:
            all_rows.extend(report.get("top", []))

    top_df = _rank_rows(all_rows, top_n=top_n)
    return _write_artifacts(
        symbol=symbol,
        stage="refine",
        intervals=intervals,
        experiments=experiments,
        workers=workers,
        output_dir=output_dir,
        top_df=top_df,
        extra={
            "source_top_path": str(source_top_path),
            "research_paths": research_paths,
            "refine_focus_count": len(focuses),
            "per_focus_experiments": per_focus_experiments,
        },
    )


def _run_stress(
    *,
    symbol: str,
    intervals: list[str],
    experiments: int,
    workers: int,
    output_dir: Path,
    top_n: int,
    seed: int,
    config_path: str = "config.yaml",
    source_top_path: str | Path | None = None,
) -> dict:
    """Fold evaluation under harsh stress conditions (re-runs actual backtests)."""
    result = evaluate_folds(
        symbol=symbol,
        intervals=intervals,
        experiments=experiments,
        workers=workers,
        top_n=top_n,
        seed=seed,
        config_path=config_path,
        output_dir=str(output_dir),
        mode="stress",
        source_top_path=str(source_top_path) if source_top_path else None,
    )
    return {
        "symbol": symbol,
        "stage": "stress",
        "intervals": intervals,
        "experiments": experiments,
        "workers": workers,
        "generated_at_utc": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "top_candidates_path": result.get("csv_path", ""),
        "summary_path": result.get("summary_path", ""),
        "markdown_path": result.get("report_path", ""),
        "stress_conditions": STRESS_OVERRIDES,
        "survival_counts": _count_fold_survivors(result.get("df", pd.DataFrame())),
        "top_candidates": _fold_df_to_dicts(result.get("df", pd.DataFrame()), top_n),
    }


def run_optimization(
    symbol: str = "BTCUSDT",
    intervals: Iterable[str] | None = None,
    experiments: int = 5000,
    workers: int = 4,
    stage: str = "broad",
    config_path: str = "config.yaml",
    output_dir: str | Path = "results/optimization",
    top_n: int = 50,
    seed: int = 42,
    wf_folds: int = 4,
    source_top_path: str | Path | None = None,
    refine_from_top: int = 20,
) -> dict:
    symbol = symbol.upper()
    intervals = list(intervals or ["1h"])
    output_dir = Path(output_dir)
    stage = stage.lower()

    if stage == "broad":
        return _run_broad(
            symbol=symbol, intervals=intervals, experiments=experiments,
            workers=workers, config_path=config_path, output_dir=output_dir,
            top_n=top_n, seed=seed, wf_folds=wf_folds,
        )
    if stage == "stress":
        return _run_stress(
            symbol=symbol, intervals=intervals, experiments=experiments,
            workers=workers, output_dir=output_dir, top_n=top_n,
            seed=seed + 1000, config_path=config_path,
            source_top_path=source_top_path,
        )
    if stage == "all":
        print("=" * 60)
        print("STAGE 1/2: Broad fold evaluation (baseline conditions)")
        print("=" * 60)
        broad = _run_broad(
            symbol=symbol, intervals=intervals, experiments=experiments,
            workers=workers, config_path=config_path, output_dir=output_dir,
            top_n=top_n, seed=seed, wf_folds=wf_folds,
        )
        print("\n" + "=" * 60)
        print("STAGE 2/2: Stress fold evaluation (9 harsh conditions)")
        print("=" * 60)
        stress = _run_stress(
            symbol=symbol, intervals=intervals, experiments=experiments,
            workers=workers, output_dir=output_dir, top_n=top_n,
            seed=seed + 1000, config_path=config_path,
            source_top_path=broad.get("top_candidates_path"),
        )
        # Merge summaries
        stress["broad_survival"] = broad.get("survival_counts", {})
        stress["broad_report"] = {k: v for k, v in broad.items() if k != "top_candidates"}
        return stress
    raise ValueError(f"Unknown optimization stage: {stage}")


def _count_fold_survivors(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"7_of_7": 0, "6_of_7": 0, "5_of_7": 0, "4_of_7": 0, "total": 0}
    return {
        "7_of_7": int((df["pos_folds"] >= 7).sum()),
        "6_of_7": int((df["pos_folds"] >= 6).sum()),
        "5_of_7": int((df["pos_folds"] >= 5).sum()),
        "4_of_7": int((df["pos_folds"] >= 4).sum()),
        "total": int(len(df)),
    }


def _fold_df_to_dicts(df: pd.DataFrame, top_n: int) -> list[dict]:
    if df.empty:
        return []
    return df.head(top_n).to_dict(orient="records")


def format_optimization_report(report: dict) -> str:
    lines = [
        "[investmentsystem Optimization]",
        f"Symbol: {report.get('symbol')}",
        f"Stage: {report.get('stage')}",
        f"Intervals: {', '.join(report.get('intervals', []))}",
        f"Experiments: {report.get('experiments')}",
        f"Workers: {report.get('workers')}",
        f"Top candidates CSV: {report.get('top_candidates_path')}",
        f"Summary JSON: {report.get('summary_path')}",
        f"Markdown report: {report.get('markdown_path')}",
        "",
        "[Top Candidates]",
    ]
    candidates = report.get("top_candidates") or []
    if not candidates:
        lines.append("No candidates.")
        return "\n".join(lines)

    for rank, row in enumerate(candidates[:10], start=1):
        lines.append("")
        lines.extend(_candidate_markdown(row, rank))

    return "\n".join(lines)
