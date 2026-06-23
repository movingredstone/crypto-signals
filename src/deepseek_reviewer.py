from pathlib import Path
import json
import pandas as pd

from src.deepseek_client import call_deepseek_text, try_extract_json, log_raw_response


def run_deepseek_review(symbol: str, interval: str, top_n: int = 40) -> dict:
    path = Path("results/research") / f"{symbol}_{interval}_research.csv"

    if not path.exists():
        raise FileNotFoundError(f"Research result not found: {path}")

    df = pd.read_csv(path)

    numeric_cols = [
        "train_return_pct", "train_mdd_pct", "train_pf", "train_trades",
        "val_return_pct", "val_mdd_pct", "val_pf", "val_trades",
        "test_return_pct", "test_mdd_pct", "test_pf", "test_trades",
        "test_win_rate_pct", "test_fees", "test_slippage_cost",
        "wf_folds", "wf_pos_folds", "wf_mean_return", "wf_min_return", "wf_total_trades",
        "robust_score",
    ]

    for col in ["wf_folds", "wf_pos_folds", "wf_mean_return", "wf_min_return", "wf_total_trades"]:
        if col not in df.columns:
            df[col] = 0

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["train_pf", "val_pf", "test_pf", "robust_score"]:
        if col in df.columns:
            df[col] = df[col].replace([float("inf"), -float("inf")], None)

    df["valid_sample"] = (
        (df["train_trades"] >= 30)
        & (df["val_trades"] >= 20)
        & (df["test_trades"] >= 15)
    )

    df["alpha_clue"] = (
        df["valid_sample"]
        & (df["test_return_pct"] > 0)
        & (df["test_pf"] > 1.0)
    )

    df["robust_candidate"] = (
        df["valid_sample"]
        & (df["train_return_pct"] > 0)
        & (df["val_return_pct"] > 0)
        & (df["test_return_pct"] > 0)
        & (df["test_pf"] > 1.10)
    )

    df["sort_score"] = pd.to_numeric(df["robust_score"], errors="coerce").fillna(-9999)

    selected = pd.concat(
        [
            df[df["robust_candidate"]].sort_values("sort_score", ascending=False).head(top_n),
            df[df["alpha_clue"]].sort_values("sort_score", ascending=False).head(top_n),
            df.sort_values("sort_score", ascending=False).head(top_n),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["experiment_id"]).head(top_n)

    records = []

    for _, row in selected.iterrows():
        try:
            params = json.loads(row.get("params_json", "{}"))
        except Exception:
            params = {}

        records.append(
            {
                "experiment_id": int(row.get("experiment_id", 0)),
                "family": row.get("family", ""),
                "params": params,
                "train": {
                    "return_pct": float(row.get("train_return_pct", 0)),
                    "pf": None if pd.isna(row.get("train_pf")) else float(row.get("train_pf")),
                    "mdd_pct": float(row.get("train_mdd_pct", 0)),
                    "trades": int(row.get("train_trades", 0)),
                },
                "validation": {
                    "return_pct": float(row.get("val_return_pct", 0)),
                    "pf": None if pd.isna(row.get("val_pf")) else float(row.get("val_pf")),
                    "mdd_pct": float(row.get("val_mdd_pct", 0)),
                    "trades": int(row.get("val_trades", 0)),
                },
                "test": {
                    "return_pct": float(row.get("test_return_pct", 0)),
                    "pf": None if pd.isna(row.get("test_pf")) else float(row.get("test_pf")),
                    "mdd_pct": float(row.get("test_mdd_pct", 0)),
                    "trades": int(row.get("test_trades", 0)),
                    "win_rate_pct": float(row.get("test_win_rate_pct", 0)),
                },
                "walk_forward": {
                    "folds": int(row.get("wf_folds", 0)),
                    "pos_folds": int(row.get("wf_pos_folds", 0)),
                    "mean_return_pct": float(row.get("wf_mean_return", 0)),
                    "min_return_pct": float(row.get("wf_min_return", 0)),
                },
                "robust_score": None if pd.isna(row.get("robust_score")) else float(row.get("robust_score")),
                "valid_sample": bool(row.get("valid_sample")),
                "alpha_clue": bool(row.get("alpha_clue")),
                "robust_candidate": bool(row.get("robust_candidate")),
            }
        )

    family_summary = (
        df.groupby("family")
        .agg(
            experiments=("experiment_id", "count"),
            valid_sample=("valid_sample", "sum"),
            alpha_clue=("alpha_clue", "sum"),
            robust_candidate=("robust_candidate", "sum"),
            avg_test_return=("test_return_pct", "mean"),
            avg_test_pf=("test_pf", "mean"),
            avg_test_trades=("test_trades", "mean"),
        )
        .reset_index()
        .sort_values(["robust_candidate", "alpha_clue", "valid_sample"], ascending=False)
        .to_dict(orient="records")
    )

    payload = {
        "symbol": symbol,
        "interval": interval,
        "counts": {
            "total_experiments": int(len(df)),
            "valid_sample": int(df["valid_sample"].sum()),
            "alpha_clue": int(df["alpha_clue"].sum()),
            "robust_candidate": int(df["robust_candidate"].sum()),
        },
        "top_records": records,
        "family_summary": family_summary,
    }

    system_prompt = """
You are an autonomous BTCUSDT crypto futures multi-factor strategy research director.

Your role:
- Review local backtest results.
- Separate alpha clue from robust candidate.
- Do not recommend live trading.
- Do not blindly select the highest return.
- BTCUSDT applicability is mandatory.
- Test-only positive is only an alpha clue, not a robust strategy.
- Reject too few trades and PF=inf from too few trades.
- Return JSON only.
"""

    user_prompt = f"""
Review this BTCUSDT research result.

Data:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON with this exact structure:

{{
  "overall_verdict": "string",
  "has_paper_trading_candidate": false,
  "alpha_hunt_summary": "string",
  "robust_candidate_summary": "string",
  "promising_families": [
    {{
      "family": "string",
      "reason": "string",
      "next_action": "string"
    }}
  ],
  "weak_families": [
    {{
      "family": "string",
      "reason": "string"
    }}
  ],
  "repeated_parameter_patterns": [
    "string"
  ],
  "next_telegram_commands": [
    "string"
  ],
  "next_focus": {{
    "families": ["..."],
    "direction_filters": ["..."],
    "lookbacks": [20],
    "volume_mins": [1.0],
    "atr_stop_mults": [1.0],
    "take_profit_rs": [1.5],
    "max_holding_bars": [24],
    "stop_rules": ["atr"],
    "adx_mins": [0],
    "regimes": ["any"],
    "trailing_atr_mults": [null],
    "breakeven_rs": [null],
    "partial_tp_rs": [null]
  }},
  "implementation_suggestions": [
    "string"
  ],
  "korean_summary": "string"
}}

Rules for next_focus:
- It is the next experiment design. The local engine will random-sample only
  from the values you list. Include only the keys you want to constrain.
- Prefer narrowing around families/params that are positive across walk-forward
  folds AND train/val/test, not test-only.
- A test-only positive with few trades is just an alpha clue (noise risk); do
  not over-commit to it.
"""

    def _attempt(user_msg, temperature, tokens, tag):
        # Catch BOTH API errors (HTTP 4xx -> RuntimeError) and truncated/invalid
        # JSON. Returns dict on success, None otherwise. Never raises.
        try:
            raw = call_deepseek_text(system_prompt, user_msg, temperature, tokens)
        except Exception as e:
            log_raw_response(f"{tag}_error", f"{type(e).__name__}: {e}")
            return None
        log_raw_response(tag, raw)
        return try_extract_json(raw)

    # 1st attempt — large output budget to avoid truncated JSON.
    review = _attempt(user_prompt, 0.2, 8000, "review_attempt1")

    # 2nd attempt — strict + shorter strings, smaller token budget.
    if not isinstance(review, dict):
        strict_user = (
            user_prompt
            + "\n\nYOUR PREVIOUS REPLY WAS NOT VALID JSON (likely truncated).\n"
            "Return ONLY one valid JSON object matching the schema. No prose, no "
            "markdown fences. Keep every string field short (max 2 sentences) so "
            "the whole object fits."
        )
        review = _attempt(strict_user, 0.0, 4000, "review_attempt2")

    # Degrade gracefully — never crash the flow. Local research already succeeded.
    if not isinstance(review, dict):
        review = {
            "overall_verdict": "DeepSeek 리뷰 JSON 파싱 실패(응답 잘림). 로컬 통계만 표시합니다.",
            "has_paper_trading_candidate": False,
            "alpha_hunt_summary": "",
            "robust_candidate_summary": "",
            "promising_families": [
                {
                    "family": fs.get("family"),
                    "reason": f"robust={fs.get('robust_candidate')}, alpha={fs.get('alpha_clue')}, valid={fs.get('valid_sample')}",
                    "next_action": "",
                }
                for fs in family_summary[:5]
            ],
            "weak_families": [],
            "repeated_parameter_patterns": [],
            "next_telegram_commands": [],
            "next_focus": {},
            "implementation_suggestions": [],
            "korean_summary": (
                f"총 {payload['counts']['total_experiments']}개 실험 중 "
                f"robust {payload['counts']['robust_candidate']}, "
                f"alpha {payload['counts']['alpha_clue']}, "
                f"valid {payload['counts']['valid_sample']}. "
                "DeepSeek JSON 응답이 잘려 자동 요약만 제공. "
                "raw 응답은 logs/deepseek_router_raw.log 확인."
            ),
            "_parse_failed": True,
        }

    out_dir = Path("results/deepseek")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{symbol}_{interval}_deepseek_review.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)

    review["_saved_path"] = str(out_path)

    return review


def format_deepseek_review(review: dict) -> str:
    lines = []

    lines.append("[DeepSeek Research Review]")
    lines.append(f"Saved: {review.get('_saved_path', '')}")
    lines.append("")

    lines.append("[Overall Verdict]")
    lines.append(str(review.get("overall_verdict", "")))
    lines.append("")

    lines.append("[Paper Trading Candidate]")
    lines.append(str(review.get("has_paper_trading_candidate", False)))
    lines.append("")

    lines.append("[Alpha Hunt Summary]")
    lines.append(str(review.get("alpha_hunt_summary", "")))
    lines.append("")

    lines.append("[Robust Candidate Summary]")
    lines.append(str(review.get("robust_candidate_summary", "")))
    lines.append("")

    lines.append("[Promising Families]")
    for item in review.get("promising_families", []):
        lines.append(f"- {item.get('family')}: {item.get('reason')} | next: {item.get('next_action')}")

    lines.append("")
    lines.append("[Repeated Parameter Patterns]")
    for item in review.get("repeated_parameter_patterns", []):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("[Next Telegram Commands]")
    for cmd in review.get("next_telegram_commands", []):
        lines.append(str(cmd))

    next_focus = review.get("next_focus")
    if next_focus:
        lines.append("")
        lines.append("[Next Experiment Focus (DeepSeek 설계)]")
        lines.append(json.dumps(next_focus, ensure_ascii=False))

    lines.append("")
    lines.append("[Korean Summary]")
    lines.append(str(review.get("korean_summary", "")))

    return "\n".join(lines)
