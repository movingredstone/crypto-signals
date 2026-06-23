import asyncio
import json
import re
from pathlib import Path

import pandas as pd


BTC_ONLY = True
ALLOWED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
ALLOWED_INTERVALS = {"15m", "1h", "4h"}


def research_path(symbol: str, interval: str) -> Path:
    return Path("results/research") / f"{symbol}_{interval}_research.csv"


async def send_long(update, text: str, limit: int = 3800):
    text = str(text)

    if len(text) <= limit:
        await update.message.reply_text(text)
        return

    chunks = []
    current = ""

    for line in text.splitlines():
        if len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = line
        else:
            current += ("\n" if current else "") + line

    if current:
        chunks.append(current)

    for i, chunk in enumerate(chunks, 1):
        await update.message.reply_text(f"[{i}/{len(chunks)}]\n{chunk}")


def parse_symbol(text: str) -> str:
    t = text.lower()

    if any(x in t for x in ["btcusdt", "btc", "bitcoin", "비트코인", "비트"]):
        return "BTCUSDT"
    if any(x in t for x in ["ethusdt", "eth", "ethereum", "이더리움", "이더"]):
        return "ETHUSDT"
    if any(x in t for x in ["solusdt", "sol", "솔라나"]):
        return "SOLUSDT"
    if any(x in t for x in ["bnbusdt", "bnb", "비앤비"]):
        return "BNBUSDT"

    return "BTCUSDT"


def parse_interval(text: str):
    t = text.lower().replace(" ", "")

    if any(x in t for x in ["15m", "15분", "15분봉"]):
        return "15m"
    if any(x in t for x in ["1h", "1시간", "1시간봉", "한시간", "한시간봉"]):
        return "1h"
    if any(x in t for x in ["4h", "4시간", "4시간봉", "네시간", "네시간봉"]):
        return "4h"

    return None


def parse_experiments(text: str, default: int = 1000) -> int:
    t = text.lower().replace(",", "")
    m = re.search(r"(\d{3,5})", t)

    if not m:
        return default

    n = int(m.group(1))
    return max(100, min(n, 10000))


def parse_workers(text: str, default: int = 4) -> int:
    t = text.lower()

    patterns = [
        r"workers?\s*(\d+)",
        r"worker\s*(\d+)",
        r"워커\s*(\d+)",
        r"(\d+)\s*workers?",
        r"(\d+)\s*워커",
    ]

    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return max(1, min(int(m.group(1)), 8))

    return default


def parse_rounds(text: str, default: int = 1) -> int:
    t = text.lower()
    m = re.search(r"(\d+)\s*라운드", t)

    if not m:
        m = re.search(r"rounds?\s*(\d+)", t)

    if not m:
        return default

    return max(1, min(int(m.group(1)), 5))


def wants_auto_agent(text: str) -> bool:
    t = text.lower()

    words = [
        "알아서", "자동", "자동으로", "헤르메스", "hermes",
        "멀티팩터", "multi", "factor", "조합", "전략", "전략가",
        "플러스", "+", "수익", "알파", "alpha", "찾아", "발굴", "탐색",
        "설계", "실행까지", "리뷰까지"
    ]

    return any(w in t for w in words)


def wants_review(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["딥시크", "deepseek", "리뷰", "review", "평가", "분석"])


def wants_research(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ["리서치", "research", "백테스트", "돌려", "실험"])


def force_btc(symbol: str) -> str:
    if BTC_ONLY:
        return "BTCUSDT"
    return symbol


async def run_research_local(update, symbol: str, interval: str, experiments: int, workers: int):
    symbol = force_btc(symbol)

    await update.message.reply_text(
        "[Hermes Local Research]\n"
        f"Symbol: {symbol}\n"
        f"Interval: {interval}\n"
        f"Experiments: {experiments}\n"
        f"Workers: {workers}\n\n"
        "백테스트 계산은 네 컴퓨터에서 실행합니다."
    )

    from src.research_engine import run_research, format_research_report

    report = await asyncio.to_thread(
        run_research,
        symbol=symbol,
        interval=interval,
        max_experiments=experiments,
        config_path="config.yaml",
        max_workers=workers,
    )

    text = format_research_report(report)
    await send_long(update, text)

    return report


async def ensure_research(update, symbol: str, interval: str, experiments: int, workers: int):
    symbol = force_btc(symbol)
    path = research_path(symbol, interval)

    if path.exists():
        await update.message.reply_text(
            "[Hermes]\n"
            f"기존 research 파일 발견: {path}\n"
            "새로 계산하지 않고 이 파일을 사용합니다."
        )
        return str(path)

    await update.message.reply_text(
        "[Hermes]\n"
        f"research 파일이 없습니다: {path}\n"
        "네 컴퓨터에서 자동으로 먼저 계산합니다."
    )

    await run_research_local(update, symbol, interval, experiments, workers)

    return str(path)


async def review_with_deepseek(update, symbol: str, interval: str, experiments: int, workers: int):
    symbol = force_btc(symbol)

    await ensure_research(update, symbol, interval, experiments, workers)

    await update.message.reply_text(
        "[Hermes → DeepSeek]\n"
        "계산 결과 CSV는 로컬에 있고, DeepSeek에는 요약본만 보내서 설계/판단을 맡깁니다."
    )

    from src.deepseek_reviewer import run_deepseek_review, format_deepseek_review

    review = await asyncio.to_thread(
        run_deepseek_review,
        symbol=symbol,
        interval=interval,
        top_n=50,
    )

    text = format_deepseek_review(review)
    await send_long(update, text)

    return review


def compact_research_summary(symbol: str, interval: str, top_n: int = 20) -> dict:
    path = research_path(symbol, interval)

    if not path.exists():
        return {
            "symbol": symbol,
            "interval": interval,
            "exists": False,
        }

    df = pd.read_csv(path)

    numeric_cols = [
        "train_return_pct", "val_return_pct", "test_return_pct",
        "train_pf", "val_pf", "test_pf",
        "train_trades", "val_trades", "test_trades",
        "train_mdd_pct", "val_mdd_pct", "test_mdd_pct",
        "robust_score",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
        & (df["test_pf"] > 1.1)
    )

    sort_col = "robust_score" if "robust_score" in df.columns else "test_return_pct"
    df["sort_score"] = pd.to_numeric(df[sort_col], errors="coerce").fillna(-9999)

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

        records.append({
            "experiment_id": int(row.get("experiment_id", 0)),
            "family": row.get("family", ""),
            "params": params,
            "train_return_pct": float(row.get("train_return_pct", 0)),
            "val_return_pct": float(row.get("val_return_pct", 0)),
            "test_return_pct": float(row.get("test_return_pct", 0)),
            "train_pf": None if pd.isna(row.get("train_pf")) else float(row.get("train_pf")),
            "val_pf": None if pd.isna(row.get("val_pf")) else float(row.get("val_pf")),
            "test_pf": None if pd.isna(row.get("test_pf")) else float(row.get("test_pf")),
            "train_trades": int(row.get("train_trades", 0)),
            "val_trades": int(row.get("val_trades", 0)),
            "test_trades": int(row.get("test_trades", 0)),
            "robust_score": None if pd.isna(row.get("robust_score")) else float(row.get("robust_score")),
            "alpha_clue": bool(row.get("alpha_clue")),
            "robust_candidate": bool(row.get("robust_candidate")),
        })

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

    return {
        "symbol": symbol,
        "interval": interval,
        "exists": True,
        "total_experiments": int(len(df)),
        "valid_sample": int(df["valid_sample"].sum()),
        "alpha_clue": int(df["alpha_clue"].sum()),
        "robust_candidate": int(df["robust_candidate"].sum()),
        "top_records": records,
        "family_summary": family_summary,
    }


async def ask_deepseek_for_next_plan(update, symbol: str, intervals: list[str]):
    symbol = force_btc(symbol)

    summaries = [
        compact_research_summary(symbol, interval, top_n=20)
        for interval in intervals
    ]

    system_prompt = """
You are Hermes AI, an autonomous BTCUSDT multi-factor strategy research director.

Local Python does all heavy computation.
You do not execute shell commands directly.
You only design the next allowed Telegram commands.

Goal:
- Find BTCUSDT multi-factor combinations that can produce positive net return after fees and slippage.
- Separate alpha clues from robust candidates.
- Do not recommend live trading.
- Do not overuse tokens. Use the compact summaries only.
- Return JSON only.

Allowed command format:
- /research BTCUSDT 15m N 4
- /research BTCUSDT 1h N 4
- /research BTCUSDT 4h N 4
- /deepseek_review BTCUSDT 15m N 4
- /deepseek_review BTCUSDT 1h N 4
- /deepseek_review BTCUSDT 4h N 4

Do not output commands for ETH, SOL, BNB.
Do not output shell commands.
"""

    user_prompt = f"""
Current BTCUSDT research summaries:
{json.dumps(summaries, ensure_ascii=False, indent=2)}

Decide what Hermes should do next.

Return JSON exactly:

{{
  "overall_verdict": "string",
  "paper_trading_allowed": false,
  "best_alpha_clues": ["string"],
  "weak_patterns": ["string"],
  "next_commands": [
    "/research BTCUSDT 4h 1000 4"
  ],
  "why_these_commands": "string",
  "implementation_needed": [
    "string"
  ],
  "korean_summary": "string"
}}
"""

    from src.deepseek_client import call_deepseek_json

    plan = await asyncio.to_thread(
        call_deepseek_json,
        system_prompt,
        user_prompt,
        0.2,
        3000,
    )

    out_dir = Path("results/deepseek")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{symbol}_next_plan.json"
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "[Hermes DeepSeek Next Plan]",
        f"Saved: {out_path}",
        "",
        "[Overall]",
        str(plan.get("overall_verdict", "")),
        "",
        "[Next Commands]",
    ]

    for cmd in plan.get("next_commands", []):
        lines.append(str(cmd))

    lines.extend([
        "",
        "[Why]",
        str(plan.get("why_these_commands", "")),
        "",
        "[Korean Summary]",
        str(plan.get("korean_summary", "")),
    ])

    await send_long(update, "\n".join(lines))

    return plan


def parse_allowed_command(cmd: str):
    cmd = cmd.strip()

    m = re.match(
        r"^/(research|deepseek_review)\s+(BTCUSDT)\s+(15m|1h|4h)\s+(\d{3,5})\s+(\d+)$",
        cmd,
        re.IGNORECASE,
    )

    if not m:
        return None

    action = m.group(1).lower()
    symbol = m.group(2).upper()
    interval = m.group(3)
    experiments = max(100, min(int(m.group(4)), 10000))
    workers = max(1, min(int(m.group(5)), 8))

    return action, symbol, interval, experiments, workers


async def execute_plan_commands(update, commands: list[str], max_commands: int = 2):
    executed = 0

    for cmd in commands:
        parsed = parse_allowed_command(str(cmd))

        if not parsed:
            await update.message.reply_text(
                f"[Skipped unsafe/unsupported command]\n{cmd}"
            )
            continue

        action, symbol, interval, experiments, workers = parsed

        if executed >= max_commands:
            await update.message.reply_text(
                "[Hermes]\n"
                f"자동 실행 한도 {max_commands}개에 도달했습니다. 나머지는 실행하지 않습니다."
            )
            break

        await update.message.reply_text(f"[Hermes executing]\n{cmd}")

        if action == "research":
            await run_research_local(update, symbol, interval, experiments, workers)
        elif action == "deepseek_review":
            await review_with_deepseek(update, symbol, interval, experiments, workers)

        executed += 1


async def autonomous_btc_workflow(update, text: str):
    symbol = "BTCUSDT"
    interval = parse_interval(text)
    experiments = parse_experiments(text, default=1000)
    workers = parse_workers(text, default=4)
    rounds = parse_rounds(text, default=1)

    if interval:
        intervals = [interval]
    else:
        intervals = ["15m", "1h", "4h"]

    await update.message.reply_text(
        "[Hermes Autonomous Mode]\n"
        "이제 네가 수동으로 파일 확인/리서치/리뷰 순서를 관리하지 않습니다.\n\n"
        f"Target: {symbol}\n"
        f"Intervals: {intervals}\n"
        f"Experiments: {experiments}\n"
        f"Workers: {workers}\n"
        f"Rounds: {rounds}\n\n"
        "로컬 계산은 네 컴퓨터에서 실행하고, DeepSeek는 설계와 판단만 합니다."
    )

    for r in range(1, rounds + 1):
        await update.message.reply_text(f"[Hermes Round {r}/{rounds}]")

        for iv in intervals:
            await review_with_deepseek(
                update=update,
                symbol=symbol,
                interval=iv,
                experiments=experiments,
                workers=workers,
            )

        plan = await ask_deepseek_for_next_plan(update, symbol, intervals)

        commands = plan.get("next_commands", [])

        # 사용자가 '실행까지', '알아서', '자동'이라고 했으면 DeepSeek 추천 명령도 일부 자동 실행
        if any(w in text.lower() for w in ["실행까지", "알아서", "자동", "계속"]):
            await update.message.reply_text(
                "[Hermes]\n"
                "DeepSeek가 제안한 안전한 research/review 명령을 자동 실행합니다."
            )
            await execute_plan_commands(update, commands, max_commands=2)

    await update.message.reply_text(
        "[Hermes Autonomous Workflow Completed]\n"
        "요약: 리서치 파일 확인 → 없으면 계산 → DeepSeek 리뷰 → 다음 설계까지 완료했습니다."
    )


async def handle_natural_language(update, context):
    text = update.message.text.strip()

    # 명령어 형태도 여기서 처리 가능
    if text.startswith("/"):
        return False

    if wants_auto_agent(text):
        await autonomous_btc_workflow(update, text)
        return True

    if wants_review(text):
        symbol = force_btc(parse_symbol(text))
        interval = parse_interval(text) or "1h"
        experiments = parse_experiments(text, 1000)
        workers = parse_workers(text, 4)

        await review_with_deepseek(update, symbol, interval, experiments, workers)
        return True

    if wants_research(text):
        symbol = force_btc(parse_symbol(text))
        interval = parse_interval(text)

        if interval is None:
            await autonomous_btc_workflow(update, text)
            return True

        experiments = parse_experiments(text, 1000)
        workers = parse_workers(text, 4)

        await run_research_local(update, symbol, interval, experiments, workers)
        return True

    return False
