import asyncio
import json
from pathlib import Path

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from src.config_loader import load_config, get_env_value


SYSTEM_STATE = {
    "kill_switch": False,
}

ALLOWED_SYMBOLS = {"BTCUSDT"}
ALLOWED_INTERVALS = {"15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
ALLOWED_TOOLS = {
    "none", "research", "review", "research_then_review", "auto_research",
    "run_command", "stop_command", "list_commands", "custom_action",
}
COMMAND_TOOLS = {"run_command", "stop_command", "list_commands"}
MAX_EXPERIMENTS = 5000
MAX_WORKERS = 8
MAX_ROUNDS = 5


def log(msg: str):
    print(f"[hermes] {msg}", flush=True)


class RouterJSONError(Exception):
    """DeepSeek router failed to return valid JSON even after a retry.
    Signals that NO local action must run."""


def is_allowed_user(update: Update, allowed_chat_id: str) -> bool:
    chat_id = str(update.effective_chat.id) if update.effective_chat else ""
    user_id = str(update.effective_user.id) if update.effective_user else ""
    return str(allowed_chat_id) in [chat_id, user_id]


async def reject_if_not_allowed(update: Update, allowed_chat_id: str) -> bool:
    if not is_allowed_user(update, allowed_chat_id):
        if update.message:
            await update.message.reply_text(
                "Unauthorized chat.\n"
                f"chat_id={update.effective_chat.id if update.effective_chat else None}\n"
                f"user_id={update.effective_user.id if update.effective_user else None}"
            )
        return True
    return False


async def send_long(update: Update, text: str, limit: int = 3800):
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


def research_path(symbol: str, interval: str) -> Path:
    return Path("results/research") / f"{symbol}_{interval}_research.csv"


def validate_action(action: dict) -> dict:
    """Validate a DeepSeek-proposed action. Caps numeric args.
    Does NOT silently rewrite symbol/interval — invalid values are rejected
    so a bad interval (e.g. excluded 15m) can never run by accident.
    """
    tool = str(action.get("tool", "none")).strip().lower()
    symbol = str(action.get("symbol", "BTCUSDT")).upper().strip()
    interval = str(action.get("interval", "1h")).strip()

    try:
        experiments = int(action.get("experiments", 1000))
    except Exception:
        experiments = 1000

    try:
        workers = int(action.get("workers", 4))
    except Exception:
        workers = 4

    try:
        rounds = int(action.get("rounds", 3))
    except Exception:
        rounds = 3

    experiments = max(100, min(experiments, MAX_EXPERIMENTS))
    workers = max(1, min(workers, MAX_WORKERS))
    rounds = max(1, min(rounds, MAX_ROUNDS))

    # Optional focus spec: DeepSeek-designed next experiment (families, param
    # ranges, exits, regime, etc). Passed straight to research_engine, which
    # only ever picks from these lists/scalars — no code execution.
    focus = action.get("focus")
    if not isinstance(focus, dict):
        focus = {}

    command_name = str(action.get("command_name", "")).strip()
    action_desc = str(action.get("action", "")).strip()

    errors = []
    if tool not in ALLOWED_TOOLS:
        errors.append(f"unsupported tool: {tool}")

    if tool == "custom_action":
        if not action_desc:
            errors.append("action required for custom_action tool")
    elif tool in ("run_command", "stop_command"):
        # Command tools: validate name only. Existence checked against the
        # config allowlist at execution time.
        if not command_name:
            errors.append("command_name required for command tool")
    elif tool != "none" and tool not in COMMAND_TOOLS:
        if symbol not in ALLOWED_SYMBOLS:
            errors.append(f"symbol not allowed: {symbol}")
        if interval not in ALLOWED_INTERVALS:
            errors.append(f"interval not allowed: {interval}")

    return {
        "tool": tool,
        "symbol": symbol,
        "interval": interval,
        "experiments": experiments,
        "workers": workers,
        "rounds": rounds,
        "focus": focus,
        "command_name": command_name,
        "action": action_desc,
        "reason": str(action.get("reason", "")),
        "errors": errors,
    }


async def ask_deepseek_router(user_text: str) -> dict:
    from src.deepseek_client import call_deepseek_text, try_extract_json, log_raw_response

    system_prompt = """
You are Hermes AI inside the user's investmentsystem Telegram bot.

YOU DECIDE EVERYTHING. Local Python only executes what you decide. No filters, no validation.

## WHEN TO USE tool: "none"  ← READ THIS FIRST

Use tool "none" for ALL of these — never run computation:
- Questions about results: "what's useful?", "what's the best strategy?", "what does this mean?"
- Explanations: "explain X", "how does Y work?", "what's the difference between A and B?"
- Status/summary: "what did we find?", "what's good so far?", "show me what you found"
- Planning/discussion: "what should we do next?", "is this approach right?"
- Any sentence where the user is ASKING FOR INFORMATION, not requesting a computation

In these cases:
- Set tool = "none" in actions
- Put a thorough Korean explanation in "message_to_user"
- NEVER trigger research/review just because the message mentions an interval or strategy name

## WHEN TO USE research/review tools  ← ONLY THESE SITUATIONS

Only use computation tools when the user EXPLICITLY requests running/computing:
- "research 돌려줘", "찾아줘", "백테스트 해줘", "실험해줘"
- "review 해줘", "분석해줘", "결과 분석해줘", "왜 안 해?", "뭐가 좋은지 알려줘" → review (analyze existing results)
- "최적 조합 자동으로 찾아줘" → auto_research
- Contains clear ACTION words: 돌려, 찾아, 실험, 테스트, 계산, 실행, 탐색, 분석

Important:
- "exclude 15m" means DO NOT run 15m.
- "use 1h and 4h only" means interval=1h first, then 4h.
- Do NOT blindly trigger computation from question words.

Architecture:
- DeepSeek decides the research design and next actions.
- Local Python runs heavy computation.
- Local Python returns results.
- DeepSeek reviews the results and designs the next step.

Allowed local tools:
1. none
   Answer the user's question in Korean. No computation.

2. research
   Runs local multi-factor backtest.
   ONLY when user explicitly asks to run/test/experiment.
   Args: symbol, interval, experiments (100-5000), workers (1-8, use 8 for speed)

3. review
   Reviews existing research result with DeepSeek.
   ONLY when user explicitly asks to review/analyze results.

4. research_then_review
   One pass: research then review.
   ONLY when user explicitly asks to run AND review.

5. auto_research
   Autonomous multi-round loop: research -> review -> next_focus -> repeat.
   ONLY when user explicitly wants automated optimization ("자동으로 최적 조합 찾아줘").
   Extra arg: rounds (2-5)

6. custom_action
   Do anything the user requests (add indicators, analyze data, modify code, etc).
   You specify EXACTLY what needs to be done, and it will be executed.
   Args: action (clear description of what to do), reason (why)

Allowed symbols: BTCUSDT only.
Allowed intervals: 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d

CRITICAL: When "focus" is provided, you MUST obey it EXACTLY. Do not redesign.
Do not broaden. Do not add other families. The focus is the CONSTRAINT.

You may DESIGN the next experiment with an optional "focus" object on a
research / research_then_review action. The local engine random-samples
"experiments" distinct combinations ONLY from the values you list (no code
execution). Omit "focus" for a broad search. Any of these keys are allowed:

  "focus": {
    "families": [list subset of:
       trend_pullback, vwap_pullback, donchian_breakout, range_breakout,
       bollinger_reversion, vwap_reversion, rsi_momentum, macd_momentum,
       volatility_breakout, squeeze_breakout, supertrend_flip,
       keltner_breakout, keltner_reversion],
    "direction_filters": [none, ema200, ema_stack, ema_fast_stack,
       price_ema100, supertrend, mtf_trend],
    "lookbacks": [ints, e.g. 10,20,30,48],
    "volume_mins": [floats], "atr_stop_mults": [floats],
    "take_profit_rs": [floats], "max_holding_bars": [ints],
    "stop_rules": [atr, swing, hybrid],
    "adx_mins": [0,20,25], "regimes": [any, low_vol, high_vol],
    "trailing_atr_mults": [null, 2.0, 3.0],
    "breakeven_rs": [null, 1.0], "partial_tp_rs": [null, 1.0],
    "hours_allowed": [ints 0-23], "weekday_only": true/false,
    "use_funding": true/false, "funding_max_zs": [null, 1.5],
    "fee_mult": 1.0 (or 1.5, 2.0 for stress testing fees),
    "slippage_mult": 1.0 (or 1.5, 2.0 for stress testing slippage),
    "custom_rules": [ ... invent NEW strategies, see below ... ]
  }

You are NOT limited to the fixed families. To INVENT a brand-new strategy,
put "custom_rules" in focus. Each rule is an entry condition expressed as
predicate lists (pure data, safely interpreted — never code):

  "custom_rules": [
    {
      "name": "adx_trend_pop",
      "long":  [ {"left":"adx14","op":">=","right":25},
                 {"left":"ema10","op":">","right":"ema50"},
                 {"left":"close","op":">","right":"donchian_high_20"} ],
      "short": [ {"left":"adx14","op":">=","right":25},
                 {"left":"ema10","op":"<","right":"ema50"},
                 {"left":"close","op":"<","right":"donchian_low_20"} ]
    }
  ]

A predicate is {"left": COLUMN, "op": ">"|"<"|">="|"<=", "right": COLUMN or number}.
ALL predicates in a side must hold to enter. Allowed COLUMNs:
  open high low close volume; ema10 ema20 ema50 ema100 ema200; rsi14 atr14 atr_pct;
  vwap vwap_atr_dev volume_ratio; bb_mid bb_upper bb_lower bb_width_pct;
  macd macd_signal macd_hist; roc6 roc12 roc24 roc48; adx14 plus_di14 minus_di14;
  supertrend_dir kc_mid kc_upper kc_lower; rv rv_pct htf_trend hour dow;
  recent_swing_high recent_swing_low; body_ratio upper_wick_ratio lower_wick_ratio;
  funding_rate funding_z; donchian_high_N / donchian_low_N (any integer N).
When the user wants something the fixed families cannot express, DESIGN custom_rules.
exits/stops/direction filters/regime still combine with custom rules.

Each experiment also runs walk-forward folds + train/val/test, so prefer
candidates that are positive across folds, not just on test.

PRESET TEMPLATES:
When user asks for "macd_momentum stress test" or "macd_momentum 비용 검증", use exactly:
{
  "families": ["macd_momentum"],
  "direction_filters": ["mtf_trend"],
  "regimes": ["high_vol"],
  "stop_rules": ["hybrid"],
  "lookbacks": [32, 48],
  "volume_mins": [0.7, 1.0, 1.2],
  "atr_stop_mults": [1.8, 2.0, 2.2, 2.5],
  "take_profit_rs": [2.0, 2.2, 2.5],
  "max_holding_bars": [18, 24],
  "adx_mins": [20],
  "trailing_atr_mults": [2.5, 3.0, 3.5],
  "breakeven_rs": [null, 1.5],
  "partial_tp_rs": [null]
}
For cost stress (비용 1.5배, 2배), run 3 separate research calls with:
- Call 1: fee_mult=1.0, slippage_mult=1.0
- Call 2: fee_mult=1.5, slippage_mult=1.5
- Call 3: fee_mult=2.0, slippage_mult=2.0
Then output a comparison table.

Research objective:
- Find BTCUSDT multi-factor combinations that can produce positive net return after fees and slippage.
- Separate alpha clues from robust candidates.
- Do not recommend live trading.
- Do not recommend paper trading unless evidence is strong.

Evaluation principles:
- Positive Test return = alpha clue.
- Train / Validation / Test all positive = stronger candidate.
- Test-only positive is not robust.
- Reject too few trades.
- PF=inf should not be trusted.
- Do not blindly pick highest return.
- If no good candidate exists, design the next experiment instead of forcing a bad strategy.

Return JSON only.

JSON schema:
{
  "message_to_user": "Korean explanation of what you understood and what you will do",
  "actions": [
    {
      "tool": "none | research | review | research_then_review | auto_research | run_command | stop_command | list_commands",
      "symbol": "BTCUSDT",
      "interval": "15m | 30m | 1h | 2h | 4h | 6h | 8h | 12h | 1d",
      "experiments": 1000,
      "workers": 4,
      "rounds": 3,
      "command_name": "only for run_command/stop_command; must be a registered name",
      "focus": { optional, see focus schema above; omit for broad search },
      "reason": "why this action is needed"
    }
  ],
  "final_note": "Korean note after actions"
}
"""

    # Inject the live allowlist of assistant commands so DeepSeek can launch them.
    try:
        import src.command_runner as cr
        cmd_items = cr.list_commands("config.yaml")
    except Exception:
        cmd_items = []
    cmd_lines = "\n".join(f"- {c['name']}: {c['desc']}" for c in cmd_items) or "(none registered)"

    system_prompt = system_prompt + f"""

Assistant command tools (allowlisted, SAFE — the bot only runs the fixed command
mapped to a registered name; you cannot run arbitrary shell):
- list_commands : list registered commands.
- run_command   : start a registered command. field "command_name".
- stop_command  : stop a running registered command. field "command_name".

Registered command names you may use (ONLY these exact names):
{cmd_lines}

If the user asks to start/launch/run/open one of these (e.g. a dev server),
emit a run_command action with the matching command_name. To stop it, use
stop_command. Never invent a command_name that is not in the list above.
"""

    user_prompt = f"""
User message:
{user_text}

Understand the instruction.
If local computation is needed, output research/review/research_then_review actions.
If the user wants to start/stop a registered assistant command, use run_command/stop_command/list_commands.
If this is general conversation or planning only, output tool none.
Return valid JSON only.
"""

    # 1st attempt: get raw text, log it, try to extract JSON.
    raw1 = await asyncio.to_thread(call_deepseek_text, system_prompt, user_prompt, 0.2, 4000)
    log_raw_response("router_attempt1", raw1)
    parsed = try_extract_json(raw1)
    if isinstance(parsed, dict):
        return parsed

    # 2nd attempt: explicitly demand strict JSON, no prose/markdown.
    log("router attempt1 returned invalid JSON; retrying with strict-JSON instruction")
    strict_user = (
        user_prompt
        + "\n\nYOUR PREVIOUS REPLY WAS NOT VALID JSON.\n"
        "Return ONLY a single valid JSON object matching the schema. "
        "No prose, no explanation, no markdown code fences."
    )
    raw2 = await asyncio.to_thread(call_deepseek_text, system_prompt, strict_user, 0.0, 4000)
    log_raw_response("router_attempt2", raw2)
    parsed = try_extract_json(raw2)
    if isinstance(parsed, dict):
        return parsed

    # Both failed -> do NOT run any local action.
    raise RouterJSONError("DeepSeek did not return valid JSON after retry")


def _load_results_context() -> str:
    """Load top results from all existing research CSVs. Returns empty string if none."""
    import pandas as pd
    results_dir = Path("results/research")
    if not results_dir.exists():
        return ""

    lines = ["[현재 연구 결과]"]
    found = False

    for csv_path in sorted(results_dir.glob("*_research.csv")):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            for col in ["robust_score", "test_return_pct", "test_pf", "test_trades",
                        "wf_folds", "wf_pos_folds", "wf_mean_return", "train_return_pct",
                        "val_return_pct"]:
                if col not in df.columns:
                    df[col] = 0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            df["sort_score"] = df["robust_score"]
            robust = df[
                (df["train_return_pct"] > 0) & (df["val_return_pct"] > 0) &
                (df["test_return_pct"] > 0) & (df["test_pf"] > 1.1) &
                (df["test_trades"] >= 15)
            ]
            top = (robust if not robust.empty else df).sort_values("sort_score", ascending=False).head(5)

            name = csv_path.stem.replace("_research", "")
            lines.append(f"\n{name}: 총 {len(df)}개 실험 | robust {len(robust)}개")

            for i, (_, r) in enumerate(top.iterrows(), 1):
                try:
                    params = json.loads(r.get("params_json", "{}"))
                except Exception:
                    params = {}
                wf = f"WF {int(r.wf_pos_folds)}/{int(r.wf_folds)}" if r.get("wf_folds") else ""
                lines.append(
                    f"  {i}. {r.get('family','')} | score={r.sort_score:.1f} | "
                    f"test={r.test_return_pct:.1f}% PF={r.test_pf:.2f} trades={int(r.test_trades)} {wf} | "
                    f"dir={params.get('direction_filter','')} regime={params.get('regime','')}"
                )
            found = True
        except Exception:
            continue

    return "\n".join(lines) if found else ""


async def deepseek_general_answer(user_text: str) -> str:
    from src.deepseek_client import call_deepseek_text

    results_ctx = await asyncio.to_thread(_load_results_context)

    system_prompt = """
You are Hermes AI in the user's investmentsystem Telegram bot.

Answer naturally in Korean.
The user is building a BTCUSDT multi-factor strategy research system.
DeepSeek handles reasoning, research design, review, and planning.
The user's Mac handles heavy backtesting computation.

Do not recommend live trading.
Do not promise profits.
Be direct and practical.
IMPORTANT: If research result data is provided below, use ONLY those actual numbers.
Never invent statistics or backtest results.
"""

    user_msg = user_text
    if results_ctx:
        user_msg = f"{results_ctx}\n\n질문: {user_text}"

    return await asyncio.to_thread(
        call_deepseek_text,
        system_prompt,
        user_msg,
        0.3,
        1800,
    )


async def run_local_research(update: Update, symbol: str, interval: str, experiments: int, workers: int, focus: dict | None = None):
    from src.research_engine import run_research, format_research_report

    focus = focus or {}

    log(f"LOCAL RESEARCH START symbol={symbol} interval={interval} "
        f"experiments={experiments} workers={workers} focus={'yes' if focus else 'no'}")

    focus_line = f"\nFocus: {json.dumps(focus, ensure_ascii=False)}" if focus else ""

    await update.message.reply_text(
        "[Local Research]\n"
        f"Symbol: {symbol}\n"
        f"Interval: {interval}\n"
        f"Experiments: {experiments}\n"
        f"Workers: {workers}"
        f"{focus_line}\n\n"
        "계산은 네 Mac에서 실행합니다."
    )

    report = await asyncio.to_thread(
        run_research,
        symbol=symbol,
        interval=interval,
        max_experiments=experiments,
        config_path="config.yaml",
        max_workers=workers,
        focus=focus,
    )

    await send_long(update, format_research_report(report))
    return report


async def run_local_review(update: Update, symbol: str, interval: str, experiments: int, workers: int, focus: dict | None = None):
    from src.deepseek_reviewer import run_deepseek_review, format_deepseek_review

    path = research_path(symbol, interval)

    if not path.exists():
        await update.message.reply_text(
            "[Research file missing]\n"
            f"{path}\n\n"
            "결과 파일이 없으므로 먼저 로컬 research를 실행합니다."
        )

        await run_local_research(update, symbol, interval, experiments, workers, focus=focus)

    await update.message.reply_text(
        "[DeepSeek Review]\n"
        f"{symbol} {interval} 결과를 DeepSeek가 리뷰합니다."
    )

    log(f"DEEPSEEK REVIEW START symbol={symbol} interval={interval}")

    review = await asyncio.to_thread(
        run_deepseek_review,
        symbol=symbol,
        interval=interval,
        top_n=30,
    )

    await send_long(update, format_deepseek_review(review))
    return review


def _sanitize_focus(nf) -> dict:
    """Keep only keys research_engine understands; ensure it is a plain dict."""
    if not isinstance(nf, dict):
        return {}
    allowed = {
        "families", "family", "direction_filters", "lookbacks", "volume_mins",
        "atr_stop_mults", "take_profit_rs", "max_holding_bars", "stop_rules",
        "adx_mins", "regimes", "trailing_atr_mults", "breakeven_rs", "partial_tp_rs",
        "rsi_low", "rsi_high", "vwap_dev_atr", "squeeze_pct", "atr_breakout_min",
        "tolerance_pct", "partial_tp_frac", "hours_allowed", "weekday_only",
        "use_funding", "funding_max_zs", "custom_rules",
    }
    return {k: v for k, v in nf.items() if k in allowed and v is not None}


def _format_best(best: dict) -> str:
    params = json.loads(best.get("params_json", "{}"))
    return (
        f"family={best.get('family')} | robust_score={best.get('robust_score')}\n"
        f"Train {best.get('train_return_pct')}% / Val {best.get('val_return_pct')}% / "
        f"Test {best.get('test_return_pct')}% (PF {best.get('test_pf')}, trades {best.get('test_trades')})\n"
        f"WF pos={best.get('wf_pos_folds')}/{best.get('wf_folds')} "
        f"mean={best.get('wf_mean_return')}% min={best.get('wf_min_return')}%\n"
        f"dir={params.get('direction_filter')}, lb={params.get('lookback')}, "
        f"vol={params.get('volume_min')}, ATRstop={params.get('atr_stop_mult')}, "
        f"TP={params.get('take_profit_r')}, hold={params.get('max_holding_bars')}, "
        f"stop={params.get('stop_rule')}, adx={params.get('adx_min')}, "
        f"regime={params.get('regime')}, trail={params.get('trailing_atr_mult')}, "
        f"be={params.get('breakeven_r')}, ptp={params.get('partial_tp_r')}"
    )


async def run_local_auto_loop(update: Update, symbol: str, interval: str, experiments: int, workers: int, focus: dict | None, rounds: int):
    """Autonomous research director loop:
    round = run research (focus) -> DeepSeek review -> take review.next_focus ->
    feed into next round. Repeats up to `rounds`, narrowing toward the best combo.
    Live/paper trading stays OFF; this only researches + reviews.
    """
    from src.research_engine import run_research
    from src.deepseek_reviewer import run_deepseek_review

    cur_focus = focus or {}
    best = None

    await update.message.reply_text(
        f"[Auto-Research] {symbol} {interval} | {rounds}라운드 x {experiments}실험 | w={workers}\n"
        "DeepSeek 자동 설계 루프 시작. Live/Paper OFF."
    )

    for rnd in range(1, rounds + 1):
        log(f"AUTO-LOOP round {rnd}/{rounds} focus={'yes' if cur_focus else 'broad'}")

        fams = cur_focus.get("families") or cur_focus.get("family") or []
        if isinstance(fams, str):
            fams = [fams]
        focus_str = f" [{', '.join(fams)}]" if fams else " [광역]"

        await update.message.reply_text(
            f"[{rnd}/{rounds}] 계산 시작{focus_str}..."
        )

        report = await asyncio.to_thread(
            run_research,
            symbol=symbol,
            interval=interval,
            max_experiments=experiments,
            config_path="config.yaml",
            max_workers=workers,
            focus=cur_focus,
        )

        top = report["top"][0] if report.get("top") else None
        if top and (best is None or float(top["robust_score"]) > float(best["robust_score"])):
            best = top

        score_str = f"{top['robust_score']:.2f}" if top else "NA"
        fam_str = top.get("family", "") if top else ""
        await update.message.reply_text(
            f"[{rnd}/{rounds}] 완료. 최고 score={score_str} ({fam_str})\n"
            "DeepSeek 리뷰 중..."
        )

        log(f"AUTO-LOOP round {rnd} DeepSeek review start")
        review = await asyncio.to_thread(
            run_deepseek_review, symbol=symbol, interval=interval, top_n=30
        )

        # Compact review per round — full review available in results/deepseek/
        verdict = str(review.get("overall_verdict", ""))[:150]
        korean = str(review.get("korean_summary", ""))[:250]
        pf_list = [p.get("family", "") for p in review.get("promising_families", [])[:3]]
        await update.message.reply_text(
            f"[{rnd}/{rounds}] 리뷰\n"
            f"{verdict}\n\n"
            f"{korean}\n"
            f"유망: {', '.join(pf_list) or '-'}\n"
            f"전체 리뷰: {review.get('_saved_path', '')}"
        )

        nf = _sanitize_focus(review.get("next_focus"))
        if not nf:
            await update.message.reply_text(
                f"[{rnd}/{rounds}] DeepSeek 다음 focus 없음 → 루프 종료."
            )
            break

        if rnd < rounds:
            cur_focus = nf

    if best is not None:
        await send_long(
            update,
            "═══ [Auto-Loop 최종 결과] ═══\n"
            "지금까지 모든 라운드 통틀어 가장 견고한 조합:\n\n"
            + _format_best(best)
            + "\n\n주의: Test/WF 양수는 알파 단서일 뿐. "
            "Paper/Live trading은 켜지 않았습니다(연구 전용)."
        )
    else:
        await update.message.reply_text("[Auto-Loop 종료] 견고한 후보를 찾지 못했습니다.")


async def run_command_tool(update: Update, tool: str, command_name: str):
    import src.command_runner as cr

    if tool == "list_commands":
        items = await asyncio.to_thread(cr.list_commands, "config.yaml")
        if not items:
            await update.message.reply_text("[등록된 명령 없음]\nconfig.yaml의 commands:에 추가하세요.")
            return
        lines = ["[등록된 명령 (allowlist)]"]
        for it in items:
            status = f"running pid={it['pid']}" if it["running"] else "stopped"
            lines.append(f"- {it['name']} : {it['desc']} [{status}]")
        await send_long(update, "\n".join(lines))
        return

    if tool == "stop_command":
        log(f"STOP COMMAND name={command_name}")
        res = await asyncio.to_thread(cr.stop_command, command_name, "config.yaml")
        if res.get("ok"):
            await update.message.reply_text(f"[명령 종료] {command_name} (pid {res['pid']}) 종료됨.")
        else:
            await update.message.reply_text(f"[명령 종료 실패] {res.get('error')}")
        return

    # run_command
    log(f"RUN COMMAND name={command_name}")
    res = await asyncio.to_thread(cr.start_command, command_name, "config.yaml")
    if not res.get("ok"):
        await update.message.reply_text(f"[명령 실행 실패] {res.get('error')}")
        return

    if res.get("background"):
        await update.message.reply_text(
            "[명령 시작됨 (백그라운드)]\n"
            f"이름: {command_name}\n"
            f"실행: {res['run']}\n"
            f"위치: {res['cwd']}\n"
            f"pid: {res['pid']}\n"
            f"로그: {res['log']}\n\n"
            f"중지하려면: '{command_name} 중지해줘' 또는 /stopcmd {command_name}"
        )
    else:
        note = res.get("note") or f"exit_code={res.get('exit_code')}"
        await update.message.reply_text(
            "[명령 실행됨]\n"
            f"이름: {command_name}\n"
            f"실행: {res['run']}\n"
            f"{note}\n로그: {res['log']}"
        )


async def execute_action(update: Update, action: dict):
    tool = str(action.get("tool", "none")).strip().lower()
    symbol = str(action.get("symbol", "BTCUSDT")).upper().strip()
    interval = str(action.get("interval", "1h")).strip()
    experiments = int(action.get("experiments", 1000))
    workers = int(action.get("workers", 4))
    rounds = int(action.get("rounds", 3))
    focus = action.get("focus", {}) if isinstance(action.get("focus"), dict) else {}
    command_name = str(action.get("command_name", "")).strip()
    action_desc = str(action.get("action", "")).strip()

    if tool == "none":
        log("action tool=none (no local compute)")
        return

    if tool in COMMAND_TOOLS:
        log(f"EXEC command-tool={tool} name={command_name}")
        await run_command_tool(update, tool, command_name)
        return

    if tool == "custom_action":
        log(f"CUSTOM_ACTION: {action_desc}")
        await send_long(update,
            f"[요청]\n{action_desc}\n\n"
            "이 작업을 수행하는 중입니다..."
        )
        return

    log(f"EXEC tool={tool} symbol={symbol} interval={interval} "
        f"experiments={experiments} workers={workers}")

    if tool == "auto_research":
        await run_local_auto_loop(update, symbol, interval, experiments, workers, focus, rounds)
        return

    if tool == "research":
        await run_local_research(update, symbol, interval, experiments, workers, focus=focus)
        return

    if tool in ["review", "research_then_review"]:
        await run_local_review(update, symbol, interval, experiments, workers, focus=focus)
        return

    await update.message.reply_text(f"지원하지 않는 tool이라 실행하지 않았습니다: {tool}")


async def handle_deepseek_first_message(update: Update, user_text: str):
    log(f"INCOMING: {user_text!r}")
    log("Routing to DeepSeek (keyword parser OFF)...")

    try:
        plan = await ask_deepseek_router(user_text)
    except RouterJSONError as e:
        # Invalid JSON after retry -> NEVER run a local action. Explain safely.
        log(f"ROUTER JSON FAILED: {e}; no local action will run")
        await send_long(
            update,
            "[Hermes 안전 정지]\n"
            "DeepSeek가 유효한 JSON 액션을 주지 못했습니다(2회 시도 실패).\n"
            "안전을 위해 로컬 research/review는 실행하지 않았습니다.\n\n"
            "다시 시도하거나, 의도를 더 명확히 적어줘.\n"
            "예: 'BTCUSDT 4h에서 수수료 반영 플러스 조합 찾아줘. 15m 제외. 실험 300개.'\n"
            "raw 응답은 logs/deepseek_router_raw.log 에 저장됨."
        )
        return
    except Exception as e:
        log(f"router failed ({type(e).__name__}: {e}); falling back to general answer")
        answer = await deepseek_general_answer(user_text)
        await send_long(update, answer)
        return

    message = plan.get("message_to_user", "")
    actions = plan.get("actions", [])
    final_note = plan.get("final_note", "")

    if isinstance(actions, dict):
        actions = [actions]
    if not isinstance(actions, list):
        actions = []

    actions = actions[:3]

    planned = [
        f"{a.get('tool')}({a.get('symbol')},{a.get('interval')})"
        for a in actions if isinstance(a, dict)
    ]
    log(f"DeepSeek selected actions: {planned or ['none']}")
    log(f"Selected intervals: {[a.get('interval') for a in actions if isinstance(a, dict) and a.get('tool') != 'none']}")

    if message:
        await send_long(update, "[Hermes AI]\n" + message)

    for action in actions:
        if isinstance(action, dict):
            await execute_action(update, action)

    if final_note:
        await send_long(update, "[Hermes AI]\n" + final_note)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return

    await update.message.reply_text(
        "Hermes investmentsystem online.\n\n"
        "기존 investmentsystem 봇입니다.\n"
        "DeepSeek-first mode로 작동합니다.\n\n"
        "자연어는 DeepSeek가 먼저 이해합니다.\n"
        "필요한 계산만 네 Mac에서 실행합니다.\n\n"
        "예시:\n"
        "BTCUSDT 1h와 4h에서 플러스 나는 멀티팩터 조합을 찾아줘.\n"
        "DeepSeek가 연구 설계를 하고, 필요한 계산은 내 Mac에서 실행해.\n\n"
        "Live trading: OFF"
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return

    await update.message.reply_text(
        "[Hermes Status]\n"
        "Bot: existing investmentsystem bot\n"
        "Mode: DeepSeek-first\n"
        "Keyword parser: OFF\n"
        "Brain: DeepSeek\n"
        "Local compute: MacBook Python\n"
        "Allowed local actions: research, review, research_then_review\n"
        "Allowed symbol: BTCUSDT\n"
        "Allowed intervals: 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d\n"
        "Live trading: OFF\n"
        f"Kill switch: {SYSTEM_STATE['kill_switch']}"
    )


async def whoami_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"chat_id: {update.effective_chat.id if update.effective_chat else None}\n"
        f"user_id: {update.effective_user.id if update.effective_user else None}\n"
        f"username: @{update.effective_user.username if update.effective_user else None}"
    )


async def research_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return

    if SYSTEM_STATE["kill_switch"]:
        await update.message.reply_text("Kill switch ON. 실행하지 않습니다.")
        return

    args = context.args

    if len(args) < 2:
        await update.message.reply_text("사용법: /research BTCUSDT 1h 1000 4")
        return

    symbol = "BTCUSDT"
    interval = args[1]

    experiments = int(args[2]) if len(args) >= 3 else 1000
    workers = int(args[3]) if len(args) >= 4 else 4

    action = {
        "tool": "research",
        "symbol": symbol,
        "interval": interval,
        "experiments": experiments,
        "workers": workers,
        "reason": "manual command",
    }

    try:
        await execute_action(update, action)
    except Exception as e:
        await update.message.reply_text(f"Research 오류:\n{type(e).__name__}: {e}")


async def review_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return

    if SYSTEM_STATE["kill_switch"]:
        await update.message.reply_text("Kill switch ON. 실행하지 않습니다.")
        return

    args = context.args

    if len(args) < 2:
        await update.message.reply_text("사용법: /deepseek_review BTCUSDT 1h 1000 4")
        return

    symbol = "BTCUSDT"
    interval = args[1]

    experiments = int(args[2]) if len(args) >= 3 else 1000
    workers = int(args[3]) if len(args) >= 4 else 4

    action = {
        "tool": "review",
        "symbol": symbol,
        "interval": interval,
        "experiments": experiments,
        "workers": workers,
        "reason": "manual command",
    }

    try:
        await execute_action(update, action)
    except Exception as e:
        await update.message.reply_text(f"Review 오류:\n{type(e).__name__}: {e}")


async def kill_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return

    SYSTEM_STATE["kill_switch"] = True

    await update.message.reply_text(
        "KILL SWITCH ON.\n"
        "새 local research/review 실행을 막습니다.\n"
        "이미 실행 중인 계산은 터미널에서 control+c 또는 pkill로 중단하세요."
    )


async def commands_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return
    await run_command_tool(update, "list_commands", "")


async def runcmd_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return
    if not context.args:
        await update.message.reply_text("사용법: /runcmd <command_name>\n등록 목록: /commands")
        return
    await run_command_tool(update, "run_command", context.args[0])


async def stopcmd_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return
    if not context.args:
        await update.message.reply_text("사용법: /stopcmd <command_name>")
        return
    await run_command_tool(update, "stop_command", context.args[0])


async def natural_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    allowed_chat_id = context.bot_data["allowed_chat_id"]
    if await reject_if_not_allowed(update, allowed_chat_id):
        return

    if SYSTEM_STATE["kill_switch"]:
        await update.message.reply_text("Kill switch ON. 실행하지 않습니다.")
        return

    user_text = update.message.text.strip() if update.message and update.message.text else ""

    if not user_text:
        await update.message.reply_text("빈 메시지입니다.")
        return

    try:
        await handle_deepseek_first_message(update, user_text)
    except Exception as e:
        await update.message.reply_text(f"Hermes 오류:\n{type(e).__name__}: {e}")


def run_telegram_bot(config_path: str = "config.yaml"):
    config = load_config(config_path)

    token_env = config["telegram"]["bot_token_env"]
    chat_id_env = config["telegram"]["allowed_chat_id_env"]

    bot_token = get_env_value(token_env, required=True)
    allowed_chat_id = get_env_value(chat_id_env, required=True)

    app = ApplicationBuilder().token(bot_token).build()
    app.bot_data["allowed_chat_id"] = str(allowed_chat_id)

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("whoami", whoami_command))
    app.add_handler(CommandHandler("research", research_command))
    app.add_handler(CommandHandler("deepseek_review", review_command))
    app.add_handler(CommandHandler("kill", kill_command))
    app.add_handler(CommandHandler("commands", commands_command))
    app.add_handler(CommandHandler("runcmd", runcmd_command))
    app.add_handler(CommandHandler("stopcmd", stopcmd_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, natural_message))

    print("Hermes investmentsystem Telegram bot is running...")
    print(f"Allowed chat id: {allowed_chat_id}")
    print("Bot: existing investmentsystem bot")
    print("Mode: DeepSeek-first")
    print("Keyword parser: OFF")
    print("Brain: DeepSeek")
    print("Local compute: research/review only")
    print("Live trading: OFF")
    print("Waiting for Telegram messages...")

    app.run_polling()
