# Prompt for Independent Verification Model

다음 GitHub repo의 페이퍼 트레이딩 시스템을 독립 검증해줘:

https://github.com/movingredstone/crypto-signals

반드시 먼저 이 문서를 읽어:
https://github.com/movingredstone/crypto-signals/blob/main/HANDOFF.md

검증 목표:
우리가 백테스트/워크포워드/스트레스 테스트로 선택한 전략 파라미터와 백테스트 엔진의 알고리즘 로직이 `paper_trader_github.py`에 정확히 옮겨졌는지 확인해줘. 여기서 “트레인했던 데이터”를 그대로 코드에 넣는다는 뜻이 아니라, 학습/검증 과정에서 선택된 `전략 family + interval + parameter set + entry/exit/risk logic`이 실전 페이퍼 트레이딩 코드에 동일하게 구현됐는지를 확인하는 것이다.

검증할 파일:
1. `paper_trader_github.py` — GitHub Actions에서 실제 실행되는 페이퍼 트레이더
2. `src/research_engine.py` — 백테스트/최적화 엔진의 source of truth
3. `src/indicators.py` — indicator 계산 source of truth
4. `HANDOFF.md` — 현재 시스템 설명 및 검증 체크리스트
5. `.github/workflows/signal.yml` — 4시간 자동 실행 workflow

확인해야 할 전략 3개:

1. DOGEUSDT macd_momentum/8h
- allocation: 110
- risk: 5%
- direction_filter: price_ema100
- lookback: 48
- volume_min: 1.2
- atr_stop_mult: 2.0
- take_profit_r: 3.0
- max_holding_bars: 12
- stop_rule: swing
- adx_min: 20
- regime: low_vol
- breakeven_r: 1.0
- partial_tp_r: 1.0
- partial_tp_frac: 0.5

2. SUIUSDT macd_momentum/4h
- allocation: 110
- risk: 5%
- direction_filter: ema_fast_stack
- lookback: 48
- volume_min: 2.0
- atr_stop_mult: 3.0
- take_profit_r: 4.0
- max_holding_bars: 24
- stop_rule: swing
- adx_min: 20
- regime: any
- partial_tp_r: 1.0
- partial_tp_frac: 0.5

3. AVAXUSDT trend_pullback/8h
- allocation: 110
- risk: 5%
- direction_filter: price_ema100
- lookback: 48
- volume_min: 1.2
- atr_stop_mult: 2.0
- take_profit_r: 5.0
- max_holding_bars: 12
- stop_rule: swing
- adx_min: 0
- regime: low_vol
- pullback_ref: ema20
- partial_tp_frac: 0.5
- tolerance_pct: 0.006

검증 포인트:

A. 파라미터 일치 여부
- `paper_trader_github.py`의 `STRATEGIES`가 위 값과 정확히 일치하는지 확인해.
- 특히 SUI의 `ema_fast_stack`은 백테스트 엔진 기준으로 `ema10 > ema20 > ema50`이어야 한다.
- 특히 AVAX는 `direction_filter=price_ema100`, `stop_rule=swing`인지 확인해.

B. 시그널 로직 일치 여부
`paper_trader_github.py`가 `src/research_engine.py`와 아래 로직을 동일하게 구현했는지 확인해:

1. `entry_trigger()` / `check_entry()`
- macd_momentum LONG: `macd_hist > 0 and macd_hist > prev_macd_hist`
- macd_momentum SHORT: `macd_hist < 0 and macd_hist < prev_macd_hist`
- trend_pullback: `near(close, row[pullback_ref], tolerance_pct)`

2. `direction_allowed()`
- none, ema200, ema_stack, ema_fast_stack, price_ema100, supertrend, mtf_trend 동작이 `research_engine.py`와 일치하는지 확인해.
- 단, 현재 3전략은 price_ema100과 ema_fast_stack만 실제로 사용한다.

3. `confirmation_ok()`
- volume_ratio >= volume_min
- atr_pct_min / atr_pct_max 필터
- adx_min 필터
- regime 필터는 `rv_pct` 기준이어야 한다. low_vol이면 `rv_pct <= 50`, high_vol이면 `rv_pct >= 50`.

4. `make_stop_take()`
- stop_rule='swing'이면 LONG은 `recent_swing_low`, SHORT는 `recent_swing_high`를 써야 한다.
- TP는 `take_profit_r × actual risk distance`로 계산되어야 한다.

C. Indicator 계산 일치 여부
`paper_trader_github.py.compute_indicators()`가 `src/indicators.py.add_indicators()` + `src/research_engine.py.enrich_features()`와 필요한 컬럼을 동일하게 계산하는지 확인해:
- ema10, ema20, ema50, ema100, ema200
- atr14 with Wilder smoothing
- atr_pct
- rv_pct
- macd_hist with min_periods
- rsi14 with Wilder smoothing
- adx14 with Wilder smoothing
- volume_ratio
- recent_swing_high / recent_swing_low shifted by 1 bar to avoid lookahead

D. 리스크/페이퍼 트레이딩 동작
- position sizing formula: `allocation * risk_pct / (abs(entry-stop)/entry)`
- max_holding_bars가 적용되는지
- breakeven_r가 DOGE에만 적용되는지
- partial_tp_r/partial_tp_frac가 파라미터에는 있으나 실제 fill simulation에 반영되는지 확인해. 반영 안 되어 있으면 “known missing feature”로 보고해.
- 수수료/슬리피지 반영 여부 확인. 반영 안 되어 있으면 페이퍼 결과가 낙관적일 수 있다고 지적해.

E. GitHub Actions
- workflow가 `paper_trader_github.py`를 실행하는지 확인해. 이전 legacy `paper_signal.py`는 사용하면 안 됨.
- `permissions: contents: write`가 있어서 `paper_state.json`, `paper_trades.csv`를 commit할 수 있는지 확인해.
- TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, GITHUB_TOKEN 환경변수가 올바르게 전달되는지 확인해.

결과 보고 형식:
1. PASS/FAIL 요약
2. 정확히 일치하는 부분
3. 불일치/버그/누락된 기능
4. “페이퍼 트레이딩으로 돌려도 되는가?”에 대한 위험 평가
5. 반드시 수정해야 할 항목과 선택 수정 항목

중요:
- 실제 학습 데이터 전체를 코드에 옮기는 것이 목적이 아니다. 목적은 학습/검증에서 선택된 전략 파라미터와 백테스트 엔진 로직을 실시간 페이퍼 트레이더에 동일하게 옮기는 것이다.
- CSV 결과 파일이 repo에 없으면, `HANDOFF.md`, `STRATEGIES`, `src/research_engine.py`, `src/indicators.py` 기준으로 검증하고, “원본 stress CSV의 params_json은 별도 확인 필요”라고 명시해.
