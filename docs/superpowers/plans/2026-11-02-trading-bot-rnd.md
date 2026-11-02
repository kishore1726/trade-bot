# Trading Bot (R&D Phase) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python end-of-day swing-trading backtester + paper trader for NSE stocks/ETFs that reports to the owner on Telegram, with safety rules enforced in code.

**Architecture:** Pure, tested core modules (costs, risk, portfolio, strategies, metrics) driven by two entry points: `backtest` (historical) and `run-daily` (paper, idempotent with catch-up). State lives in one SQLite file. Telegram is outbound HTTP for reports plus a long-polling command process. macOS launchd schedules both.

**Tech Stack:** Python 3.12+ (machine has 3.14), venv + pip, pandas, numpy, yfinance, httpx, PyYAML, pytest. No paid services.

**Spec:** `docs/superpowers/specs/2026-11-02-trading-bot-design.md`

## Global Constraints

- No real orders, no broker API, no paid data. R&D cost ₹0.
- NSE cash, delivery, long-only. No F&O, no intraday, no shorting.
- No LLM in any decision path. All decisions deterministic.
- Starting paper capital ₹20,000; profits compound inside the portfolio.
- Max per position 25% of equity; max 4 open positions; max loss per trade 2% of equity (risk = (entry − stop) × qty); kill switch at 15% drawdown from peak equity — no new entries until owner sends `/resume`.
- Signals computed after close (~16:30 IST); paper fills at next trading day's open ± slippage.
- Stops evaluated from daily bars: if open ≤ stop → exit at open; elif low ≤ stop → exit at stop.
- Bad/stale data → no trading that day + Telegram alert. Never trade on bad data.
- Paper trading never starts until owner picks a strategy via `/choose`. A below-benchmark strategy is labelled `BELOW BENCHMARK` in every report.
- All numeric settings live in `config.yaml`; Telegram token and chat id in `.env` (git-ignored).
- All times IST (`Asia/Kolkata`).
- Owner is a non-programmer: every user-facing message is plain English, no stack traces in Telegram (log them to `logs/`).

## Review Focus

1. **Mac asleep / missed days** — `run-daily` must process every unprocessed trading day in order, exactly once, and say on Telegram which days were caught up. (Test in Task 9.)
2. **Yahoo returns partial or stale data** (today's bar missing for some tickers, NaNs, duplicated dates) — skip trading that day and alert, rather than trading on the subset. (Test in Task 3.)
3. **Running `run-daily` twice for the same day** — second run is a no-op, no duplicate fills or messages. (Test in Task 9.)
4. **Position too small to buy one share** (e.g. ₹5k cap vs a ₹6k stock) or cash short — signal skipped with a reason in the report, never a fractional or negative quantity. (Test in Task 4.)
5. **Telegram unreachable** — daily run still completes and persists; unsent messages queue in the store and send on the next run. (Test in Task 8.)

---

## File Structure

```
trading-bot/
  pyproject.toml
  config.yaml                 # all tunables (capital, limits, costs, tax, universe)
  .env.example                # TELEGRAM_TOKEN=, TELEGRAM_CHAT_ID=
  .gitignore                  # .env, data/, state/, logs/, .venv/
  README.md                   # owner-facing setup guide
  holidays/nse_2026.txt       # one ISO date per line; used only by the live daily run (backtests use bar dates)
  launchd/                    # plist templates
  tradebot/
    config.py                 # Config dataclass + load_config()
    calendar.py               # trading-day logic
    costs.py                  # Indian delivery cost model
    data.py                   # fetch/cache/validate EOD bars
    risk.py                   # sizing + kill switch
    portfolio.py              # positions, fills, stops, equity
    strategies/base.py        # Signal, Strategy protocol, registry
    strategies/trend.py       # TrendFollow
    strategies/meanrev.py     # MeanReversion
    metrics.py                # trust metrics + tax
    backtest.py               # engine + walk-forward + ranking
    store.py                  # SQLite persistence
    notify.py                 # Telegram send + outbox
    reports.py                # message text builders
    daily.py                  # paper run orchestrator
    bot.py                    # Telegram command loop
    cli.py                    # `tradebot` entry point
  tests/…                     # one test file per module
```

---

### Task 1: Project scaffold, config, calendar

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `config.yaml`, `holidays/nse_2026.txt`, `tradebot/__init__.py`, `tradebot/config.py`, `tradebot/calendar.py`
- Test: `tests/test_config.py`, `tests/test_calendar.py`

**Interfaces:**
- Produces:
  - `Config` (frozen dataclass) with fields: `capital: float`, `max_position_pct: float`, `max_positions: int`, `max_risk_pct: float`, `kill_switch_dd_pct: float`, `slippage_pct: float`, `costs: CostConfig`, `tax: TaxConfig`, `universe: list[str]`, `benchmark: str`, `data_dir: Path`, `state_path: Path`, `holidays_dir: Path`.
  - `load_config(path: Path = Path("config.yaml")) -> Config`
  - `is_trading_day(d: date, cfg: Config) -> bool`; `trading_days_between(start: date, end: date, cfg: Config) -> list[date]` (inclusive, ascending); `previous_trading_day(d: date, cfg: Config) -> date`

- [ ] **Step 1:** `git init`; create venv (`python3 -m venv .venv`), `pyproject.toml` with deps `pandas numpy yfinance httpx pyyaml` and dev dep `pytest`, console script `tradebot = tradebot.cli:main`; `pip install -e '.[dev]'`.
- [ ] **Step 2:** Write `config.yaml` with these defaults:
  `capital: 20000`, `max_position_pct: 0.25`, `max_positions: 4`, `max_risk_pct: 0.02`, `kill_switch_dd_pct: 0.15`, `slippage_pct: 0.001`, `benchmark: NIFTYBEES.NS`,
  universe = `NIFTYBEES.NS BANKBEES.NS JUNIORBEES.NS GOLDBEES.NS` + current Nifty 50 constituents with `.NS` suffix,
  costs and tax blocks per Task 2. Add a comment: *"Verify cost values against Groww's current pricing page before trusting results."*
- [ ] **Step 3:** Fill `holidays/nse_2026.txt` from NSE's published 2026 trading-holiday list (equity segment).
- [ ] **Step 4: Failing tests**
  - `test_load_config_defaults`: `load_config()` → `capital == 20000`, `max_positions == 4`, `"NIFTYBEES.NS" in universe`.
  - `test_load_config_rejects_bad_pct`: a config with `max_position_pct: 1.5` raises `ValueError`.
  - `test_weekend_not_trading_day`: `is_trading_day(date(2026,11,7))` (Saturday) is False.
  - `test_holiday_not_trading_day`: a date written to a temp holidays file is False.
  - `test_trading_days_between_skips_weekend`: Fri 2026-11-06 → Mon 2026-11-09 returns `[Nov 6, Nov 9]`.
- [ ] **Step 5:** Run `pytest tests/test_config.py tests/test_calendar.py -v` → FAIL; implement; re-run → PASS.
- [ ] **Step 6:** Commit `feat: scaffold, config, trading calendar`.

---

### Task 2: Cost and tax model

**Files:** Create `tradebot/costs.py`; Test `tests/test_costs.py`

**Interfaces:**
- Consumes: `Config.costs`, `Config.tax`
- Produces:
  - `CostConfig` fields (defaults in `config.yaml`): `brokerage_pct: 0.001`, `brokerage_max: 20.0`, `brokerage_min: 5.0`, `stt_pct: 0.001` (both sides), `exchange_pct: 0.0000297`, `sebi_per_crore: 10.0`, `stamp_buy_pct: 0.00015`, `gst_pct: 0.18` (on brokerage+exchange+sebi), `dp_charge_per_sell: 20.0`.
  - `TaxConfig` fields: `stcg_pct: 0.20`, `ltcg_pct: 0.125`, `ltcg_exempt_per_year: 125000`.
  - `trade_cost(side: Literal["BUY","SELL"], value: float, cc: CostConfig) -> float`
  - `estimate_tax_stcg(net_gain: float, tc: TaxConfig) -> float` (0 if gain ≤ 0)
  - `estimate_tax_ltcg(net_gain: float, tc: TaxConfig) -> float` (applies exemption)

- [ ] **Step 1: Failing tests**
  - `test_buy_5000`: `trade_cost("BUY", 5000, default)` ≈ brokerage 5.0 + STT 5.0 + exch 0.1485 + sebi 0.005 + stamp 0.75 + GST 0.18×(5.0+0.1485+0.005) → assert `pytest.approx(11.83, abs=0.02)`.
  - `test_sell_includes_dp`: `trade_cost("SELL", 5000)` − (same without DP) == 20.0; no stamp duty on sell.
  - `test_brokerage_capped`: on ₹1,00,000 brokerage component is 20.0.
  - `test_stcg_zero_on_loss`: `estimate_tax_stcg(-100)` == 0.
  - `test_ltcg_exemption`: `estimate_tax_ltcg(130000)` == 0.125 × 5000.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Commit `feat: Indian delivery cost and tax model`.

---

### Task 3: Market data fetch, cache, validate

**Files:** Create `tradebot/data.py`; Test `tests/test_data.py`

**Interfaces:**
- Consumes: `Config`, `calendar.previous_trading_day`
- Produces:
  - `Bars = dict[str, pd.DataFrame]` — per ticker, DatetimeIndex (date, ascending, unique), columns `open high low close volume` (split/dividend-adjusted).
  - `fetch_bars(tickers: list[str], start: date, end: date, cfg: Config, source: Callable | None = None) -> Bars` — uses yfinance by default (`auto_adjust=True`); caches each ticker as CSV in `cfg.data_dir`, only downloading missing ranges. `source` is injectable for tests.
  - `validate_day(bars: Bars, d: date) -> list[str]` — returns problems (empty = OK). Problems: ticker missing bar for `d`; any NaN in OHLC on `d`; `low > min(open,close)` or `high < max(open,close)`; duplicate index dates.

- [ ] **Step 1: Failing tests** (use a fake `source`, no network)
  - `test_fetch_uses_cache`: second `fetch_bars` call for same range does not call `source`.
  - `test_fetch_dedupes_and_sorts`: source returning shuffled duplicated rows → result index unique and ascending.
  - `test_validate_ok`: clean bars → `[]`.
  - `test_validate_missing_ticker_bar`: one of two tickers lacks day `d` → one problem naming that ticker.
  - `test_validate_nan`: NaN close on `d` → problem.
  - `test_validate_bad_ohlc`: low 105, open 100 → problem.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Manual smoke: `python -c "from tradebot.data import fetch_bars; ..."` for `NIFTYBEES.NS` last 30 days → prints a non-empty frame.
- [ ] **Step 4:** Commit `feat: EOD data fetch with cache and validation`.

---

### Task 4: Risk sizing, portfolio, fills and stops

**Files:** Create `tradebot/risk.py`, `tradebot/portfolio.py`; Test `tests/test_risk.py`, `tests/test_portfolio.py`

**Interfaces:**
- Consumes: `Config`, `costs.trade_cost`
- Produces:
  - `size_position(equity: float, cash: float, entry: float, stop: float, open_count: int, cfg: Config) -> tuple[int, str | None]` — qty = floor(min(equity×max_position_pct, equity×max_risk_pct/(entry−stop), cash_after_costs)/entry); returns `(0, reason)` when qty < 1, `open_count >= max_positions`, or `stop >= entry`.
  - `kill_switch_tripped(equity: float, peak: float, cfg: Config) -> bool` — `equity <= peak × (1 − kill_switch_dd_pct)`.
  - `@dataclass Position(ticker, qty, entry_price, entry_date, stop, strategy)`
  - `@dataclass Trade(ticker, qty, entry_date, entry_price, exit_date, exit_price, costs, pnl, reason)` (`reason` ∈ `"stop" | "signal"`)
  - `class Portfolio(cash: float)` with `positions: dict[str, Position]`, `trades: list[Trade]`, `peak_equity: float`:
    - `buy(ticker, qty, open_price, d, stop, strategy, cfg) -> None` (fill = open × (1+slippage), cost deducted)
    - `sell(ticker, price, d, reason, cfg) -> Trade` (fill = price × (1−slippage))
    - `apply_stops(d: date, bars: Bars, cfg) -> list[Trade]` — rule from Global Constraints; includes positions entered today.
    - `equity(d: date, bars: Bars) -> float` (cash + qty × close); `update_peak(equity)`
    - `to_dict() / from_dict()` for persistence.

- [ ] **Step 1: Failing tests**
  - `test_size_risk_limited`: equity 20000, entry 100, stop 90 → risk cap 400/10 = 40 sh, position cap 5000/100 = 50 → qty 40.
  - `test_size_position_limited`: entry 100, stop 99 → qty 50.
  - `test_size_too_expensive`: entry 6000, stop 5800 → `(0, reason)` with "one share" in reason.
  - `test_size_max_positions`: open_count 4 → `(0, reason)`.
  - `test_size_bad_stop`: stop ≥ entry → `(0, reason)`.
  - `test_kill_switch`: peak 20000, equity 17000 → True; 17001 → False.
  - `test_gap_down_exits_at_open`: stop 95, bar open 90 low 88 → exit price 90×(1−slip), reason "stop".
  - `test_intraday_stop_exits_at_stop`: stop 95, open 98 low 94 → exit 95×(1−slip).
  - `test_no_stop_hit`: low 96 → no trade.
  - `test_buy_reduces_cash_by_value_plus_cost`.
  - `test_roundtrip_dict`: `Portfolio.from_dict(p.to_dict())` equal.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Commit `feat: risk sizing, portfolio, stop handling`.

---

### Task 5: Strategy interface and two strategies

**Files:** Create `tradebot/strategies/__init__.py`, `base.py`, `trend.py`, `meanrev.py`; Test `tests/test_strategies.py`

**Interfaces:**
- Consumes: `Bars`, `Position`
- Produces:
  - `@dataclass Signal(ticker: str, action: Literal["BUY","SELL"], stop: float | None, reason: str)` — `reason` is plain English (shown on Telegram).
  - `class Strategy(Protocol)`: `name: str`; `lookback_days: int`; `signals(bars: Bars, d: date, positions: dict[str, Position]) -> list[Signal]` — uses only rows with index ≤ d.
  - `REGISTRY: dict[str, type[Strategy]]` with keys `"trend"`, `"meanrev"`.
  - `TrendFollow` (`name="trend"`): BUY when close > SMA(100) and 63-day return ranks in top 4 of universe and > 0; stop = entry_ref − 2×ATR(20) using d's close as entry_ref; SELL when close < SMA(100).
  - `MeanReversion` (`name="meanrev"`): BUY when close > SMA(200) and RSI(2) < 10; stop = close − 2×ATR(20); SELL when close > SMA(5) or held ≥ 10 trading days.
  - BUY signals are never emitted for tickers already held.

- [ ] **Step 1: Failing tests** (synthetic bars)
  - `test_no_lookahead`: for each strategy, signals at `d` are identical whether bars include rows after `d` or not.
  - `test_trend_buys_uptrend` / `test_trend_sells_below_sma`.
  - `test_meanrev_buys_dip_in_uptrend` / `test_meanrev_time_exit_after_10_days`.
  - `test_buy_has_stop_below_close`.
  - `test_no_buy_when_held`.
  - `test_insufficient_history_no_signal`: fewer rows than `lookback_days` → `[]`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Commit `feat: strategy interface, trend and mean-reversion`.

---

### Task 6: Metrics, backtest engine, walk-forward ranking

**Files:** Create `tradebot/metrics.py`, `tradebot/backtest.py`; Test `tests/test_metrics.py`, `tests/test_backtest.py`

**Interfaces:**
- Consumes: Tasks 1–5
- Produces:
  - `@dataclass Metrics(trades: int, win_rate: float, avg_win: float, avg_loss: float, expectancy: float, max_drawdown: float, total_return: float, cagr: float, benchmark_return: float, benchmark_cagr: float, total_costs: float, post_tax_return: float, benchmark_post_tax_return: float, below_benchmark: bool)`
  - `compute_metrics(trades: list[Trade], equity: pd.Series, benchmark_close: pd.Series, start_capital: float, cfg: Config) -> Metrics` — strategy tax via STCG on net yearly gains; benchmark via LTCG if held > 1 year else STCG.
  - `run_backtest(strategy: Strategy, bars: Bars, start: date, end: date, cfg: Config) -> tuple[Metrics, pd.Series, list[Trade]]` — day loop: (1) fill yesterday's pending BUY/SELL at today's open, (2) apply stops, (3) record equity, update peak, check kill switch (if tripped: no new BUYs for rest of run), (4) compute signals at today's close → pending for tomorrow. Same order as `daily.py`.
  - `walk_forward(strategy_names: list[str], bars, cfg, holdout_years: int = 2) -> dict[str, dict[Literal["in_sample","out_of_sample"], Metrics]]`
  - `rank(results) -> list[str]` — order: out-of-sample `max_drawdown <= kill_switch_dd_pct` first, then higher `expectancy`, then higher `total_return − benchmark_return`.

- [ ] **Step 1: Failing tests**
  - `test_win_rate_and_expectancy`: trades pnl [+300, −100, −100] → win_rate 1/3, avg_win 300, avg_loss −100, expectancy 33.33.
  - `test_max_drawdown`: equity [100,120,90,130] → 0.25.
  - `test_below_benchmark_flag`.
  - `test_backtest_fills_next_open`: a signal on day t fills at day t+1 open (synthetic bars, stub strategy).
  - `test_backtest_costs_reduce_equity`: one round trip with flat prices ends below starting capital by exactly the costs.
  - `test_backtest_kill_switch_blocks_new_buys`.
  - `test_rank_order`: three fake results ordered per the rule above.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Smoke: run both strategies on 2016-01-01 → today on real data; print metrics table. Record the output in the commit message body.
- [ ] **Step 4:** Commit `feat: metrics, backtester, walk-forward ranking`.

---

### Task 7: State store

**Files:** Create `tradebot/store.py`; Test `tests/test_store.py`

**Interfaces:**
- Produces `class Store(path: Path)` (SQLite, created on first use):
  - `get(key: str, default=None) -> Any` / `set(key: str, value: Any)` — JSON values. Keys used: `"portfolio"`, `"pending_signals"`, `"last_processed_day"`, `"chosen_strategy"`, `"paused"`, `"kill_switch"`, `"backtest_summary"`.
  - `log_event(kind: str, payload: dict)` — append-only trade/approval/error log with timestamp.
  - `outbox_add(text: str)`, `outbox_pending() -> list[tuple[int,str]]`, `outbox_mark_sent(id: int)`.
  - `transaction()` context manager — all writes of one daily run commit together or not at all.

- [ ] **Step 1: Failing tests:** `test_get_set_roundtrip`, `test_persists_across_instances`, `test_transaction_rollback_on_error`, `test_outbox_order_and_mark_sent`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Commit `feat: SQLite state store with outbox`.

---

### Task 8: Telegram notifications and report text

**Files:** Create `tradebot/notify.py`, `tradebot/reports.py`; Test `tests/test_notify.py`, `tests/test_reports.py`

**Interfaces:**
- Consumes: `Store`, `Metrics`, `Portfolio`, `Signal`, `Trade`
- Produces:
  - `send(text: str, store: Store, http: httpx.Client | None = None) -> bool` — adds to outbox, then flushes the whole outbox in order via `POST https://api.telegram.org/bot{TOKEN}/sendMessage`; network failure → returns False, messages stay queued; never raises.
  - `daily_summary(d, portfolio, equity, signals, exits, caught_up_days, strategy_name, below_benchmark) -> str`
  - `weekly_report(metrics: Metrics, strategy_name: str, backtest_oos: Metrics | None) -> str` — when `backtest_oos` given, adds a "Paper vs backtest" line comparing win rate, expectancy and drawdown (spec §8).
  - `backtest_comparison(results, ranking) -> str` — ends with "Reply /choose <name> to start paper trading."
  - `alert(text: str) -> str` (prefix "⚠️ ").
  - Any report for a below-benchmark strategy contains the literal `BELOW BENCHMARK`.

- [ ] **Step 1: Failing tests**
  - `test_send_queues_on_network_error` (mock transport raising) → returns False, `outbox_pending()` has 1.
  - `test_send_flushes_backlog_in_order`.
  - `test_daily_summary_lists_signal_with_stop_and_reason`.
  - `test_reports_label_below_benchmark`.
  - `test_weekly_report_has_win_rate_avg_win_loss_drawdown`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Commit `feat: telegram outbox and report text`.

---

### Task 9: Daily paper run with catch-up

**Files:** Create `tradebot/daily.py`; Test `tests/test_daily.py`

**Interfaces:**
- Consumes: everything above.
- Produces `run_daily(today: date, cfg: Config, store: Store, fetch=fetch_bars, sender=send) -> list[date]` (days processed):
  1. If `chosen_strategy` unset → send one reminder alert, return `[]`.
  2. Days = `trading_days_between(last_processed_day + 1, today)`; exclude `today` if now IST < 16:15.
  3. For each day in order, inside one `store.transaction()`: fetch, `validate_day`; if problems → alert listing them, stop processing (do not advance `last_processed_day`). Else same step order as backtest: fill pending at open (skip BUYs if paused/kill switch, record skip reason), apply stops, equity/peak, kill-switch check (trip → set flags + alert), compute signals → `pending_signals`, set `last_processed_day`.
  4. Send one `daily_summary` for the last processed day mentioning caught-up days; on Saturdays (or first run after) also send `weekly_report` of the paper record.
  5. Any unexpected exception → logged to `logs/`, plain-English alert, state unchanged.

- [ ] **Step 1: Failing tests** (fake fetch + fake sender, temp store)
  - `test_requires_chosen_strategy`.
  - `test_catch_up_processes_missed_days_in_order`: last processed Mon, today Thu → processes Tue, Wed, Thu; summary mentions 2 caught-up days.
  - `test_idempotent_same_day`: second call → `[]`, no new fills, no message.
  - `test_bad_data_halts_and_alerts`: validate problem on Wed → Tue processed, Wed/Thu not, alert sent, `last_processed_day` == Tue.
  - `test_paused_skips_buys_but_runs_stops`.
  - `test_kill_switch_trips_and_alerts`.
  - `test_matches_backtest`: over 30 synthetic days, `run_daily` day-by-day produces the same trades as `run_backtest`.
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Commit `feat: daily paper run with catch-up`.

---

### Task 10: Telegram command bot

**Files:** Create `tradebot/bot.py`; Test `tests/test_bot.py`

**Interfaces:**
- Produces `handle_command(text: str, chat_id: int, cfg: Config, store: Store) -> str` (pure, tested) and `run_bot(cfg, store)` (long-polls `getUpdates`, calls `handle_command`, replies; ignores chats other than `TELEGRAM_CHAT_ID`).
- Commands: `/status` (cash, equity, positions with stops, paused/kill state, chosen strategy), `/report` (latest weekly metrics), `/pause`, `/resume` (clears `paused` and `kill_switch`, resets peak to current equity, logs event), `/choose <name>` (only names in `backtest_summary`; sets `chosen_strategy`, initialises portfolio with `capital` if none), `/help`. Unknown → help text.

- [ ] **Step 1: Failing tests:** `test_status_shows_positions`, `test_pause_resume_flags`, `test_resume_resets_peak`, `test_choose_rejects_unknown`, `test_choose_initialises_portfolio`, `test_ignores_other_chat` (in `run_bot` dispatch helper).
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** Commit `feat: telegram commands`.

---

### Task 11: CLI, scheduling, owner README

**Files:** Create `tradebot/cli.py`, `launchd/com.tradebot.daily.plist`, `launchd/com.tradebot.bot.plist`, `README.md`; Test `tests/test_cli.py`

**Interfaces:**
- `tradebot backtest [--start 2016-01-01]` → walk-forward both strategies, store `backtest_summary`, send `backtest_comparison`.
- `tradebot run-daily` → `run_daily(today_ist, …)`.
- `tradebot bot` → `run_bot`.
- `tradebot test-telegram` → sends "Hello from your trading bot ✅".
- `tradebot install-schedule` → writes plists to `~/Library/LaunchAgents` with absolute venv paths and `launchctl load`s them. Daily plist: `StartCalendarInterval` 16:30 Mon–Fri **and** `RunAtLoad` + hourly interval so a sleeping Mac catches up (idempotency makes extra runs harmless). Bot plist: `KeepAlive true`.

- [ ] **Step 1: Failing tests:** `test_cli_parses_commands` (each subcommand dispatches to a mocked function); `test_install_schedule_writes_absolute_paths` (to a temp dir).
- [ ] **Step 2:** Run → FAIL; implement; run → PASS.
- [ ] **Step 3:** README for a non-programmer: create bot with @BotFather, get chat id, copy `.env.example` → `.env`, run four commands (`pip install`, `tradebot test-telegram`, `tradebot backtest`, `tradebot install-schedule`), what each Telegram message means, how to `/pause`. Note survivorship bias (today's Nifty 50 list used for history) and that costs must be checked against Groww's pricing.
- [ ] **Step 4:** Full suite `pytest -v` → all PASS. End-to-end: `tradebot test-telegram` delivers; `tradebot backtest` delivers a comparison.
- [ ] **Step 5:** Commit `feat: CLI, launchd scheduling, README`.
