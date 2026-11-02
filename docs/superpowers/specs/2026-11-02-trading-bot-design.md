# Trading Bot — Design (R&D Phase)

Date: 2026-11-02
Owner: the user (Groww customer, India)

## 1. Intent and the why

The owner wants a **slow, steady secondary income** from the Indian stock market,
built on a system they can **trust**. The long-term goal is ₹25k–40k/month, reached by
reinvesting profits and growing capital over time — not from the starting ₹20k.

Right now the project is **R&D**. Success is a system that runs correctly every day and
produces an honest, measurable track record. Profit is not the yardstick yet.

The owner is not a programmer. They will **run it and read Telegram reports**, not edit code.
Everything they need to see must reach them on Telegram; everything they need to do
must be one command or one button tap.

## 2. Goals

1. **Backtester**: test rule-based strategies on historical NSE daily data, with realistic
   Indian delivery costs, and report trust metrics (below).
2. **Paper trader**: run the chosen strategy every trading day on live (end-of-day) data,
   simulate fills, keep a paper portfolio of ₹20k, no real money.
3. **Telegram bot**: deliver daily signals, daily paper P&L, weekly reports, and error alerts.
4. **Safety rules** enforced in code from day one (section 6), in paper mode too.
5. Leave a clean seam for a later **live phase** (section 9) without building it now.

## 3. Non-goals (this phase)

- No real orders, no Groww API, no paid data, no static IP / VPS. Cost of R&D phase: ₹0.
- No F&O (futures/options). No intraday trading. No short selling.
- No LLM (Claude or otherwise) making trade decisions. Decisions are deterministic rules.
- No dashboards/web UI. Telegram is the only interface.
- No orchestration across multiple Claude accounts.

## 4. Anti-goals (failure even if it "works")

- A backtest that looks great but ignores costs, uses future data (look-ahead bias),
  or is over-tuned to the past.
- Paper results that silently diverge from what live trading would do (e.g. filling at
  prices that weren't available).
- The bot failing silently — a missed day or crashed job without a Telegram alert.
- Any single loss or day that could wipe out a large chunk of capital.
- Reporting only "win rate" in a way that hides losing expectancy.

## 5. Trading scope (decided)

- **Instruments**: NSE cash segment, delivery only. Universe = liquid ETFs
  (e.g. NIFTYBEES, BANKBEES, JUNIORBEES, GOLDBEES) + Nifty 50 stocks.
- **Style**: swing/positional — holds days to weeks. Signals computed **once per day after
  market close**; paper fills at the **next day's open**.
- **Capital**: ₹20,000 paper. Profits stay in the paper portfolio (compounding).

## 6. Safety rules (decided; exact numbers are builder defaults, configurable)

- Max per position: 25% of capital (≈ ₹5k). Max open positions: 4.
- Every entry has a stop-loss; max loss per trade ≈ 2% of capital.
- **Kill switch**: if portfolio drawdown from peak exceeds 15%, stop opening new trades
  and send a Telegram alert. Resuming needs an explicit owner command.
- If data is missing/stale or a check fails, do nothing that day and alert — never trade on
  bad data.

## 7. Strategies

The builder chooses **two simple, well-known, rule-based strategies** to start (e.g. a
trend/momentum rule and a mean-reversion rule on large caps/ETFs), implemented behind a
common interface so more can be added. Each is backtested and compared against a
**buy-and-hold NIFTYBEES benchmark**. The one with the better trust metrics on
out-of-sample data runs in paper mode (both may run side by side in paper if cheap).

**Checkpoint (my call):** paper trading does not start automatically. The backtest
comparison (both strategies vs benchmark, section 8 metrics) is sent to the owner on
Telegram, and the owner confirms which strategy goes to paper. Ranking order when
comparing: max drawdown within limit first, then expectancy after costs, then return vs
benchmark.

**If no strategy beats the benchmark** (owner deferred to my recommendation): still
paper-trade the better one — paper is free and it validates the whole pipeline live — and
label it clearly in every report as "below benchmark". In parallel, keep trying new
strategies in the backtester; strategies are pluggable, so a better one can replace it in
paper without other changes. Real money is never considered for a below-benchmark strategy.

Guard against over-fitting: few parameters, test on a period not used for tuning
(walk-forward or a held-out recent period), report both.

## 8. Metrics (trust metrics) — reported in backtest and weekly Telegram report

- Number of trades, **win rate**, **average win vs average loss**, expectancy per trade
- **Max drawdown**, total and annualised return, vs NIFTYBEES buy-and-hold
- Total costs paid
- Returns shown **pre-tax and estimated post-tax** (my call): strategy trades taxed as
  short-term gains (20%); benchmark as long-term (12.5% above ₹1.25L/yr exemption)
- Paper phase: how paper fills compared to backtest assumptions

## 9. Daily flow

| When | What happens |
|---|---|
| ~4:30 pm IST, trading days | Fetch day's EOD data → validate → run strategy → update paper portfolio → Telegram summary (signals for tomorrow with stock, amount, entry, stop-loss, reason; today's P&L) |
| Next open | Paper fills at open price (+ slippage assumption) |
| ~4:30 pm run | Stop-losses checked against the day's low (gap-down = exit at open, else at stop price); stop-hit alerts arrive in the evening summary |
| Any time | Telegram alert on kill switch, or any error/missed run |

*My call:* no intraday monitoring in R&D. Stops are evaluated from end-of-day data, which
mirrors a broker-held stop order (e.g. Groww GTT) in the live phase, so paper and live use
the same stop model.
| Weekly (e.g. Sat) | Telegram weekly report with section 8 metrics |

Must respect NSE trading holidays.

## 10. Costs to model (builder verifies current Groww/NSE rates)

Groww delivery brokerage, STT (0.1% buy and sell on delivery), exchange transaction
charges, SEBI fee, stamp duty (buy), GST, **DP charge per sell** (material at ₹5k position
sizes), plus a slippage assumption. Model conservatively.

## 11. Where it runs (my call)

On the owner's Mac for the R&D phase, scheduled automatically after market close.
If the Mac was asleep, the job catches up when it wakes and still sends the report;
if a day is missed, Telegram says so. Moving to a cloud server is deferred to the live phase
(needed then anyway for SEBI's static-IP rule).

## 12. Owner experience

- One-time setup guided by a plain-language README (create Telegram bot, paste token).
- Telegram commands at minimum: status (portfolio/positions), report (latest metrics),
  pause/resume.
- All state persisted locally so restarts lose nothing; full trade log kept.

## 13. Later: live phase (out of scope now, recorded as decided intent)

- Broker: **Groww Trade API** (~₹499/month) with SEBI-required static IP and daily login.
- **Approve-first**: each proposed trade arrives on Telegram with Approve / Reject buttons;
  approved trades are placed at the open; no response by ~9:10 am = skipped.
- Stop-loss exits: automatic vs approval — to be decided by the owner then.
- Go-live only after the owner sees a sustained paper record (suggested ≥ 3 months, results
  consistent with backtest, drawdown within limits, no unhandled errors for 4 weeks).
  The owner makes the final call.

## 14. Left to the builder

Language is Python. Libraries, data source (free EOD NSE data, e.g. yfinance or NSE
bhavcopy — note survivorship bias of using today's Nifty 50 list), project layout,
storage format, scheduler mechanism, exact strategy rules and parameters, test approach.
