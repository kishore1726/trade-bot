# Trading bot (paper trading only)

This bot looks at Indian stock prices once a day, after the market closes, and **pretends** to trade with 20,000 rupees of play money. It then sends you a short message on Telegram. **It never places real orders and never touches your Groww account.**

You do not need to know programming. Follow the steps in order. Commands are typed in the **Terminal** app (press Cmd+Space, type "Terminal", press Enter).

## Setup

### Step 1. Install

In Terminal, go to the bot's folder (the one containing this README), then type these one at a time:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

(If a later command says "tradebot: command not found", run `source .venv/bin/activate` again in that folder.)

### Step 2. Create your Telegram bot

1. Open Telegram and search for **@BotFather** (the official one, with a blue tick).
2. Send it `/newbot`. Pick a name, then a username ending in `bot`.
3. BotFather replies with a long **token** that looks like `123456789:ABCdef...`. Keep it private; it is like a password.

### Step 3. Find your chat id

The chat id tells the bot which Telegram chat is yours. Nobody else can control it.

1. In Telegram, open your new bot and send it any message, e.g. "hi".
2. In your web browser, open this address, replacing `<TOKEN>` with your token (keep the word `bot` before it):
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. You will see a page of text. Find `"chat":{"id":` followed by a number, e.g. `987654321`. That number is your chat id.

### Step 4. Save the token and chat id

In Terminal, in the bot's folder:

    cp .env.example .env
    open -e .env

Fill in the two lines (no spaces, no quotes), save and close:

    TELEGRAM_TOKEN=123456789:ABCdef...
    TELEGRAM_CHAT_ID=987654321

### Step 5. Test Telegram

    tradebot test-telegram

You should get "Hello from your trading bot" on your phone. If it says the message was only queued, re-check `.env`. Nothing is lost: queued messages are sent once it works.

### Step 6. Run the backtest

    tradebot backtest

This replays about ten years of past prices through both strategies (this can take a few minutes and needs internet). The comparison is printed on screen and sent to Telegram.

### Step 7. Choose a strategy

In Telegram, reply to your bot with `/choose trend` or `/choose meanrev` (use the name from the comparison). Paper trading starts from the next trading day. The Telegram command bot must be running for this to work, so you may do Step 8 first and then choose.

### Step 8. Make it run by itself

    tradebot install-schedule

After this the bot runs on its own every weekday at 16:30 and listens to your Telegram commands. You can close Terminal.

**Your Mac must be switched on** (it may be asleep, but not shut down). If it was asleep or off at 16:30, the bot catches up by itself when the Mac wakes; days are never processed twice.

## What the Telegram messages mean

- **Daily summary** (each trading day): what the bot bought or sold in paper trading, what it holds, and its total value compared with the Nifty index.
- **Weekly report** (after Friday): how the strategy did over the week and in total, and how that compares with the backtest.
- **Alerts** (start with a warning sign): something needs your attention, for example missing price data or the safety switch.
- **BELOW BENCHMARK**: the strategy did worse than simply holding the Nifty index over that period. If you see this often, the strategy may not be worth following.
- **Kill switch**: if play-money value falls 15% from its highest point, the bot stops opening new positions and tells you. You decide what happens next (see `/resume`).

## Telegram commands

- `/status` - what the bot holds and whether it is paused.
- `/report` - the latest weekly report.
- `/pause` - stop making new paper buys (exits for existing positions still happen).
- `/resume` - allow new buys again (this also clears the kill switch).
- `/choose <name>` - pick the strategy to paper trade, e.g. `/choose trend`.
- `/help` - list of commands.

## How to stop everything

    tradebot uninstall-schedule

This stops the automatic runs and the Telegram listener and removes them. Your saved results are kept. To start again, run `tradebot install-schedule`. To only stop new buys, send `/pause` instead.

## Honest limitations

- **Survivorship bias.** The backtest uses today's Nifty 50 list for all past years. Companies that fell out of the index are missing, so past results look better than real life would have been.
- **Costs must be checked.** The brokerage, tax-on-sale and other charges in `config.yaml` are estimates. Compare each value with Groww's pricing page before trusting any result.
- **Tax is an estimate** and is grouped by calendar year, not the Indian financial year (April to March). It is not tax advice.
- **The first (in-sample) part of the backtest has no warm-up**, so the strategies have less history to work with at the very start.
- **Check yearly:** the holiday list (`holidays/`) and the Nifty 50 list in `config.yaml` change; update them every year.
- **Paper results are not a guarantee.** Real trading has delays, price gaps and emotions that play money does not.
- **No real orders are ever placed** in this version.
