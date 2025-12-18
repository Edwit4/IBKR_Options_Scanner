import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
import math
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
from ib_async import IB, Option, ComboLeg, Contract, Stock

# Reuse shared settings
from options_list import (
    OUTPUT_DIR,
    MC_TP_FRAC,
    MC_SL_DEBIT_FRAC,
    MC_SL_CREDIT_MULT,
    STRATEGY_CONFIG,
    RISK_FREE_RATE_FALLBACK,
    API_PAUSE_SEC,
)
from mc import simulate_vertical_pop_ev_torch

HOST = '127.0.0.1'
#PORT = 7496 # Live trading
PORT = 7497              # Paper trading
CLIENT_ID = 199

# Risk budget (max total risk across selected trades)
TOTAL_RISK_BUDGET = 3000.0
# Max positions to place per run
MAX_POSITIONS = 5
# Default quantity per spread
DEFAULT_QTY = 1

# Slippage / fill handling
MAX_SLIPPAGE_FRAC_DEBIT = 0.2    # as fraction of width, extra debit we are willing to pay
MAX_SLIPPAGE_FRAC_CREDIT = 0.2   # as fraction of width, credit we are willing to concede
SLIPPAGE_STEPS = 4               # number of price levels between mid and worst case (inclusive)
FILL_TIMEOUT = 60.0              # seconds to wait for parent fill before trying worse price
FILL_CHECK_INTERVAL = 2.0        # polling interval for order fill status

# Exit timing (to avoid same-day exits / PDT)
DELAY_EXITS_UNTIL_NEXT_DAY = True
EXIT_ACTIVATION_TIME = "09:35:00"  # local time for exit orders to become active on the next day


def latest_output_csv():
    """Pick the newest scanner CSV from OUTPUT_DIR (by modification time)."""
    if not OUTPUT_DIR.exists():
        return None
    files = list(OUTPUT_DIR.glob("ibkr_spreads_*.csv"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def parse_strategy(row):
    """
    Parse the Strategy column into structure + strikes.
    Example values:
      "100.0/105.0 Bull Call"
      "100.0/105.0 Bear Call"
      "95.0/100.0 Bull Put"
      "100.0/95.0 Bear Put"
    Returns (structure, lower_strike, upper_strike, right_side)
    """
    strategy = row['Strategy']
    strikes_part, structure = strategy.split(' ', 1)
    s1, s2 = strikes_part.split('/')
    k1 = float(s1)
    k2 = float(s2)
    lower, upper = sorted([k1, k2])
    if 'Call' in structure:
        right = 'C'
    else:
        right = 'P'
    return structure, lower, upper, right


def legs_for_vertical(structure, lower, upper, right):
    """
    Return legs as list of (action, strike) tuples with right known.
    """
    if structure == 'Bull Call':
        # Long lower call, short higher call
        return [('BUY', lower), ('SELL', upper)]
    if structure == 'Bear Call':
        # Short lower call, long higher call
        return [('SELL', lower), ('BUY', upper)]
    if structure == 'Bull Put':
        # Short higher put, long lower put
        return [('SELL', upper), ('BUY', lower)]
    if structure == 'Bear Put':
        # Long higher put, short lower put
        return [('BUY', upper), ('SELL', lower)]
    raise ValueError(f"Unsupported structure: {structure}")


async def build_combo(ib: IB, symbol: str, expiry: str, structure: str, right: str, lower: float, upper: float):
    """
    Build a combo contract with qualified legs.
    """
    legs_def = legs_for_vertical(structure, lower, upper, right)
    combo_legs = []
    for action, strike in legs_def:
        opt = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange='SMART',
            currency='USD',
        )
        qualified = await ib.qualifyContractsAsync(opt)
        if not qualified:
            raise RuntimeError(f"Failed to qualify option {symbol} {expiry} {strike} {right}")
        leg_contract = qualified[0]
        combo_legs.append(
            ComboLeg(
                conId=leg_contract.conId,
                ratio=1,
                action=action,
                exchange='SMART'
            )
        )

    combo = Contract()
    combo.symbol = symbol
    combo.secType = 'BAG'
    combo.currency = 'USD'
    combo.exchange = 'SMART'
    combo.comboLegs = combo_legs
    return combo


def calc_targets(entry: float, max_loss: float, max_profit: float, entry_type: str):
    """
    Compute take-profit and stop prices matching simulation rules.
    """
    if entry_type == 'Debit':
        tp_price = entry + MC_TP_FRAC * max_profit
        stop_price = max(entry - MC_SL_DEBIT_FRAC * max_loss, 0.01)
        parent_action = 'BUY'
    else:  # Credit
        tp_price = max(entry * (1 - MC_TP_FRAC), 0.01)
        stop_price = entry * (1 + MC_SL_CREDIT_MULT)
        parent_action = 'SELL'

    return parent_action, entry, tp_price, stop_price


def derive_strikes(structure: str, lower: float, upper: float) -> Tuple[float, float]:
    """
    Return (long_strike, short_strike) consistent with strategy.
    """
    if structure == 'Bull Call':
        return lower, upper
    if structure == 'Bear Call':
        return lower, upper  # long lower, short upper handled via actions
    if structure == 'Bull Put':
        return upper, lower  # long higher, short lower (long_strike > short_strike for puts)
    if structure == 'Bear Put':
        return upper, lower
    raise ValueError(f"Unsupported structure: {structure}")


def quote_ready(ticker):
    return (
        ticker is not None and
        ticker.bid is not None and ticker.ask is not None and
        ticker.bid > 0 and ticker.ask > 0 and
        (
            (ticker.modelGreeks is not None and ticker.modelGreeks.impliedVol is not None) or
            (ticker.impliedVolatility is not None)
        )
    )


async def fetch_leg_market(ib: IB, opt: Option, timeout: float = 5.0):
    """
    Request market data for a single option leg and wait briefly for bid/ask/iv.
    """
    ticker = ib.reqMktData(opt, genericTickList='100,106', snapshot=False, regulatorySnapshot=False)
    start = time.time()
    while time.time() - start < timeout:
        if quote_ready(ticker):
            break
        await asyncio.sleep(0.2)
    if not quote_ready(ticker):
        return None, None
    bid = float(ticker.bid)
    ask = float(ticker.ask)
    mid = 0.5 * (bid + ask)
    iv = None
    if ticker.modelGreeks and ticker.modelGreeks.impliedVol is not None:
        iv = float(ticker.modelGreeks.impliedVol)
    elif ticker.impliedVolatility is not None:
        iv = float(ticker.impliedVolatility)
    return ticker.contract, {'bid': bid, 'ask': ask, 'mid': mid, 'iv': iv}


def price_from_quotes(structure: str, right: str, lower: float, upper: float, quotes: Dict[float, Dict[str, float]]):
    """
    Compute mid-price entry for the spread using the provided per-strike quotes.
    """
    legs = legs_for_vertical(structure, lower, upper, right)
    price = 0.0
    for action, strike in legs:
        q = quotes.get(strike)
        if q is None or q.get('mid') is None:
            return None
        mid = q['mid']
        if not math.isfinite(mid) or mid <= 0:
            return None
        price += mid if action == 'BUY' else -mid
    return abs(price)


def run_mc_check(
    entry: float,
    entry_type: str,
    width: float,
    T: float,
    long_strike: float,
    short_strike: float,
    right: str,
    underlying_price: float,
    sigma_input: float,
    risk_free_rate: float,
    config: Dict[str, Any]
) -> Tuple[bool, Dict[str, float]]:
    """
    Re-run MC and apply the same EV/ROI filters. Returns (accepted, metrics).
    """
    if (
        entry is None or entry <= 0 or
        sigma_input is None or not math.isfinite(sigma_input) or sigma_input <= 0 or
        T is None or not math.isfinite(T) or T <= 0 or
        underlying_price is None or not math.isfinite(underlying_price) or underlying_price <= 0
    ):
        return False, {}

    max_profit = width - entry if entry_type == 'Debit' else entry
    max_loss = entry if entry_type == 'Debit' else width - entry
    if max_profit <= 0 or max_loss <= 0:
        return False, {}

    pop_mc, ev_mc, avg_ht, p_daytrade, pop_se, ev_se = simulate_vertical_pop_ev_torch(
        S0=underlying_price,
        r=risk_free_rate,
        sigma=sigma_input,
        T=T,
        long_strike=long_strike,
        short_strike=short_strike,
        right=right,
        entry=entry,
        max_profit=max_profit,
        max_loss=max_loss,
        entry_type=entry_type,
        tp_frac=config['MC_TP_FRAC'],
        sl_frac=config['MC_SL_DEBIT_FRAC'] if entry_type == 'Debit' else config['MC_SL_CREDIT_MULT'],
        n_paths=config['MC_N_PATHS'],
        n_steps=config['MC_N_STEPS'],
        daytrade_threshold_days=config['MC_DAYTRADE_THRESHOLD_DAYS'],
        device=None,
        mu=None,
        realized_vol_frac=None
    )

    # Overwrite T in MC call with daily approx using config; assume 30 days if unknown
    hold_days = avg_ht * 365.0
    risk = entry if entry_type == 'Debit' else max_loss
    roi_trade = ev_mc / risk
    ev_per_risk = roi_trade
    roi_monthly = roi_trade * (30.0 / hold_days)
    portfolio_roi_monthly = roi_monthly * config['PORTFOLIO_RISK_BUDGET_FRAC']

    accepted = (
        pop_mc > config['MIN_POP'] and
        (not config['REQUIRE_POSITIVE_EV'] or ev_mc > 0) and
        hold_days >= config['MIN_HOLD_DAYS'] and
        ev_mc >= config['MIN_EV'] and
        ev_per_risk >= config['MIN_EV_PER_RISK'] and
        portfolio_roi_monthly >= config['TARGET_MONTHLY_RETURN']
    )

    return accepted, {
        'pop': pop_mc,
        'ev': ev_mc,
        'avg_ht': avg_ht,
        'p_daytrade': p_daytrade,
        'pop_se': pop_se,
        'ev_se': ev_se,
        'ev_per_risk': ev_per_risk,
        'roi_monthly': roi_monthly,
        'portfolio_roi_monthly': portfolio_roi_monthly,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'hold_days': hold_days,
    }


async def wait_for_fill(ib: IB, parent_order_id: int, timeout: float) -> bool:
    """
    Poll trades for the given parent order id until filled or timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        trades = [t for t in ib.trades() if t.order.orderId == parent_order_id]
        for t in trades:
            status = t.orderStatus.status
            if status.lower() == 'filled':
                return True
        await asyncio.sleep(FILL_CHECK_INTERVAL)
    return False


async def place_bracket(ib: IB, combo: Contract, parent_action: str, qty: int, entry: float, tp_price: float, stop_price: float, good_after_time: Optional[str] = None):
    """
    Submit a bracket order on the combo contract. Returns (orders, parent_order_id).
    If good_after_time is provided, apply it to the exit legs to delay activation.
    """
    bracket = ib.bracketOrder(
        parent_action,
        qty,
        entry,
        tp_price,
        stop_price,
        tif='DAY'
    )
    parent = bracket[0]
    tp = bracket[1]
    sl = bracket[2]

    if good_after_time:
        tp.goodAfterTime = good_after_time
        sl.goodAfterTime = good_after_time

    ib.placeOrder(combo, parent)
    tp.parentId = parent.orderId
    sl.parentId = parent.orderId
    ib.placeOrder(combo, tp)
    ib.placeOrder(combo, sl)
    return bracket, parent.orderId


async def main(csv_path: Optional[Path] = None):
    ib = IB()
    try:
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, readonly=False)
    except Exception as e:
        print(f"Failed to connect to IB: {e}")
        return

    if csv_path is None:
        csv_path = latest_output_csv()
    if csv_path is None or not csv_path.exists():
        print("No scanner output CSV found.")
        ib.disconnect()
        return
    else:
        print(f"Using scanner output: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty; nothing to trade.")
        await ib.disconnectAsync()
        return

    # Ensure best trades first
    if 'EV_Per_Risk' in df.columns:
        df = df.sort_values(['EV_Per_Risk', 'PoP'], ascending=False)

    total_risk = 0.0
    placed = 0

    for _, row in df.iterrows():
        if placed >= MAX_POSITIONS:
            break

        symbol = row['Symbol']
        expiry_str = str(int(row['Expiry']))
        structure, lower, upper, right = parse_strategy(row)

        # Underlying price and risk-free rate
        underlying_contract = Stock(symbol, 'SMART', 'USD')
        await ib.qualifyContractsAsync(underlying_contract)
        under_ticker = ib.reqMktData(underlying_contract, '106', False, False)
        underlying_price = None
        risk_free_rate = None
        for _ in range(50):
            if under_ticker.last is not None and not math.isnan(under_ticker.last):
                underlying_price = under_ticker.last
            elif under_ticker.close is not None and not math.isnan(under_ticker.close):
                underlying_price = under_ticker.close
            if under_ticker.modelGreeks and under_ticker.modelGreeks.riskFreeRate is not None:
                risk_free_rate = under_ticker.modelGreeks.riskFreeRate
            if underlying_price is not None and risk_free_rate is not None:
                break
            await asyncio.sleep(0.1)
        if underlying_price is None or underlying_price <= 0:
            print(f"Skipping {symbol}: no valid underlying price")
            continue
        if risk_free_rate is None or not math.isfinite(risk_free_rate):
            risk_free_rate = RISK_FREE_RATE_FALLBACK

        # Contracts and quotes
        try:
            legs_def = legs_for_vertical(structure, lower, upper, right)
            quotes = {}
            for action, strike in legs_def:
                opt = Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=expiry_str,
                    strike=strike,
                    right=right,
                    exchange='SMART',
                    currency='USD',
                )
                qualified = await ib.qualifyContractsAsync(opt)
                if not qualified:
                    raise RuntimeError(f"Failed to qualify {symbol} {expiry_str} {strike} {right}")
                contract, quote = await fetch_leg_market(ib, qualified[0])
                if quote is None:
                    raise RuntimeError(f"No quote for {symbol} {strike} {right}")
                quotes[strike] = quote
                await asyncio.sleep(API_PAUSE_SEC)
        except Exception as e:
            print(f"Skipping {symbol} {row['Strategy']}: {e}")
            continue

        # Compute mid entry and width
        mid_entry = price_from_quotes(structure, right, lower, upper, quotes)
        if mid_entry is None:
            print(f"Skipping {symbol} {row['Strategy']}: unable to compute mid price")
            continue
        width = abs(upper - lower)
        if width <= 0:
            print(f"Skipping {symbol} {row['Strategy']}: non-positive width")
            continue

        # Time to expiry for MC (in years)
        now = pd.Timestamp.now()
        expiry_dt = pd.to_datetime(expiry_str)
        days_to_expiry = max((expiry_dt - now).days, 1)
        T = days_to_expiry / 365.0

        # Average IV
        ivs = [q['iv'] for q in quotes.values() if q.get('iv') and math.isfinite(q['iv']) and q['iv'] > 0]
        sigma_input = float(sum(ivs) / len(ivs)) if ivs else None
        if sigma_input is None or not math.isfinite(sigma_input) or sigma_input <= 0:
            print(f"Skipping {symbol} {row['Strategy']}: no usable IV for MC")
            continue
        entry_type = row['Entry Type']

        # Build price ladder from mid toward worse acceptable slippage
        max_slip = (MAX_SLIPPAGE_FRAC_DEBIT if entry_type == 'Debit' else MAX_SLIPPAGE_FRAC_CREDIT) * width
        price_levels = []
        for i in range(SLIPPAGE_STEPS + 1):
            frac = i / SLIPPAGE_STEPS
            if entry_type == 'Debit':
                price_levels.append(mid_entry + frac * max_slip)
            else:
                price_levels.append(max(mid_entry - frac * max_slip, 0.01))

        long_strike, short_strike = derive_strikes(structure, lower, upper)

        # Pre-evaluate MC across price ladder
        accepted_prices = []
        for price in price_levels:
            accepted, metrics = run_mc_check(
                entry=price,
                entry_type=entry_type,
                width=width,
                T=T,
                long_strike=long_strike,
                short_strike=short_strike,
                right=right,
                underlying_price=underlying_price,
                sigma_input=sigma_input,
                risk_free_rate=risk_free_rate,
                config=STRATEGY_CONFIG
            )
            if accepted:
                accepted_prices.append((price, metrics))

        if not accepted_prices:
            print(f"Skipping {symbol} {row['Strategy']}: EV filters failed across slippage range.")
            continue

        # Budget check with best accepted price (first in list)
        first_max_loss = accepted_prices[0][1]['max_loss']
        if total_risk + first_max_loss > TOTAL_RISK_BUDGET:
            continue

        # Try to fill from best to worst accepted price
        filled = False
        for price, metrics in accepted_prices:
            max_profit = metrics['max_profit']
            max_loss = metrics['max_loss']

            if total_risk + max_loss > TOTAL_RISK_BUDGET:
                continue

            try:
                combo = await build_combo(ib, symbol, expiry_str, structure, right, lower, upper)
            except Exception as e:
                print(f"Skipping {symbol} {row['Strategy']}: {e}")
                break

            # Exit activation timing
            good_after_time = None
            if DELAY_EXITS_UNTIL_NEXT_DAY:
                # Use next calendar day at configured time
                next_day = (pd.Timestamp.now() + pd.Timedelta(days=1)).normalize()
                activation_ts = next_day + pd.to_timedelta(EXIT_ACTIVATION_TIME)
                good_after_time = activation_ts.strftime("%Y%m%d %H:%M:%S")

            parent_action, entry, tp_price, stop_price = calc_targets(price, max_loss, max_profit, entry_type)
            try:
                bracket_orders, parent_id = await place_bracket(ib, combo, parent_action, DEFAULT_QTY, entry, tp_price, stop_price, good_after_time=good_after_time)
                got_fill = await wait_for_fill(ib, parent_id, FILL_TIMEOUT)
                if got_fill:
                    filled = True
                    total_risk += max_loss
                    placed += 1
                    print(f"Filled {structure} on {symbol} ({row['Strategy']}) entry={entry:.2f} tp={tp_price:.2f} stop={stop_price:.2f} risk={max_loss:.2f} total_risk={total_risk:.2f}")
                    break
                else:
                    for o in bracket_orders:
                        ib.cancelOrder(o)
                    await asyncio.sleep(API_PAUSE_SEC)
            except Exception as e:
                print(f"Order attempt failed for {symbol} {row['Strategy']}: {e}")
                for o in locals().get('bracket_orders', []):
                    ib.cancelOrder(o)
                await asyncio.sleep(API_PAUSE_SEC)
                continue

        if not filled:
            print(f"Failed to fill {symbol} {row['Strategy']} within slippage bounds.")

    print(f"Done. Placed {placed} orders. Total risk: {total_risk:.2f}")
    ib.disconnect()


if __name__ == '__main__':
    path = latest_output_csv()
    asyncio.run(main(path))
