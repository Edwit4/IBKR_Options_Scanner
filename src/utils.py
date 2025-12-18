import asyncio
import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from ib_async import Stock
from ib_async.contract import Option
from tqdm import tqdm

from calibration import ensure_connected, fetch_daily_history


def count_upper_pairs(df, min_opt_vol):
    dfv = df[df['volume'] >= min_opt_vol]
    strikes = dfv['strike'].values
    total = 0
    for s in strikes:
        total += (strikes > s).sum()
    return int(total)


def count_lower_pairs(df, min_opt_vol):
    dfv = df[df['volume'] >= min_opt_vol]
    strikes = dfv['strike'].values
    total = 0
    for s in strikes:
        total += (strikes < s).sum()
    return int(total)


def bar_update(bar, n=1):
    if bar:
        bar.update(n)


async def wait_for_quotes(
    tickers: Iterable,
    max_wait: float = 10,
    ready_ratio: float = 0.4,
    stale_cutoff: float = 1.5,
    poll_interval: float = 0.5,
    min_wait_before_early_exit: float = 2.0,
    desc: str = "settle"
):
    """
    Poll tickers until enough have bid/ask/iv populated or max_wait seconds elapse.
    Break early if progress stalls; allow early exit even with sparse data after a short wait.
    """
    tickers = list(tickers)
    if not tickers:
        return

    total = int(math.ceil(max_wait))
    bar = tqdm(total=total, desc=desc, leave=False)
    loop = asyncio.get_event_loop()
    start = loop.time()
    last_ready = 0
    last_change = start
    saw_any = False

    def is_ready(t):
        iv = t.modelGreeks.impliedVol if t.modelGreeks else t.impliedVolatility
        bid = t.bid if t.bid and t.bid > 0 else None
        ask = t.ask if t.ask and t.ask > 0 else None
        return (iv is not None and math.isfinite(float(iv))) and bid and ask

    while True:
        ready = sum(1 for t in tickers if is_ready(t))
        now = loop.time()
        elapsed = now - start

        if ready > last_ready:
            last_ready = ready
            last_change = now
            saw_any = saw_any or ready > 0

        ready_target = max(1, math.ceil(ready_ratio * len(tickers))) if ready_ratio > 0 else 0
        if (ready_ratio > 0 and ready >= ready_target) or (ready_ratio <= 0 and ready > 0):
            bar.n = min(int(elapsed), total)
            bar.refresh()
            break

        if elapsed >= max_wait:
            break

        if saw_any and (now - last_change) >= stale_cutoff and elapsed >= min_wait_before_early_exit:
            bar.n = min(int(elapsed), total)
            bar.refresh()
            break
        if not saw_any and (now - last_change) >= stale_cutoff and elapsed >= min_wait_before_early_exit:
            bar.n = min(int(elapsed), total)
            bar.refresh()
            break

        await asyncio.sleep(poll_interval)
        bar.n = min(int(loop.time() - start), total)
        bar.refresh()

    bar.n = min(int(loop.time() - start), total)
    bar.refresh()
    bar.close()


def leg_is_tight(row, max_abs, max_rel):
    bid, ask = row.get('bid'), row.get('ask')
    if bid is None or ask is None:
        return False, None, None
    spread = ask - bid
    if spread < 0:
        return False, spread, None
    mid = 0.5 * (bid + ask)
    if mid <= 0:
        return False, spread, mid
    tight = (spread <= max_abs) and ((spread / mid) <= max_rel)
    return tight, spread, mid


def percentile_rank(values: Sequence[Optional[float]], val: Optional[float]):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals or val is None or not math.isfinite(val):
        return 0.0
    vals_sorted = sorted(vals)
    below = sum(1 for x in vals_sorted if x <= val)
    return below / len(vals_sorted)


async def fetch_symbol_snapshot(ib, symbol, api_pause_sec, host, port, client_id, max_wait=6):
    """
    Light-weight snapshot to rank symbols:
    - Underlying 6-day history for momentum
    - First SMART chain, first expiry, 1-2 ATM strikes (both calls/puts) for volume/OI/IV/spread
    Returns dict with raw metrics or None on failure.
    """
    contract = Stock(symbol, 'SMART', 'USD')
    try:
        await ib.qualifyContractsAsync(contract)
    except Exception:
        return None

    bars = await fetch_daily_history(
        ib,
        contract,
        years=0.03,  # ~11 days
        api_pause_sec=api_pause_sec,
        host=host,
        port=port,
        client_id=client_id
    )
    momentum = None
    if len(bars) >= 6:
        closes = [b.close for b in bars]
        if closes[-6] and closes[-1] and closes[-6] > 0:
            momentum = (closes[-1] / closes[-6]) - 1.0

    try:
        chains = await ib.reqSecDefOptParamsAsync(contract.symbol, '', contract.secType, contract.conId)
    except Exception:
        return {
            'symbol': symbol,
            'momentum_raw': momentum,
            'opt_vol_raw': None,
            'oi_raw': None,
            'iv_raw': None,
            'iv_change_raw': None,
            'spread_quality_raw': None
        }

    smart_chains = [c for c in chains if c.exchange == 'SMART']
    if not smart_chains:
        return {
            'symbol': symbol,
            'momentum_raw': momentum,
            'opt_vol_raw': None,
            'oi_raw': None,
            'iv_raw': None,
            'iv_change_raw': None,
            'spread_quality_raw': None
        }
    chain = smart_chains[0]
    expirations = sorted(pd.to_datetime(d) for d in chain.expirations)
    if not expirations:
        return {
            'symbol': symbol,
            'momentum_raw': momentum,
            'opt_vol_raw': None,
            'oi_raw': None,
            'iv_raw': None,
            'iv_change_raw': None,
            'spread_quality_raw': None
        }
    target_exp = expirations[0].strftime('%Y%m%d')

    ticker_under = ib.reqMktData(contract, '', False, False)
    price = None
    for _ in range(40):
        if ticker_under.last is not None and not math.isnan(ticker_under.last):
            price = ticker_under.last
            break
        if ticker_under.close is not None and not math.isnan(ticker_under.close):
            price = ticker_under.close
            break
        await asyncio.sleep(0.05)

    if price is None or price <= 0:
        return {
            'symbol': symbol,
            'momentum_raw': momentum,
            'opt_vol_raw': None,
            'oi_raw': None,
            'iv_raw': None,
            'iv_change_raw': None,
            'spread_quality_raw': None
        }

    strikes = [k for k in chain.strikes if 0.95 * price < k < 1.05 * price]
    strikes = sorted(strikes)[:2] if strikes else sorted(chain.strikes)[:2]
    if not strikes:
        return {
            'symbol': symbol,
            'momentum_raw': momentum,
            'opt_vol_raw': None,
            'oi_raw': None,
            'iv_raw': None,
            'iv_change_raw': None,
            'spread_quality_raw': None
        }

    contracts = []
    for k in strikes:
        for right in ['C', 'P']:
            try:
                cds = await ib.reqContractDetailsAsync(Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=target_exp,
                    strike=k,
                    right=right,
                    exchange='SMART',
                    currency='USD',
                    tradingClass=chain.tradingClass or symbol,
                    multiplier=chain.multiplier or '100'
                ))
                if cds:
                    contracts.append(cds[0].contract)
            except Exception:
                continue

    if not contracts:
        return {
            'symbol': symbol,
            'momentum_raw': momentum,
            'opt_vol_raw': None,
            'oi_raw': None,
            'iv_raw': None,
            'iv_change_raw': None,
            'spread_quality_raw': None
        }

    tickers = []
    for c in contracts:
        try:
            t = ib.reqMktData(c, genericTickList='100,101,106', snapshot=False, regulatorySnapshot=False)
            tickers.append(t)
            await asyncio.sleep(api_pause_sec)
        except Exception:
            continue

    await wait_for_quotes(tickers, max_wait=max_wait, ready_ratio=0.3, desc=f"{symbol} rank")

    vols = []
    ois = []
    ivs = []
    spreads = []
    for t in tickers:
        bid = t.bid if t.bid and t.bid > 0 else None
        ask = t.ask if t.ask and t.ask > 0 else None
        vol = t.volume or t.bidSize or t.askSize or t.lastSize or 0
        oi = getattr(t, 'optionOpenInterest', None) or getattr(t, 'openInterest', None) or 0
        iv = t.modelGreeks.impliedVol if t.modelGreeks else t.impliedVolatility
        if vol:
            vols.append(float(vol))
        if oi:
            ois.append(float(oi))
        if iv and math.isfinite(iv) and iv > 0:
            ivs.append(float(iv))
        if bid and ask:
            spreads.append(ask - bid)

    opt_vol_raw = float(np.mean(vols)) if vols else None
    oi_raw = float(np.mean(ois)) if ois else None
    iv_raw = float(np.median(ivs)) if ivs else None
    spread_quality = None
    if spreads:
        avg_spread = float(np.mean(spreads))
        if avg_spread > 0:
            spread_quality = 1.0 / avg_spread

    return {
        'symbol': symbol,
        'momentum_raw': momentum,
        'opt_vol_raw': opt_vol_raw,
        'oi_raw': oi_raw,
        'iv_raw': iv_raw,
        'iv_change_raw': None,  # unavailable in this quick snapshot
        'spread_quality_raw': spread_quality
    }


async def quick_option_liquidity_check(
    ib,
    contract,
    target_exp,
    strikes,
    min_opt_vol,
    api_pause_sec,
    host,
    port,
    client_id
):
    """
    Probe 1-2 at-the-money options for basic liquidity; returns True/False.
    """
    if not strikes:
        return False
    if not await ensure_connected(ib, host, port, client_id):
        return False
    strikes_sorted = sorted(strikes, key=lambda k: abs(k - strikes[0]))
    sample = strikes_sorted[:2]
    tickers = []
    for k in sample:
        for right in ['C', 'P']:
            c = Option(
                symbol=contract.symbol,
                lastTradeDateOrContractMonth=target_exp,
                strike=k,
                right=right,
                exchange='SMART',
                currency=contract.currency or 'USD',
                multiplier=contract.multiplier or '100',
                tradingClass=contract.symbol
            )
            try:
                t = ib.reqMktData(c, genericTickList='100,106', snapshot=False, regulatorySnapshot=False)
                tickers.append(t)
            except Exception:
                continue
    await wait_for_quotes(tickers, max_wait=5, ready_ratio=0.5, desc=f"{contract.symbol} quick liq")
    for t in tickers:
        bid = t.bid if t.bid and t.bid > 0 else None
        ask = t.ask if t.ask and t.ask > 0 else None
        vol = t.volume or t.bidSize or t.askSize or t.lastSize or 0

        # Loose gate: any sign of life with some activity/size
        if (bid or ask) and vol >= min_opt_vol:
            return True
    return False
