import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_calibration_cache(path: Path):
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_calibration_cache(path: Path, cache):
    try:
        with open(path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save calibration cache: {e}")


async def ensure_connected(ib, host, port, client_id):
    if ib.isConnected():
        return True
    print("Reconnecting to IBKR...")
    try:
        await ib.connectAsync(host, port, clientId=client_id, readonly=True)
        return True
    except Exception as e:
        print(f"Reconnect failed: {e}")
        return False


async def fetch_daily_history(ib, contract, years, api_pause_sec, host, port, client_id):
    if years <= 0:
        return []

    # IB requires duration in years for windows >= 365 days
    if years >= 1.0:
        duration_years = max(1, int(math.ceil(years)))
        duration_str = f"{duration_years} Y"
    else:
        duration_days = max(10, int(years * 365))
        duration_str = f"{duration_days} D"

    if not await ensure_connected(ib, host, port, client_id):
        return []
    try:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration_str,
            barSizeSetting='1 day',
            whatToShow='ADJUSTED_LAST',
            useRTH=True,
            formatDate=1,
            keepUpToDate=False
        )
        await asyncio.sleep(api_pause_sec)
        return bars or []
    except Exception as e:
        print(f"  Historical data fetch failed for {contract.symbol}: {e}")
        return []


def compute_calibration(prices: pd.Series, implied_vol_hint):
    """
    Compute annualized drift/vol and realized/implied ratio.
    """
    if len(prices) < 30:
        return None
    rets = np.log(prices / prices.shift(1)).dropna()
    if rets.empty:
        return None

    sigma_hist = float(rets.std() * math.sqrt(252))

    # realized vs implied adjustment
    realized_vol_frac = None
    if implied_vol_hint is not None and implied_vol_hint > 0 and math.isfinite(implied_vol_hint):
        realized_vol_frac = sigma_hist / implied_vol_hint if implied_vol_hint > 0 else None

    mu_ann = float(rets.mean() * 252)

    # short momentum for trend alignment
    momentum_short = float((prices.iloc[-1] / prices.iloc[-20]) - 1.0) if len(prices) >= 20 else None

    return {
        'mu': mu_ann,
        'sigma_hist': sigma_hist,
        'realized_vol_frac': realized_vol_frac,
        'momentum_short': momentum_short
    }


async def calibrate_symbol(ib, contract, implied_vol_hint, cache, settings):
    sym = contract.symbol
    cached = cache.get(sym)
    now = pd.Timestamp.now()
    if cached:
        ts = pd.to_datetime(cached.get('timestamp')) if cached.get('timestamp') else None
        has_rv = cached.get('realized_vol_frac') is not None
        # Only honor cache if fresh AND either we already have RV or no new hint to refresh it
        if ts is not None and (now - ts).days < settings['CALIBRATION_MAX_AGE_DAYS'] and (has_rv or implied_vol_hint is None):
            return cached, False

    bars = await fetch_daily_history(
        ib,
        contract,
        settings['MU_SIGMA_YEARS'],
        settings['API_PAUSE_SEC'],
        settings['HOST'],
        settings['PORT'],
        settings['CLIENT_ID']
    )
    if not bars:
        fallback = {
            'mu': settings['MU_DRIFT'],
            'sigma_hist': None,
            'realized_vol_frac': settings['REALIZED_VOL_FRACTION'],
            'momentum_short': None,
            'timestamp': now.isoformat()
        }
        cache[sym] = fallback
        return fallback, True

    closes = pd.Series([b.close for b in bars], index=[pd.to_datetime(b.date) for b in bars])
    calc = compute_calibration(closes, implied_vol_hint)
    if calc is None:
        fallback = {
            'mu': settings['MU_DRIFT'],
            'sigma_hist': None,
            'realized_vol_frac': settings['REALIZED_VOL_FRACTION'],
            'momentum_short': None,
            'timestamp': now.isoformat()
        }
        cache[sym] = fallback
        return fallback, True

    calc['timestamp'] = now.isoformat()
    cache[sym] = calc
    return calc, True
