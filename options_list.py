import torch
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
import pandas as pd
import numpy as np
from ib_async import *
from ib_async.contract import Option
import math
import logging
import json
from pathlib import Path
import os
from collections import defaultdict
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

logging.getLogger('ibapi').setLevel(logging.CRITICAL)
logging.getLogger('ibapi.wrapper').setLevel(logging.CRITICAL)
logging.getLogger('ibapi.client').setLevel(logging.CRITICAL)

logging.getLogger('ib_async').setLevel(logging.CRITICAL)
logging.getLogger('ib_async.wrapper').setLevel(logging.CRITICAL)
logging.getLogger('ib_async.client').setLevel(logging.CRITICAL)

# Connection settings
HOST = '127.0.0.1'
PORT = 7496
CLIENT_ID = 99

# Strategy Parameters
MAX_DEBIT = 3.00       # Max cost
MIN_OPT_VOL = 5        # Liquidity filter for individual legs
MIN_POP = 0.2         # Min Probability of Profit
RISK_FREE_RATE = 0.044 # ~4.4% (Used if IBKR yields are unavailable)
SCAN_LIMIT = 25         # Limit scanner result count to avoid pacing issues
MIN_PRICE = 10        # Skip penny/small names that often lack options data; scanner uses this floor
MIN_ENTRY = 0.15       # Min absolute entry price (debit/credit)
MIN_EV_PER_RISK = 0.12  # Filter on EV/Risk; set >0 to require edge per unit risk
MIN_EV = 0.1           # Filter on absolute EV; set >0 to require positive expectation
SCANNER_CODES = ['OPT_VOLUME_MOST_ACTIVE',
                 'HOT_BY_OPT_VOLUME',
                 'HIGH_OPT_IMP_VOLAT', 
                 'HIGH_OPT_IMP_VOLAT_OVER_HIST',
                 'OPT_OPEN_INTEREST_MOST_ACTIVE', 
                 'MOST_ACTIVE'] 
MANUAL_SYMBOLS = []   # Optional override list; if non-empty, skip scanner and use this list
TARGET_MONTHLY_RETURN = 0.005   # Percent per month, as a fraction
REQUIRE_POSITIVE_EV = True     # Only keep spreads with positive expectation
MIN_DAYS = 0
MAX_DAYS = 60
VOL_PREMIUM_MIN = 1.0  # require IV/realized >= value at symbol level to proceed

# MC Params
MC_TP_FRAC = 0.5        # take profit at percent of max profit
MC_SL_FRAC = 0.25        # stop at percent of max loss (legacy)
MC_SL_CREDIT_MULT = 2.0  # stop-loss multiple of credit received for credit spreads
MC_SL_DEBIT_FRAC = 0.5   # stop-loss fraction of debit paid for debit spreads
MC_N_PATHS = 500000
MC_N_STEPS = 100
MC_DAYTRADE_THRESHOLD_DAYS = 1.0
MIN_HOLD_DAYS = 1.0     # skip spreads whose expected hold is under 1 day
API_PAUSE_SEC = 0.15  # small pause between IBKR requests to avoid pacing/disconnects
MIN_WIDTH = 2.0
MAX_WIDTH = 10.0

# Real-world drift & vol-premium assumptions (tweak these)
USE_REAL_WORLD_DRIFT = True
MU_DRIFT = 0.10          # 10% annual drift for bullish underlyings, example

USE_VOL_RISK_PREMIUM = False
REALIZED_VOL_FRACTION = 0.7   # realized_vol = 0.7 * implied_vol (if you turn this on)
MU_SIGMA_YEARS = 4.0          # lookback for drift/vol calibration
REALIZED_VOL_YEARS = 2.0      # lookback for realized vs implied ratio
CALIBRATION_CACHE_FILE = Path("calibration_cache.json")
REALIZED_FRAC_MIN = 0.3
REALIZED_FRAC_MAX = 1.5
CALIBRATION_MAX_AGE_DAYS = 3
IV_REALIZED_RATIO_MIN = 1.0   # require IV >= value realized (i.e., realized_vol_frac <= ~0.71)
MIN_SKEW_CAPTURE = 0.0        # minimum IV skew capture per spread (short IV - long IV); can raise to enforce skew edge
MIN_SKEW_CAPTURE_BULL = -0.05   # skew capture floor for bull structures
MIN_SKEW_CAPTURE_BEAR = -0.20 # skew capture floor for bear structures (allow slight negative if skew flips)
MU_TREND_THRESH = 0.08        # annual drift threshold for up/down classification 
MOMENTUM_WINDOW_DAYS = 20     # short-term momentum window for alignment
FILTER_VERBOSE = True         # emit reasons when candidates/spreads are filtered out
TIGHT_MAX_ABS = 0.07
TIGHT_MAX_REL = 0.15

# ==========================================
def log_filter(reason, symbol=None, expiry=None, spread=None):
    if not FILTER_VERBOSE:
        return
    ctx_parts = []
    if symbol:
        ctx_parts.append(str(symbol))
    if expiry:
        ctx_parts.append(str(expiry))
    if spread:
        ctx_parts.append(str(spread))
    ctx = " | ".join(ctx_parts)
    if ctx:
        print(f"  Filter[{ctx}]: {reason}")
    else:
        print(f"  Filter: {reason}")

def log_pass(msg, symbol=None, expiry=None, spread=None):
    if not FILTER_VERBOSE:
        return
    ctx_parts = []
    if symbol:
        ctx_parts.append(str(symbol))
    if expiry:
        ctx_parts.append(str(expiry))
    if spread:
        ctx_parts.append(str(spread))
    ctx = " | ".join(ctx_parts)
    if ctx:
        print(f"  Pass[{ctx}]: {msg}")
    else:
        print(f"  Pass: {msg}")

def bs_option_price_torch(S, K, T, r, sigma, right):
    """
    Black–Scholes price for a European call or put (vectorized).
    S, K, T, r, sigma are tensors broadcastable to same shape.
    right: 'C' or 'P'
    """
    eps = 1e-12
    T_clamp = torch.clamp(T, min=eps)
    sigma_clamp = torch.clamp(sigma, min=eps)
    S_clamp = torch.clamp(S, min=eps)
    K_clamp = torch.clamp(K, min=eps)

    sqrtT = torch.sqrt(T_clamp)
    d1 = (torch.log(S_clamp / K_clamp) + (r + 0.5 * sigma_clamp**2) * T_clamp) / (sigma_clamp * sqrtT)
    d2 = d1 - sigma_clamp * sqrtT

    if right == 'C':
        return S_clamp * torch.distributions.normal.Normal(0, 1).cdf(d1) - \
               K_clamp * torch.exp(-r * T_clamp) * torch.distributions.normal.Normal(0, 1).cdf(d2)
    else:  # 'P'
        return K_clamp * torch.exp(-r * T_clamp) * torch.distributions.normal.Normal(0, 1).cdf(-d2) - \
               S_clamp * torch.distributions.normal.Normal(0, 1).cdf(-d1)


def vertical_price_torch(S, T, r, sigma, long_strike, short_strike, right):
    """
    Mark-to-market price of a vertical spread (vectorized) in torch.
    """
    long_price = bs_option_price_torch(S, long_strike, T, r, sigma, right)
    short_price = bs_option_price_torch(S, short_strike, T, r, sigma, right)
    return long_price - short_price


def vertical_intrinsic_torch(S, long_strike, short_strike, right):
    """
    Intrinsic value of same vertical at expiry (vectorized).
    """
    if right == 'C':
        long_val = torch.clamp(S - long_strike, min=0.0)
        short_val = torch.clamp(S - short_strike, min=0.0)
    else:
        long_val = torch.clamp(long_strike - S, min=0.0)
        short_val = torch.clamp(short_strike - S, min=0.0)
    return long_val - short_val

def simulate_vertical_pop_ev_torch(
    S0,
    r,
    sigma,
    T,
    long_strike,
    short_strike,
    right,
    entry,
    max_profit,
    max_loss,
    entry_type='Debit',
    tp_frac=0.5,
    sl_frac=1.0,
    n_paths=20000,
    n_steps=80,
    daytrade_threshold_days=1.0,
    device=None,
    mu=None,                # <-- NEW: real-world drift
    realized_vol_frac=None  # <-- NEW: RV/IV adjustment
):
    """
    Monte Carlo estimator for a vertical with early TP/SL using PyTorch (GPU-capable).

    Returns:
        pop        : scalar float
        ev         : scalar float (expected P/L)
        avg_ht     : scalar float (years)
        p_daytrade : scalar float
    """
    if T <= 0 or sigma <= 0 or S0 <= 0 or max_loss <= 0 or max_profit <= 0:
        return 0.0, 0.0, 0.0, 0.0

    if device is None:
        device = choose_torch_device()

    # Scalars -> tensors on device
    S0 = torch.tensor(S0, dtype=torch.float64, device=device)
    r = torch.tensor(r, dtype=torch.float64, device=device)
    sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
    T_tot = torch.tensor(T, dtype=torch.float64, device=device)
    long_strike = torch.tensor(long_strike, dtype=torch.float64, device=device)
    short_strike = torch.tensor(short_strike, dtype=torch.float64, device=device)
    entry = torch.tensor(entry, dtype=torch.float64, device=device)
    max_profit = torch.tensor(max_profit, dtype=torch.float64, device=device)
    max_loss = torch.tensor(max_loss, dtype=torch.float64, device=device)

    # Choose effective drift and vol for path
    if mu is None:
        mu_t = r          # default: risk-neutral
    else:
        mu_t = torch.tensor(mu, dtype=torch.float64, device=device)

    if realized_vol_frac is None:
        sigma_path = sigma  # default: use implied vol as path vol
    else:
        sigma_path = sigma * realized_vol_frac

    # Rate used for BS pricing; follow calibrated drift when provided
    if mu is None:
        r_pricing = r
    else:
        r_pricing = mu_t

    tp_level = tp_frac * max_profit
    sl_level = -sl_frac * max_loss

    dt = T_tot / n_steps
    daytrade_threshold_years = daytrade_threshold_days / 365.0

    # Path state
    S = S0.expand(n_paths).clone()
    t = torch.zeros(n_paths, dtype=torch.float64, device=device)
    alive = torch.ones(n_paths, dtype=torch.bool, device=device)

    # Outputs per path
    pl_paths = torch.zeros(n_paths, dtype=torch.float64, device=device)
    ht_paths = torch.zeros(n_paths, dtype=torch.float64, device=device)

    normal = torch.distributions.normal.Normal(
        torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

    for _ in range(n_steps):
        if not alive.any():
            break

        idx = torch.nonzero(alive, as_tuple=False).squeeze(-1)
        S_alive = S[idx]
        t_alive = t[idx]

        Z = normal.sample(S_alive.shape)
        drift = (mu_t - 0.5 * sigma_path**2) * dt
        diff = sigma_path * torch.sqrt(dt) * Z
        S_new = S_alive * torch.exp(drift + diff)
        t_new = t_alive + dt

        S[idx] = S_new
        t[idx] = t_new

        T_remaining = torch.clamp(T_tot - t_new, min=1e-8)

        # Price vertical for alive paths
        vert_val = vertical_price_torch(
            S_new, T_remaining, r_pricing, sigma, long_strike, short_strike, right
        )

        if entry_type == 'Debit':
            pl = vert_val - entry
        else:  # Credit
            pl = entry - vert_val

        hit_tp = pl >= tp_level
        hit_sl = pl <= sl_level
        hit_any = hit_tp | hit_sl

        if hit_any.any():
            hit_idx_local = torch.nonzero(hit_any, as_tuple=False).squeeze(-1)
            hit_idx_global = idx[hit_idx_local]

            pl_paths[hit_idx_global] = pl[hit_idx_local]
            ht_paths[hit_idx_global] = t_new[hit_idx_local]
            alive[hit_idx_global] = False

    # Settle remaining at expiry
    if alive.any():
        idx = torch.nonzero(alive, as_tuple=False).squeeze(-1)
        S_alive = S[idx]

        intrinsic = vertical_intrinsic_torch(
            S_alive, long_strike, short_strike, right
        )

        if entry_type == 'Debit':
            pl_rem = intrinsic - entry
        else:
            pl_rem = entry - intrinsic

        pl_paths[idx] = pl_rem
        ht_paths[idx] = T_tot
        alive[idx] = False

    # Aggregate statistics (still on device)
    pop = (pl_paths > 0).double().mean()
    ev = pl_paths.mean()
    avg_ht = ht_paths.mean()

    p_daytrade = (ht_paths <= daytrade_threshold_years).double().mean()

    # Return as Python floats
    return float(pop.item()), float(ev.item()), float(avg_ht.item()), float(p_daytrade.item())

# ------------------------------------------
# Utility: count candidate vertical pairs for progress bars
# ------------------------------------------

def count_upper_pairs(df):
    dfv = df[df['volume'] >= MIN_OPT_VOL]
    strikes = dfv['strike'].values
    total = 0
    for s in strikes:
        total += (strikes > s).sum()
    return int(total)


def count_lower_pairs(df):
    dfv = df[df['volume'] >= MIN_OPT_VOL]
    strikes = dfv['strike'].values
    total = 0
    for s in strikes:
        total += (strikes < s).sum()
    return int(total)


def bar_update(bar, n=1):
    if bar:
        bar.update(n)


async def wait_for_quotes(
    tickers,
    max_wait=10,
    ready_ratio=0.4,
    stale_cutoff=1.5,
    poll_interval=0.5,
    min_wait_before_early_exit=2.0,
    desc="settle"
):
    """
    Poll tickers until enough have bid/ask/iv populated or max_wait seconds elapse.
    Break early if progress stalls; allow early exit even with sparse data after a short wait.
    """
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


def leg_is_tight(row, max_abs=TIGHT_MAX_ABS, max_rel=TIGHT_MAX_REL):
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


def percentile_rank(values, val):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals or val is None or not math.isfinite(val):
        return 0.0
    vals_sorted = sorted(vals)
    below = sum(1 for x in vals_sorted if x <= val)
    return below / len(vals_sorted)


def choose_torch_device(prefer=None):
    """
    Resolve torch device with optional override via TORCH_DEVICE env ('cpu'/'cuda').
    """
    env_dev = os.environ.get("TORCH_DEVICE")
    if env_dev:
        if env_dev.lower().startswith("cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Calibration Helpers
# ==========================================

def load_calibration_cache():
    if CALIBRATION_CACHE_FILE.exists():
        try:
            with open(CALIBRATION_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_calibration_cache(cache):
    try:
        with open(CALIBRATION_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save calibration cache: {e}")


async def ensure_connected(ib):
    if ib.isConnected():
        return True
    print("Reconnecting to IBKR...")
    try:
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, readonly=True)
        return True
    except Exception as e:
        print(f"Reconnect failed: {e}")
        return False


async def fetch_daily_history(ib, contract, years):
    if years <= 0:
        return []

    # IB requires duration in years for windows >= 365 days
    if years >= 1.0:
        duration_years = max(1, int(math.ceil(years)))
        duration_str = f"{duration_years} Y"
    else:
        duration_days = max(10, int(years * 365))
        duration_str = f"{duration_days} D"

    if not await ensure_connected(ib):
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
        await asyncio.sleep(API_PAUSE_SEC)
        return bars or []
    except Exception as e:
        print(f"  Historical data fetch failed for {contract.symbol}: {e}")
        return []


async def fetch_symbol_snapshot(ib, symbol, max_wait=6):
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

    # underlying price + 5-day momentum
    bars = await fetch_daily_history(ib, contract, years=0.03)  # ~11 days
    momentum = None
    if len(bars) >= 6:
        closes = [b.close for b in bars]
        if closes[-6] and closes[-1] and closes[-6] > 0:
            momentum = (closes[-1] / closes[-6]) - 1.0

    # option chain quick pull
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

    # underlying price for strike selection
    ticker_under = ib.reqMktData(contract, '', False, False)
    price = None
    for _ in range(40):
        if ticker_under.last is not None and not np.isnan(ticker_under.last):
            price = ticker_under.last
            break
        if ticker_under.close is not None and not np.isnan(ticker_under.close):
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
            await asyncio.sleep(API_PAUSE_SEC)
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


def compute_calibration(prices, implied_vol_hint):
    """
    Compute annualized drift/vol and realized/implied ratio.
    """
    if len(prices) < 30:
        return None
    rets = np.log(prices / prices.shift(1)).dropna()
    if rets.empty:
        return None

    mu_ann = float(rets.mean() * 252.0)
    sigma_ann = float(rets.std(ddof=0) * np.sqrt(252.0))

    window = min(len(rets), int(REALIZED_VOL_YEARS * 252))
    realized_short = float(rets.tail(window).std(ddof=0) * np.sqrt(252.0))

    momentum_short = None
    if len(prices) >= MOMENTUM_WINDOW_DAYS:
        momentum_short = float(prices.iloc[-1] / prices.iloc[-MOMENTUM_WINDOW_DAYS] - 1.0)

    realized_vol_frac = None
    if implied_vol_hint and implied_vol_hint > 0:
        realized_vol_frac = realized_short / implied_vol_hint
        realized_vol_frac = max(REALIZED_FRAC_MIN, min(REALIZED_FRAC_MAX, realized_vol_frac))

    return {
        'mu': mu_ann,
        'sigma_hist': sigma_ann,
        'realized_vol_frac': realized_vol_frac,
        'momentum_short': momentum_short
    }


async def calibrate_symbol(ib, contract, implied_vol_hint, cache):
    sym = contract.symbol
    cached = cache.get(sym)
    now = pd.Timestamp.now()
    if cached:
        ts = pd.to_datetime(cached.get('timestamp')) if cached.get('timestamp') else None
        if ts is not None and (now - ts).days < CALIBRATION_MAX_AGE_DAYS:
            return cached, False

    bars = await fetch_daily_history(ib, contract, MU_SIGMA_YEARS)
    if not bars:
        fallback = {
            'mu': MU_DRIFT,
            'sigma_hist': None,
            'realized_vol_frac': REALIZED_VOL_FRACTION,
            'momentum_short': None,
            'timestamp': now.isoformat()
        }
        cache[sym] = fallback
        return fallback, True

    closes = pd.Series([b.close for b in bars], index=[pd.to_datetime(b.date) for b in bars])
    calc = compute_calibration(closes, implied_vol_hint)
    if calc is None:
        fallback = {
            'mu': MU_DRIFT,
            'sigma_hist': None,
            'realized_vol_frac': REALIZED_VOL_FRACTION,
            'momentum_short': None,
            'timestamp': now.isoformat()
        }
        cache[sym] = fallback
        return fallback, True

    calc['timestamp'] = now.isoformat()
    cache[sym] = calc
    return calc, True

async def quick_option_liquidity_check(ib, contract, target_exp, strikes):
    """
    Probe 1-2 at-the-money options for basic liquidity; returns True/False.
    """
    if not strikes:
        return False
    if not await ensure_connected(ib):
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
        if (bid or ask) and vol >= MIN_OPT_VOL:
            return True
    return False

# ==========================================
# Async Logic
# ==========================================

async def main():

    ib = IB()

    # Silence benign scanner cancellation errors once the scan completes
    ib.errorEvent.clear()   # remove existing handlers

    def handle_error(reqId, errorCode, errorString, contract):
        # Ignore benign scanner cancellation and "no security definition" messages
        if errorCode in (162, 200, 2104, 2106, 2158, 365):
            return
        print(f"IB ERROR {errorCode} (reqId={reqId}): {errorString}")

    ib.errorEvent += handle_error

    try:
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID, readonly=True)
    except Exception as e:
        print(f"Connection failed: {e}. Is TWS/Gateway running?")
        return

    torch_device = choose_torch_device()
    print(f"Using torch device: {torch_device}")

    calibration_cache = load_calibration_cache()
    runtime_calibrations = {}
    cache_dirty = False

    # Request live data, if it fails use delayed frozen (15 min. delay, last
    # quote frozen on market close)
    try:
        ib.reqMarketDataType(1)
    except:
        ib.reqMarketDataType(4)

    print("Connected. Scanning for high option volume stocks...")

    # 1. Scanner: Find stocks with scanner

    # Filtering for US Major stocks with a price floor to ensure liquidity quality

    sub = ScannerSubscription(
        instrument='STK',
        locationCode='STK.US.MAJOR',
        scanCode='MOST_ACTIVE',  # default; overwritten per loop
        abovePrice=MIN_PRICE  # avoid penny stocks that often cause data issues
    )

    # Build universe: manual override -> multi-scan -> always-liquid fallback
    universe_syms = []
    if MANUAL_SYMBOLS:
        universe_syms.extend(MANUAL_SYMBOLS)
        print(f"Using manual symbol list: {MANUAL_SYMBOLS}")
    else:
        total_pulled = 0
        for sc in SCANNER_CODES:
            sub.scanCode = sc
            try:
                scan_data = await ib.reqScannerDataAsync(sub)
                pulled = [sd.contractDetails.contract.symbol for sd in scan_data[:SCAN_LIMIT]]
                universe_syms.extend(pulled)
                total_pulled += len(pulled)
                print(f"Scanner {sc}: pulled {len(pulled)} symbols")
            except Exception as e:
                print(f"Scanner {sc} failed: {e}")
                continue

    # dedupe preserving order (do not trim yet)
    seen = set()
    deduped = []
    for sym in universe_syms:
        if sym not in seen:
            seen.add(sym)
            deduped.append(sym)
    universe_syms = deduped

    # Quick ranking snapshot to choose top SCAN_LIMIT symbols
    ranked_metrics = []
    for sym in universe_syms:
        snap = await fetch_symbol_snapshot(ib, sym)
        if snap:
            ranked_metrics.append(snap)
    if ranked_metrics:
        opt_vol_vals = [m.get('opt_vol_raw') for m in ranked_metrics]
        oi_vals = [m.get('oi_raw') for m in ranked_metrics]
        iv_vals = [m.get('iv_raw') for m in ranked_metrics]
        momentum_vals = [m.get('momentum_raw') for m in ranked_metrics]
        spread_vals = [m.get('spread_quality_raw') for m in ranked_metrics]
        # iv_change not available -> zeros
        for m in ranked_metrics:
            m['opt_vol_score'] = percentile_rank(opt_vol_vals, m.get('opt_vol_raw'))
            m['oi_score'] = percentile_rank(oi_vals, m.get('oi_raw'))
            m['iv_rank_score'] = percentile_rank(iv_vals, m.get('iv_raw'))
            m['iv_change_score'] = 0.0
            m['momentum_score'] = percentile_rank(momentum_vals, m.get('momentum_raw'))
            m['spread_quality_score'] = percentile_rank(spread_vals, m.get('spread_quality_raw'))
            m['score'] = (
                0.40 * m['opt_vol_score'] +
                0.25 * m['oi_score'] +
                0.15 * m['iv_rank_score'] +
                0.10 * m['iv_change_score'] +
                0.10 * m['momentum_score']
            )
        ranked_metrics = sorted(ranked_metrics, key=lambda x: x['score'], reverse=True)
        universe_syms = [m['symbol'] for m in ranked_metrics[:SCAN_LIMIT]]
        print("Top ranked symbols:", universe_syms)
    else:
        universe_syms = universe_syms[:SCAN_LIMIT]

    if not universe_syms:
        fallback_symbols = ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT']
        universe_syms = fallback_symbols
        print(f"No scanner results; falling back to: {fallback_symbols}")

    print(f"Universe symbols: {universe_syms}")

    top_contracts = [Stock(sym, 'SMART', 'USD') for sym in universe_syms]

    opportunities = []
    mc_stats = defaultdict(int)

    symbols_bar = tqdm(total=len(top_contracts), desc="Symbols", leave=True)
    for contract in top_contracts:
        print(f"\nProcessing {contract.symbol}...")

        # Qualify the underlying contract (get ConID, etc)
        if not await ensure_connected(ib):
            break
        try:
            await ib.qualifyContractsAsync(contract)
            await asyncio.sleep(API_PAUSE_SEC)
        except Exception as e:
            print(f"  Qualification failed: {e}")
            continue

        # Get current underlying market data (for price reference)
        ticker = ib.reqMktData(contract, '', False, False)

        # Wait briefly for price data with fallbacks to close price
        underlying_price = None
        for _ in range(50):
            if ticker.last is not None and not np.isnan(ticker.last):
                underlying_price = ticker.last
                break
            if ticker.close is not None and not np.isnan(ticker.close):
                underlying_price = ticker.close
                break
            await asyncio.sleep(0.1)

        if underlying_price is None or underlying_price <= 0:
            print("  Skipping: unable to get valid underlying price.")
            continue

        print(f"  Underlying Price: {underlying_price}")

        # Skip illiquid/low-priced names that rarely have meaningful option markets
        if underlying_price < MIN_PRICE:
            log_filter(f"Underlying below ${MIN_PRICE}", contract.symbol)
            continue

        # 2. Get Option Chains
        try:
            chains = await ib.reqSecDefOptParamsAsync(contract.symbol, '', contract.secType, contract.conId)
        except Exception as e:
            print(f"  Failed to get chains: {e}")
            continue

        # Filter for SMART exchange to ensure we get general liquidity
        smart_chains = [c for c in chains if c.exchange == 'SMART']

        if not smart_chains:
            log_filter("No SMART exchange option chains available", contract.symbol)
            continue

        # Pick the first SMART chain; most underlyings only have one
        chain = smart_chains[0]
        chain_expirations = sorted([pd.to_datetime(exp) for exp in chain.expirations])
        print(f"  Using chain exchange={chain.exchange}, tradingClass={chain.tradingClass}, expirations(head)= {[d.strftime('%Y-%m-%d') for d in chain_expirations[:3]]}")

        now = pd.Timestamp.now()

        # All expirations strictly after now
        candidate_exps = [d for d in chain_expirations if (d - now).days > MIN_DAYS]
        if not candidate_exps:
            print("  Skipping: no usable expirations found.")
            continue

        # Keep only expirations within the MAX_DAYS window
        candidate_exps = [d for d in candidate_exps if (d - now).days <= MAX_DAYS]
        if not candidate_exps:
            print(f"  Skipping: all expirations are beyond {MAX_DAYS} days.")
            continue

        # Take up to the first 3 expirations in that window
        exp_dates = candidate_exps[:3]
        print("  Will scan expirations:", [d.strftime('%Y-%m-%d') for d in exp_dates])

        # Pre-calibration and symbol-level vol premium gate
        calib = runtime_calibrations.get(contract.symbol)
        if calib is None:
            calib, updated = await calibrate_symbol(ib, contract, None, calibration_cache)
            runtime_calibrations[contract.symbol] = calib
            cache_dirty = cache_dirty or updated

        rv_frac_filter_sym = calib.get('realized_vol_frac') if calib else None
        if rv_frac_filter_sym is None:
            rv_frac_filter_sym = REALIZED_VOL_FRACTION
        if rv_frac_filter_sym:
            vol_premium_ratio_sym = 1.0 / rv_frac_filter_sym
            if vol_premium_ratio_sym < VOL_PREMIUM_MIN:
                log_filter(f"Vol premium {vol_premium_ratio_sym:.2f} < {VOL_PREMIUM_MIN}", contract.symbol)
                continue

        # Quick liquidity check using nearest strikes on first expiry
        first_exp = exp_dates[0]
        strikes_for_liq = sorted([k for k in chain.strikes if 0.95 * underlying_price < k < 1.05 * underlying_price])
        if not strikes_for_liq:
            strikes_for_liq = sorted(chain.strikes)[:2]
        target_exp_liq = first_exp.strftime('%Y%m%d')
        if not await quick_option_liquidity_check(ib, contract, target_exp_liq, strikes_for_liq):
            print("  Skipping: failed quick option liquidity probe.")
            continue

        # >>> Everything that used to be "single-expiry" logic now goes inside this loop:
        for target_date in exp_dates:
            target_exp = target_date.strftime('%Y%m%d')
            days_to_expiry = (target_date - now).days
            T = days_to_expiry / 365.0
            print(f"  Selected expiration {target_exp} (~{days_to_expiry} days)")

            # --- keep the rest of your per-expiry logic under here ---
            # strikes = ...
            # build contracts_to_req ...
            # "Requesting data for X contracts..."
            # tickers = ...
            # df_chain / df_calls / df_puts ...
            # all four spread loops + Monte Carlo, etc.


            # Select Strikes: +/- percent of current price to keep request count low
            strikes = [k for k in chain.strikes if 0.85 * underlying_price < k < 1.15 * underlying_price]
            print(f"  Candidate strikes within +/-15%: {len(strikes)}")

            # Create Contract Objects for Calls and Puts (use contract details to ensure valid conIds)
            contracts_to_req = []
            trading_class = chain.tradingClass or contract.symbol
            multiplier = chain.multiplier or '100'

            for strike in strikes:
                for right in ['C', 'P']:
                    base_opt = dict(
                        symbol=contract.symbol,
                        lastTradeDateOrContractMonth=target_exp,
                        strike=strike,
                        right=right,
                        currency=contract.currency or 'USD',
                        multiplier=multiplier,
                        tradingClass=trading_class
                    )

                    exchanges_to_try = [chain.exchange] if chain.exchange else ['SMART']
                    for ex in exchanges_to_try:
                        c = Option(exchange=ex, **base_opt)
                        try:
                            cds = await ib.reqContractDetailsAsync(c)
                            if cds:
                                contracts_to_req.append(cds[0].contract)
                                break
                        except Exception:
                            continue

            if not contracts_to_req:
                log_filter("No option contracts qualified on available exchanges", contract.symbol, target_exp)
                continue
            if len(contracts_to_req) < 2:
                log_filter("Fewer than 2 option contracts qualified on exchanges", contract.symbol, target_exp)
                continue

            # 3. Batch Request Market Data for Options

            # We request '106' (Snapshot) and generic tick '100,101,104' (Option Greeks)

            # IBKR Tick ID 13 provides 'modelOptComp' (Model Greeks)

            print(f"  Requesting data for {len(contracts_to_req)} contracts...")

            tickers = []

            md_bar = tqdm(total=len(contracts_to_req), desc=f"{contract.symbol} {target_exp} md", leave=False)
            for c in contracts_to_req:
                # Request option volume (100) and implied vol (106); 13 is not valid for OPT generic ticks
                if not await ensure_connected(ib):
                    break
                t = ib.reqMktData(c, genericTickList='100,106', snapshot=False, regulatorySnapshot=False)
                await asyncio.sleep(API_PAUSE_SEC)
                tickers.append(t)
                md_bar.update(1)

            md_bar.close()

            # Allow time for data to populate with early-exit polling
            await wait_for_quotes(
                tickers,
                max_wait=10,
                ready_ratio=0.4,
                stale_cutoff=1.5,
                poll_interval=0.5,
                min_wait_before_early_exit=2.0,
                desc=f"{contract.symbol} {target_exp} settle"
            )

            # Process Data into a DataFrame
            chain_data = []

            parse_bar = tqdm(total=len(tickers), desc=f"{contract.symbol} {target_exp} parse", leave=False)
            for t in tickers:
                # Collect prices with fallbacks for after-hours/delayed data
                iv = t.modelGreeks.impliedVol if t.modelGreeks else t.impliedVolatility
                iv = float(iv) if iv is not None else None
                theo_price = t.modelGreeks.optPrice if t.modelGreeks else None
                theo_price = float(theo_price) if theo_price is not None else None
                if theo_price is not None and not math.isfinite(theo_price):
                    theo_price = None

                bid = t.bid if t.bid and t.bid > 0 else None
                ask = t.ask if t.ask and t.ask > 0 else None
                bid = float(bid) if bid is not None else None
                ask = float(ask) if ask is not None else None
                mid = None
                if bid and ask:
                    mid = (bid + ask) / 2
                if not bid and mid:
                    bid = mid
                if not ask and mid:
                    ask = mid
                if not bid and not ask and theo_price and theo_price > 0:
                    bid = theo_price
                    ask = theo_price

                volume = t.volume or t.bidSize or t.askSize or t.lastSize or 0

                right = getattr(t.contract, 'right', None)

                # Require finite positive IV and prices to avoid NaNs in the MC sim
                if (
                    iv is not None and math.isfinite(iv) and iv > 0 and
                    bid is not None and ask is not None and
                    math.isfinite(bid) and math.isfinite(ask) and
                    bid > 0 and ask > 0 and
                    volume is not None and right
                ):
                    chain_data.append({
                        'contract': t.contract,
                        'strike': t.contract.strike,
                        'right': right,
                        'bid': bid,
                        'ask': ask,
                        'volume': volume,
                        'iv': iv
                    })
                parse_bar.update(1)
            parse_bar.close()

            if not chain_data:
                log_filter("No valid option data received", contract.symbol, target_exp)
                continue

            df_chain = pd.DataFrame(chain_data).sort_values(['right', 'strike'])

            df_calls = df_chain[df_chain['right'] == 'C']
            df_puts = df_chain[df_chain['right'] == 'P']

            iv_median = float(df_chain['iv'].median()) if not df_chain.empty else None
            if iv_median is not None and (not math.isfinite(iv_median) or iv_median <= 0):
                iv_median = None

            calib = runtime_calibrations.get(contract.symbol)
            if calib is None:
                calib, updated = await calibrate_symbol(ib, contract, iv_median, calibration_cache)
                runtime_calibrations[contract.symbol] = calib
                cache_dirty = cache_dirty or updated

            sigma_fallback = calib.get('sigma_hist')
            mu_param = calib.get('mu') if USE_REAL_WORLD_DRIFT else None
            rv_frac_param = calib.get('realized_vol_frac')
            rv_frac_filter = rv_frac_param if rv_frac_param is not None else REALIZED_VOL_FRACTION

            if rv_frac_filter is None:
                log_filter("No realized vol calibration available", contract.symbol, target_exp)
                continue

            iv_realized_ratio = 1.0 / rv_frac_filter if rv_frac_filter > 0 else 0.0
            if iv_realized_ratio < IV_REALIZED_RATIO_MIN:
                log_filter(f"IV/Realized ratio {iv_realized_ratio:.2f} < {IV_REALIZED_RATIO_MIN}", contract.symbol, target_exp)
                continue

            if not USE_VOL_RISK_PREMIUM:
                rv_frac_param = None

            momentum_short = calib.get('momentum_short')
            trend = 'flat'
            if mu_param is not None and mu_param > MU_TREND_THRESH and (momentum_short is None or momentum_short >= 0):
                trend = 'up'
            elif mu_param is not None and mu_param < -MU_TREND_THRESH and (momentum_short is None or momentum_short <= 0):
                trend = 'down'

            # Skew metric: ATM vs ~10% OTM put IV
            def nearest_iv(df, target_strike):
                if df.empty:
                    return None
                idx = (df['strike'] - target_strike).abs().idxmin()
                val = df.loc[idx, 'iv']
                return float(val) if val is not None and math.isfinite(val) else None

            atm_iv = nearest_iv(df_puts, underlying_price)
            otm_put_iv = nearest_iv(df_puts, 0.9 * underlying_price)
            skew_metric = None
            if atm_iv and otm_put_iv:
                skew_metric = otm_put_iv - atm_iv

            # 4. Construct Vertical Spreads (Calls and Puts, Debit and Credit)
            bull_call_bar = tqdm(total=count_upper_pairs(df_calls), desc=f"{contract.symbol} {target_exp} bull calls", leave=False) if not df_calls.empty else None
            bear_call_bar = tqdm(total=count_upper_pairs(df_calls), desc=f"{contract.symbol} {target_exp} bear calls", leave=False) if not df_calls.empty else None
            bull_put_bar = tqdm(total=count_lower_pairs(df_puts), desc=f"{contract.symbol} {target_exp} bull puts", leave=False) if not df_puts.empty else None
            bear_put_bar = tqdm(total=count_lower_pairs(df_puts), desc=f"{contract.symbol} {target_exp} bear puts", leave=False) if not df_puts.empty else None

            # Bull Call Debit: long lower call, short higher call
            #if trend == 'down':
            #    log_filter("Trend down; skipping bull call candidates", contract.symbol, target_exp, "Bull Call")
            #else:
            for i, long_leg in df_calls.iterrows():
                spread_prefix = f"Bull Call long {long_leg['strike']}"
                if long_leg['volume'] < MIN_OPT_VOL:
                    log_filter(f"Long leg volume {long_leg['volume']} < {MIN_OPT_VOL}", contract.symbol, target_exp, spread_prefix)
                    continue
                short_legs = df_calls[df_calls['strike'] > long_leg['strike']]
                for j, short_leg in short_legs.iterrows():
                    bar_update(bull_call_bar)
                    spread_desc = f"Bull Call {long_leg['strike']}->{short_leg['strike']}"
                    if short_leg['volume'] < MIN_OPT_VOL:
                        log_filter(f"Short leg volume {short_leg['volume']} < {MIN_OPT_VOL}", contract.symbol, target_exp, spread_desc)
                        continue

                    tight_long, spread_long, mid_long = leg_is_tight(long_leg)
                    tight_short, spread_short, mid_short = leg_is_tight(short_leg)
                    if not (tight_long and tight_short):
                        log_filter(
                            f"Bid/ask too wide: long {spread_long:.2f} (mid {mid_long:.2f}), short {spread_short:.2f} (mid {mid_short:.2f})",
                            contract.symbol, target_exp, spread_desc)
                        continue

                    cost = long_leg['ask'] - short_leg['bid']
                    if not (MIN_ENTRY < cost < MAX_DEBIT):
                        log_filter(f"Cost {cost:.2f} outside ({MIN_ENTRY}, {MAX_DEBIT})", contract.symbol, target_exp, spread_desc)
                        continue

                    width = short_leg['strike'] - long_leg['strike']
                    if width < MIN_WIDTH or width > MAX_WIDTH:
                        log_filter(f"Width {width:.2f} outside {MIN_WIDTH}-{MAX_WIDTH}", contract.symbol, target_exp, spread_desc)
                        continue
                    max_profit = width - cost
                    if max_profit <= 0:
                        log_filter("Non-positive max profit", contract.symbol, target_exp, spread_desc)
                        continue

                    be_price = long_leg['strike'] + cost
                    avg_iv = (long_leg['iv'] + short_leg['iv']) / 2
                    skew_capture = short_leg['iv'] - long_leg['iv']

                    sigma_input = avg_iv if avg_iv and avg_iv > 0 else sigma_fallback
                    if sigma_input is None or sigma_input <= 0:
                        log_filter("No usable sigma for MC", contract.symbol, target_exp, spread_desc)
                        continue
                    if skew_capture < MIN_SKEW_CAPTURE_BULL:
                        log_filter(f"Skew capture {skew_capture:.3f} < {MIN_SKEW_CAPTURE_BULL}", contract.symbol, target_exp, spread_desc)
                        continue

                    # Canonical call vertical: long lower, short higher
                    long_strike = long_leg['strike']    # lower
                    short_strike = short_leg['strike']  # higher

                    log_pass("Running MC", contract.symbol, target_exp, spread_desc)
                    mc_stats['mc_runs'] += 1
                    pop_mc, ev_mc, avg_ht, p_daytrade = simulate_vertical_pop_ev_torch(
                        S0=underlying_price,
                        r=RISK_FREE_RATE,
                        sigma=sigma_input,
                        T=T,
                        long_strike=long_strike,
                        short_strike=short_strike,
                        right='C',
                        entry=cost,
                        max_profit=max_profit,
                        max_loss=cost,        # debit = max loss
                        entry_type='Debit',
                        tp_frac=MC_TP_FRAC,
                        sl_frac=MC_SL_DEBIT_FRAC,
                        n_paths=MC_N_PATHS,
                        n_steps=MC_N_STEPS,
                        daytrade_threshold_days=MC_DAYTRADE_THRESHOLD_DAYS,
                        device=None,
                        mu=mu_param,
                        realized_vol_frac=rv_frac_param
                    )

                    if pop_mc <= MIN_POP:
                        mc_stats['pop_fail'] += 1
                        log_filter(f"PoP {pop_mc:.3f} <= {MIN_POP}", contract.symbol, target_exp, spread_desc)
                        continue

                    ev = ev_mc
                    if REQUIRE_POSITIVE_EV and ev <= 0:
                        mc_stats['ev_fail'] += 1
                        log_filter(f"EV {ev:.3f} <= 0 with positive EV required", contract.symbol, target_exp, spread_desc)
                        continue

                    roi_trade = ev / cost
                    ev_per_risk = roi_trade
                    roi_monthly = roi_trade * (30.0 / days_to_expiry)

                    if ev < MIN_EV:
                        log_filter(f"EV {ev:.3f} < MIN_EV {MIN_EV}", contract.symbol, target_exp, spread_desc)
                        continue
                    if ev_per_risk < MIN_EV_PER_RISK:
                        log_filter(f"EV/risk {ev_per_risk:.3f} < {MIN_EV_PER_RISK}", contract.symbol, target_exp, spread_desc)
                        continue

                    hold_days = avg_ht * 365.0
                    if hold_days < MIN_HOLD_DAYS:
                        mc_stats['hold_fail'] += 1
                        log_filter(f"Avg hold {hold_days:.2f}d < {MIN_HOLD_DAYS}d", contract.symbol, target_exp, spread_desc)
                        continue
                    if roi_monthly < TARGET_MONTHLY_RETURN:
                        mc_stats['roi_fail'] += 1
                        log_filter(f"ROI/month {roi_monthly:.3f} < target {TARGET_MONTHLY_RETURN}", contract.symbol, target_exp, spread_desc)
                        continue

                    opportunities.append({
                        'Symbol': contract.symbol,
                        'Expiry': target_exp,
                        'Strategy': f"{long_leg['strike']}/{short_leg['strike']} Bull Call",
                        'Entry': round(cost, 2),
                        'Entry Type': 'Debit',
                        'Max Profit': round(max_profit, 2),
                        'Max Loss': round(cost, 2),
                        'PoP': round(pop_mc, 3),
                        'EV': round(ev, 2),
                        'EV_Per_Risk': round(ev_per_risk, 3),
                        'Skew_Capture': round(skew_capture, 3),
                        'Skew_Expiry': round(skew_metric, 3) if skew_metric is not None else None,
                        'Trend': trend,
                        'ROI_Monthly': round(roi_monthly, 3),
                        'AvgHold_days': round(hold_days, 1),
                        'P_Daytrade': round(p_daytrade, 3),
                        'IV': round(avg_iv, 3),
                        'Underlying': round(underlying_price, 2),
                        'BE': round(be_price, 2)
                    })
                    log_pass(f"MC accepted: PoP {pop_mc:.3f}, EV {ev:.2f}, ROI/mo {roi_monthly:.3f}", contract.symbol, target_exp, spread_desc)
                    mc_stats['accepted'] += 1
                                       
            # Bear Call Credit: short lower call, long higher call
            if trend == 'up':
                log_filter("Trend up; skipping bear call candidates", contract.symbol, target_exp, "Bear Call")
            else:
                for i, short_leg in df_calls.iterrows():
                    spread_prefix = f"Bear Call short {short_leg['strike']}"
                    if short_leg['volume'] < MIN_OPT_VOL:
                        log_filter(f"Short leg volume {short_leg['volume']} < {MIN_OPT_VOL}", contract.symbol, target_exp, spread_prefix)
                        continue
                    long_legs = df_calls[df_calls['strike'] > short_leg['strike']]
                    for j, long_leg in long_legs.iterrows():
                        bar_update(bear_call_bar)
                        spread_desc = f"Bear Call {short_leg['strike']}->{long_leg['strike']}"
                        if long_leg['volume'] < MIN_OPT_VOL:
                            log_filter(f"Long leg volume {long_leg['volume']} < {MIN_OPT_VOL}", contract.symbol, target_exp, spread_desc)
                            continue

                        tight_long, spread_long, mid_long = leg_is_tight(long_leg)
                        tight_short, spread_short, mid_short = leg_is_tight(short_leg)
                        if not (tight_long and tight_short):
                            log_filter(
                                f"Bid/ask too wide: short {spread_short:.2f} (mid {mid_short:.2f}), long {spread_long:.2f} (mid {mid_long:.2f})",
                                contract.symbol, target_exp, spread_desc)
                            continue

                    credit = short_leg['bid'] - long_leg['ask']
                    if credit <= MIN_ENTRY:
                        log_filter(f"Credit {credit:.2f} <= MIN_ENTRY {MIN_ENTRY}", contract.symbol, target_exp, spread_desc)
                        continue

                        width = long_leg['strike'] - short_leg['strike']
                        if width < MIN_WIDTH or width > MAX_WIDTH:
                            log_filter(f"Width {width:.2f} outside {MIN_WIDTH}-{MAX_WIDTH}", contract.symbol, target_exp, spread_desc)
                            continue
                        if width <= 0:
                            log_filter("Width non-positive", contract.symbol, target_exp, spread_desc)
                            continue
                        if credit >= width:  # inverted/arb spread; data likely stale
                            log_filter(f"Credit {credit:.2f} >= width {width:.2f}", contract.symbol, target_exp, spread_desc)
                            continue

                        max_loss = width - credit
                        if max_loss <= 0:
                            log_filter(f"Max loss {max_loss:.2f} <= 0", contract.symbol, target_exp, spread_desc)
                            continue

                        be_price = short_leg['strike'] + credit
                        avg_iv = (long_leg['iv'] + short_leg['iv']) / 2
                        skew_capture = short_leg['iv'] - long_leg['iv']
                        sigma_input = avg_iv if avg_iv and avg_iv > 0 else sigma_fallback
                        if sigma_input is None or sigma_input <= 0:
                            log_filter("No usable sigma for MC", contract.symbol, target_exp, spread_desc)
                            continue
                        if skew_capture < MIN_SKEW_CAPTURE_BEAR:
                            log_filter(f"Skew capture {skew_capture:.3f} < {MIN_SKEW_CAPTURE_BEAR}", contract.symbol, target_exp, spread_desc)
                            continue

                        # Canonical call vertical: long lower, short higher
                        long_strike = short_leg['strike']   # lower
                        short_strike = long_leg['strike']   # higher

                        log_pass("Running MC", contract.symbol, target_exp, spread_desc)
                        mc_stats['mc_runs'] += 1
                        pop_mc, ev_mc, avg_ht, p_daytrade = simulate_vertical_pop_ev_torch(
                            S0=underlying_price,
                            r=RISK_FREE_RATE,
                            sigma=sigma_input,
                            T=T,
                            long_strike=long_strike,
                            short_strike=short_strike,
                        right='C',
                        entry=credit,
                        max_profit=credit,
                        max_loss=max_loss,
                        entry_type='Credit',  # short the canonical vertical
                        tp_frac=MC_TP_FRAC,
                        sl_frac=MC_SL_CREDIT_MULT,
                        n_paths=MC_N_PATHS,
                        n_steps=MC_N_STEPS,
                            daytrade_threshold_days=MC_DAYTRADE_THRESHOLD_DAYS,
                            device=None,
                            mu=mu_param,
                            realized_vol_frac=rv_frac_param
                        )


                        if pop_mc <= MIN_POP:
                            mc_stats['pop_fail'] += 1
                            log_filter(f"PoP {pop_mc:.3f} <= {MIN_POP}", contract.symbol, target_exp, spread_desc)
                            continue

                        ev = ev_mc
                        if REQUIRE_POSITIVE_EV and ev <= 0:
                            mc_stats['ev_fail'] += 1
                            log_filter(f"EV {ev:.3f} <= 0 with positive EV required", contract.symbol, target_exp, spread_desc)
                            continue

                        roi_trade = ev / max_loss   # margin ≈ max_loss
                        ev_per_risk = roi_trade
                        roi_monthly = roi_trade * (30.0 / days_to_expiry)

                        if ev < MIN_EV:
                            log_filter(f"EV {ev:.3f} < MIN_EV {MIN_EV}", contract.symbol, target_exp, spread_desc)
                            continue
                        if ev_per_risk < MIN_EV_PER_RISK:
                            log_filter(f"EV/risk {ev_per_risk:.3f} < {MIN_EV_PER_RISK}", contract.symbol, target_exp, spread_desc)
                            continue
                        hold_days = avg_ht * 365.0
                        if hold_days < MIN_HOLD_DAYS:
                            mc_stats['hold_fail'] += 1
                            log_filter(f"Avg hold {hold_days:.2f}d < {MIN_HOLD_DAYS}d", contract.symbol, target_exp, spread_desc)
                            continue
                        if roi_monthly < TARGET_MONTHLY_RETURN:
                            mc_stats['roi_fail'] += 1
                            log_filter(f"ROI/month {roi_monthly:.3f} < target {TARGET_MONTHLY_RETURN}", contract.symbol, target_exp, spread_desc)
                            continue

                        opportunities.append({
                            'Symbol': contract.symbol,
                            'Expiry': target_exp,
                            'Strategy': f"{short_leg['strike']}/{long_leg['strike']} Bear Call",
                            'Entry': round(credit, 2),
                            'Entry Type': 'Credit',
                            'Max Profit': round(credit, 2),
                            'Max Loss': round(max_loss, 2),
                            'PoP': round(pop_mc, 3),
                            'EV': round(ev, 2),
                            'EV_Per_Risk': round(ev_per_risk, 3),
                            'Skew_Capture': round(skew_capture, 3),
                            'Skew_Expiry': round(skew_metric, 3) if skew_metric is not None else None,
                            'ROI_Monthly': round(roi_monthly, 3),
                            'AvgHold_days': round(hold_days, 1),
                            'P_Daytrade': round(p_daytrade, 3),
                            'IV': round(avg_iv, 3),
                            'Underlying': round(underlying_price, 2),
                            'BE': round(be_price, 2)
                        })
                        log_pass(f"MC accepted: PoP {pop_mc:.3f}, EV {ev:.2f}, ROI/mo {roi_monthly:.3f}", contract.symbol, target_exp, spread_desc)
                        mc_stats['accepted'] += 1
                                   
            # Bull Put Credit: short higher strike put, long lower strike put
            if trend == 'down':
                log_filter("Trend down; skipping bull put candidates", contract.symbol, target_exp, "Bull Put")
            else:
                for i, short_leg in df_puts.iterrows():
                    spread_prefix = f"Bull Put short {short_leg['strike']}"
                    if short_leg['volume'] < MIN_OPT_VOL:
                        log_filter(f"Short leg volume {short_leg['volume']} < {MIN_OPT_VOL}", contract.symbol, target_exp, spread_prefix)
                        continue
                    long_legs = df_puts[df_puts['strike'] < short_leg['strike']]
                    for j, long_leg in long_legs.iterrows():
                        bar_update(bull_put_bar)
                        spread_desc = f"Bull Put {short_leg['strike']}->{long_leg['strike']}"
                        if long_leg['volume'] < MIN_OPT_VOL:
                            log_filter(f"Long leg volume {long_leg['volume']} < {MIN_OPT_VOL}", contract.symbol, target_exp, spread_desc)
                            continue

                    tight_long, spread_long, mid_long = leg_is_tight(long_leg)
                    tight_short, spread_short, mid_short = leg_is_tight(short_leg)
                    if not (tight_long and tight_short):
                        log_filter(
                            f"Bid/ask too wide: short {spread_short:.2f} (mid {mid_short:.2f}), long {spread_long:.2f} (mid {mid_long:.2f})",
                            contract.symbol, target_exp, spread_desc)
                        continue

                    credit = short_leg['bid'] - long_leg['ask']
                    if credit <= MIN_ENTRY:
                        log_filter(f"Credit {credit:.2f} <= MIN_ENTRY {MIN_ENTRY}", contract.symbol, target_exp, spread_desc)
                        continue

                        width = short_leg['strike'] - long_leg['strike']
                        if width < MIN_WIDTH or width > MAX_WIDTH:
                            log_filter(f"Width {width:.2f} outside {MIN_WIDTH}-{MAX_WIDTH}", contract.symbol, target_exp, spread_desc)
                            continue
                        if width <= 0:
                            log_filter("Width non-positive", contract.symbol, target_exp, spread_desc)
                            continue
                        if credit >= width:  # inverted/arb spread; data likely stale
                            log_filter(f"Credit {credit:.2f} >= width {width:.2f}", contract.symbol, target_exp, spread_desc)
                            continue

                        max_loss = width - credit
                        if max_loss <= 0:
                            log_filter(f"Max loss {max_loss:.2f} <= 0", contract.symbol, target_exp, spread_desc)
                            continue

                        be_price = short_leg['strike'] - credit
                        avg_iv = (long_leg['iv'] + short_leg['iv']) / 2
                        skew_capture = short_leg['iv'] - long_leg['iv']
                        sigma_input = avg_iv if avg_iv and avg_iv > 0 else sigma_fallback
                        if sigma_input is None or sigma_input <= 0:
                            log_filter("No usable sigma for MC", contract.symbol, target_exp, spread_desc)
                            continue
                        if skew_capture < MIN_SKEW_CAPTURE_BULL:
                            log_filter(f"Skew capture {skew_capture:.3f} < {MIN_SKEW_CAPTURE_BULL}", contract.symbol, target_exp, spread_desc)
                            continue

                        # Canonical put vertical: long higher, short lower
                        long_strike = short_leg['strike']   # higher
                        short_strike = long_leg['strike']   # lower

                        log_pass("Running MC", contract.symbol, target_exp, spread_desc)
                        mc_stats['mc_runs'] += 1
                        pop_mc, ev_mc, avg_ht, p_daytrade = simulate_vertical_pop_ev_torch(
                            S0=underlying_price,
                            r=RISK_FREE_RATE,
                            sigma=sigma_input,
                            T=T,
                            long_strike=long_strike,
                            short_strike=short_strike,
                        right='P',
                        entry=credit,
                        max_profit=credit,
                        max_loss=max_loss,
                        entry_type='Credit',
                        tp_frac=MC_TP_FRAC,
                        sl_frac=MC_SL_CREDIT_MULT,
                        n_paths=MC_N_PATHS,
                        n_steps=MC_N_STEPS,
                            daytrade_threshold_days=MC_DAYTRADE_THRESHOLD_DAYS,
                            device=None,
                            mu=mu_param,
                            realized_vol_frac=rv_frac_param
                        )

                        if pop_mc <= MIN_POP:
                            mc_stats['pop_fail'] += 1
                            log_filter(f"PoP {pop_mc:.3f} <= {MIN_POP}", contract.symbol, target_exp, spread_desc)
                            continue

                        ev = ev_mc
                        if REQUIRE_POSITIVE_EV and ev <= 0:
                            mc_stats['ev_fail'] += 1
                            log_filter(f"EV {ev:.3f} <= 0 with positive EV required", contract.symbol, target_exp, spread_desc)
                            continue

                        roi_trade = ev / max_loss
                        ev_per_risk = roi_trade
                        roi_monthly = roi_trade * (30.0 / days_to_expiry)

                        if ev < MIN_EV:
                            log_filter(f"EV {ev:.3f} < MIN_EV {MIN_EV}", contract.symbol, target_exp, spread_desc)
                            continue
                        if ev_per_risk < MIN_EV_PER_RISK:
                            log_filter(f"EV/risk {ev_per_risk:.3f} < {MIN_EV_PER_RISK}", contract.symbol, target_exp, spread_desc)
                            continue
                        hold_days = avg_ht * 365.0
                        if hold_days < MIN_HOLD_DAYS:
                            mc_stats['hold_fail'] += 1
                            log_filter(f"Avg hold {hold_days:.2f}d < {MIN_HOLD_DAYS}d", contract.symbol, target_exp, spread_desc)
                            continue
                        if roi_monthly < TARGET_MONTHLY_RETURN:
                            mc_stats['roi_fail'] += 1
                            log_filter(f"ROI/month {roi_monthly:.3f} < target {TARGET_MONTHLY_RETURN}", contract.symbol, target_exp, spread_desc)
                            continue

                        opportunities.append({
                            'Symbol': contract.symbol,
                            'Expiry': target_exp,
                            'Strategy': f"{long_leg['strike']}/{short_leg['strike']} Bull Put",
                            'Entry': round(credit, 2),
                            'Entry Type': 'Credit',
                            'Max Profit': round(credit, 2),
                            'Max Loss': round(max_loss, 2),
                            'PoP': round(pop_mc, 3),
                            'EV': round(ev, 2),
                            'EV_Per_Risk': round(ev_per_risk, 3),
                            'Skew_Capture': round(skew_capture, 3),
                            'Skew_Expiry': round(skew_metric, 3) if skew_metric is not None else None,
                            'ROI_Monthly': round(roi_monthly, 3),
                            'AvgHold_days': round(hold_days, 1),
                            'P_Daytrade': round(p_daytrade, 3),
                            'IV': round(avg_iv, 3),
                            'Underlying': round(underlying_price, 2),
                            'BE': round(be_price, 2)
                        })
                        log_pass(f"MC accepted: PoP {pop_mc:.3f}, EV {ev:.2f}, ROI/mo {roi_monthly:.3f}", contract.symbol, target_exp, spread_desc)
                        mc_stats['accepted'] += 1
                                    
            # Bear Put Debit: long higher strike put, short lower strike put
            if trend == 'up':
                log_filter("Trend up; skipping bear put candidates", contract.symbol, target_exp, "Bear Put")
            else:
                for i, long_leg in df_puts.iterrows():
                    spread_prefix = f"Bear Put long {long_leg['strike']}"
                    if long_leg['volume'] < MIN_OPT_VOL:
                        log_filter(f"Long leg volume {long_leg['volume']} < {MIN_OPT_VOL}", contract.symbol, target_exp, spread_prefix)
                        continue
                    short_legs = df_puts[df_puts['strike'] < long_leg['strike']]
                    for j, short_leg in short_legs.iterrows():
                        bar_update(bear_put_bar)
                        spread_desc = f"Bear Put {long_leg['strike']}->{short_leg['strike']}"
                        if short_leg['volume'] < MIN_OPT_VOL:
                            log_filter(f"Short leg volume {short_leg['volume']} < {MIN_OPT_VOL}", contract.symbol, target_exp, spread_desc)
                            continue

                        tight_long, spread_long, mid_long = leg_is_tight(long_leg)
                        tight_short, spread_short, mid_short = leg_is_tight(short_leg)
                        if not (tight_long and tight_short):
                            log_filter(
                                f"Bid/ask too wide: long {spread_long:.2f} (mid {mid_long:.2f}), short {spread_short:.2f} (mid {mid_short:.2f})",
                                contract.symbol, target_exp, spread_desc)
                            continue

                    cost = long_leg['ask'] - short_leg['bid']
                    if not (MIN_ENTRY < cost < MAX_DEBIT):
                        log_filter(f"Cost {cost:.2f} outside ({MIN_ENTRY}, {MAX_DEBIT})", contract.symbol, target_exp, spread_desc)
                        continue

                        width = long_leg['strike'] - short_leg['strike']
                        if width < MIN_WIDTH or width > MAX_WIDTH:
                            log_filter(f"Width {width:.2f} outside {MIN_WIDTH}-{MAX_WIDTH}", contract.symbol, target_exp, spread_desc)
                            continue
                        max_profit = width - cost
                        if max_profit <= 0:
                            log_filter("Non-positive max profit", contract.symbol, target_exp, spread_desc)
                            continue

                        be_price = long_leg['strike'] - cost
                        avg_iv = (long_leg['iv'] + short_leg['iv']) / 2
                        skew_capture = short_leg['iv'] - long_leg['iv']
                        sigma_input = avg_iv if avg_iv and avg_iv > 0 else sigma_fallback
                        if sigma_input is None or sigma_input <= 0:
                            log_filter("No usable sigma for MC", contract.symbol, target_exp, spread_desc)
                            continue
                        if skew_capture < MIN_SKEW_CAPTURE_BEAR:
                            log_filter(f"Skew capture {skew_capture:.3f} < {MIN_SKEW_CAPTURE_BEAR}", contract.symbol, target_exp, spread_desc)
                            continue

                        # Canonical put vertical: long higher, short lower
                        long_strike = long_leg['strike']    # higher
                        short_strike = short_leg['strike']  # lower

                        log_pass("Running MC", contract.symbol, target_exp, spread_desc)
                        mc_stats['mc_runs'] += 1
                        pop_mc, ev_mc, avg_ht, p_daytrade = simulate_vertical_pop_ev_torch(
                            S0=underlying_price,
                            r=RISK_FREE_RATE,
                            sigma=sigma_input,
                            T=T,
                            long_strike=long_strike,
                            short_strike=short_strike,
                        right='P',
                        entry=cost,
                        max_profit=max_profit,
                        max_loss=cost,          # debit = max loss
                        entry_type='Debit',
                        tp_frac=MC_TP_FRAC,
                        sl_frac=MC_SL_DEBIT_FRAC,
                        n_paths=MC_N_PATHS,
                        n_steps=MC_N_STEPS,
                            daytrade_threshold_days=MC_DAYTRADE_THRESHOLD_DAYS,
                            device=None,
                            mu=mu_param,
                            realized_vol_frac=rv_frac_param
                        )

                        if pop_mc <= MIN_POP:
                            mc_stats['pop_fail'] += 1
                            log_filter(f"PoP {pop_mc:.3f} <= {MIN_POP}", contract.symbol, target_exp, spread_desc)
                            continue

                        ev = ev_mc
                        if REQUIRE_POSITIVE_EV and ev <= 0:
                            mc_stats['ev_fail'] += 1
                            log_filter(f"EV {ev:.3f} <= 0 with positive EV required", contract.symbol, target_exp, spread_desc)
                            continue

                        roi_trade = ev / cost
                        ev_per_risk = roi_trade
                        roi_monthly = roi_trade * (30.0 / days_to_expiry)

                        if ev < MIN_EV:
                            log_filter(f"EV {ev:.3f} < MIN_EV {MIN_EV}", contract.symbol, target_exp, spread_desc)
                            continue
                        if ev_per_risk < MIN_EV_PER_RISK:
                            log_filter(f"EV/risk {ev_per_risk:.3f} < {MIN_EV_PER_RISK}", contract.symbol, target_exp, spread_desc)
                            continue
                        hold_days = avg_ht * 365.0
                        if hold_days < MIN_HOLD_DAYS:
                            mc_stats['hold_fail'] += 1
                            log_filter(f"Avg hold {hold_days:.2f}d < {MIN_HOLD_DAYS}d", contract.symbol, target_exp, spread_desc)
                            continue
                        if roi_monthly < TARGET_MONTHLY_RETURN:
                            mc_stats['roi_fail'] += 1
                            log_filter(f"ROI/month {roi_monthly:.3f} < target {TARGET_MONTHLY_RETURN}", contract.symbol, target_exp, spread_desc)
                            continue

                        opportunities.append({
                            'Symbol': contract.symbol,
                            'Expiry': target_exp,
                            'Strategy': f"{short_leg['strike']}/{long_leg['strike']} Bear Put",
                            'Entry': round(cost, 2),
                            'Entry Type': 'Debit',
                            'Max Profit': round(max_profit, 2),
                            'Max Loss': round(cost, 2),
                            'PoP': round(pop_mc, 3),
                            'EV': round(ev, 2),
                            'EV_Per_Risk': round(ev_per_risk, 3),
                            'Skew_Capture': round(skew_capture, 3),
                            'Skew_Expiry': round(skew_metric, 3) if skew_metric is not None else None,
                            'ROI_Monthly': round(roi_monthly, 3),
                            'AvgHold_days': round(hold_days, 1),
                            'P_Daytrade': round(p_daytrade, 3),
                            'IV': round(avg_iv, 3),
                            'Underlying': round(underlying_price, 2),
                            'BE': round(be_price, 2)
                        })
                        log_pass(f"MC accepted: PoP {pop_mc:.3f}, EV {ev:.2f}, ROI/mo {roi_monthly:.3f}", contract.symbol, target_exp, spread_desc)
                        mc_stats['accepted'] += 1
                    mc_stats['accepted'] += 1

            if bull_call_bar: bull_call_bar.close()
            if bear_call_bar: bear_call_bar.close()
            if bull_put_bar: bull_put_bar.close()
            if bear_put_bar: bear_put_bar.close()

        symbols_bar.update(1)

    if cache_dirty:
        save_calibration_cache(calibration_cache)

    symbols_bar.close()
    ib.disconnect()

    # 5. Output Results

    if opportunities:
        df_res = pd.DataFrame(opportunities)

        if 'EV_Per_Risk' in df_res.columns:
            df_res = df_res.sort_values(['EV_Per_Risk', 'PoP'], ascending=False)
        else:
            df_res = df_res.sort_values(['ROI_Monthly', 'PoP'], ascending=False)

        print("\n=== HIGH VOLUME VERTICAL SPREAD OPPORTUNITIES ===")
        print(df_res.to_string(index=False))

        # Save to CSV for analysis
        # df_res.to_csv('ibkr_spreads.csv', index=False)
    else:
        print("\nNo spreads found matching criteria.")
        if mc_stats:
            print(f"  MC runs: {mc_stats.get('mc_runs', 0)} | accepted: {mc_stats.get('accepted', 0)} | PoP fails: {mc_stats.get('pop_fail', 0)} | EV fails: {mc_stats.get('ev_fail', 0)} | ROI fails: {mc_stats.get('roi_fail', 0)} | hold-time fails: {mc_stats.get('hold_fail', 0)}")

if __name__ == '__main__':
    asyncio.run(main())
