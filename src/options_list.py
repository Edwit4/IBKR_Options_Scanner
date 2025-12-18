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
from pathlib import Path
import os
from collections import defaultdict
from filters import log_filter, log_pass, set_filter_verbose
from calibration import (
    load_calibration_cache,
    save_calibration_cache,
    ensure_connected,
    fetch_daily_history,
    calibrate_symbol,
)
from mc import (
    bs_option_price_torch,
    vertical_price_torch,
    vertical_intrinsic_torch,
    simulate_vertical_pop_ev_torch,
    choose_torch_device,
)
from utils import (
    count_upper_pairs,
    count_lower_pairs,
    wait_for_quotes,
    percentile_rank,
    fetch_symbol_snapshot,
    quick_option_liquidity_check,
)
from strategies import evaluate_vertical_candidate
from scan_strategies import VerticalStrategy, run_vertical_strategy
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
#PORT = 7496 # Live trading
PORT = 7497 # Paper trading
CLIENT_ID = 99

# Strategy Parameters
MAX_DEBIT = 3.00       # Max cost
MIN_OPT_VOL = 5        # Liquidity filter for individual legs
MIN_POP = 0.2         # Min Probability of Profit
RISK_FREE_RATE_FALLBACK = 0.044 # ~4.4% (Used if IBKR yields are unavailable)
SCAN_LIMIT = 50         # Limit scanner result count to avoid pacing issues
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
# Portfolio-level risk budget (average fraction of capital actually at risk in these spreads)
PORTFOLIO_RISK_BUDGET_FRAC = 0.4   # example: 40% of capital is in spreads on average
TARGET_MONTHLY_RETURN = 0.05   # Percent per month, as a fraction
REQUIRE_POSITIVE_EV = True     # Only keep spreads with positive expectation
MIN_DAYS = 0
MAX_DAYS = 60
VOL_PREMIUM_MIN = 1.0  # require IV/realized >= value at symbol level to proceed

# MC Params
MC_TP_FRAC = 0.5        # take profit at percent of max profit
MC_SL_CREDIT_MULT = 2.0  # stop-loss multiple of credit received for credit spreads
MC_SL_DEBIT_FRAC = 0.5   # stop-loss fraction of debit paid for debit spreads
MC_N_PATHS = 500000
MC_N_STEPS = 100
MC_DAYTRADE_THRESHOLD_DAYS = 1.0
MIN_HOLD_DAYS = 1.0     # skip spreads whose expected hold is under 1 day
API_PAUSE_SEC = 0.15  # small pause between IBKR requests to avoid pacing/disconnects
MIN_WIDTH = 2.0
MAX_WIDTH = 10.0

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Real-world drift & vol-premium assumptions (tweak these)
USE_REAL_WORLD_DRIFT = True
MU_DRIFT = 0.10          # 10% annual drift for bullish underlyings, example

USE_VOL_RISK_PREMIUM = False
REALIZED_VOL_FRACTION = 0.7   # realized_vol = 0.7 * implied_vol (if you turn this on)
MU_SIGMA_YEARS = 4.0          # lookback for drift/vol calibration
REALIZED_VOL_YEARS = 2.0      # lookback for realized vs implied ratio
CALIBRATION_CACHE_FILE = BASE_DIR / "calibration_cache.json"
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

OUTPUT_DIR = BASE_DIR / "output"

CALIBRATION_SETTINGS = {
    'MU_SIGMA_YEARS': MU_SIGMA_YEARS,
    'API_PAUSE_SEC': API_PAUSE_SEC,
    'HOST': HOST,
    'PORT': PORT,
    'CLIENT_ID': CLIENT_ID,
    'MU_DRIFT': MU_DRIFT,
    'REALIZED_VOL_FRACTION': REALIZED_VOL_FRACTION,
    'CALIBRATION_MAX_AGE_DAYS': CALIBRATION_MAX_AGE_DAYS,
}
STRATEGY_CONFIG = {
    'MIN_POP': MIN_POP,
    'REQUIRE_POSITIVE_EV': REQUIRE_POSITIVE_EV,
    'MIN_EV': MIN_EV,
    'MIN_EV_PER_RISK': MIN_EV_PER_RISK,
    'MIN_HOLD_DAYS': MIN_HOLD_DAYS,
    'PORTFOLIO_RISK_BUDGET_FRAC': PORTFOLIO_RISK_BUDGET_FRAC,
    'TARGET_MONTHLY_RETURN': TARGET_MONTHLY_RETURN,
    'MC_TP_FRAC': MC_TP_FRAC,
    'MC_SL_DEBIT_FRAC': MC_SL_DEBIT_FRAC,
    'MC_SL_CREDIT_MULT': MC_SL_CREDIT_MULT,
    'MC_N_PATHS': MC_N_PATHS,
    'MC_N_STEPS': MC_N_STEPS,
    'MC_DAYTRADE_THRESHOLD_DAYS': MC_DAYTRADE_THRESHOLD_DAYS,
}

set_filter_verbose(FILTER_VERBOSE)


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

    calibration_cache = load_calibration_cache(CALIBRATION_CACHE_FILE)
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
        snap = await fetch_symbol_snapshot(ib, sym, API_PAUSE_SEC, HOST, PORT, CLIENT_ID)
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
        if not await ensure_connected(ib, HOST, PORT, CLIENT_ID):
            break
        try:
            await ib.qualifyContractsAsync(contract)
            await asyncio.sleep(API_PAUSE_SEC)
        except Exception as e:
            print(f"  Qualification failed: {e}")
            continue

        # Get current underlying market data (for price reference and risk-free rate via generic tick 106)
        ticker = ib.reqMktData(contract, '106', False, False)

        # Wait briefly for price data with fallbacks to close price and risk-free rate
        underlying_price = None
        risk_free_rate = None
        for _ in range(50):
            if ticker.last is not None and not np.isnan(ticker.last):
                underlying_price = ticker.last
            elif ticker.close is not None and not np.isnan(ticker.close):
                underlying_price = ticker.close

            if ticker.modelGreeks and ticker.modelGreeks.riskFreeRate is not None:
                risk_free_rate = ticker.modelGreeks.riskFreeRate

            if underlying_price is not None and risk_free_rate is not None:
                break

            await asyncio.sleep(0.1)

        if underlying_price is None or underlying_price <= 0:
            print("  Skipping: unable to get valid underlying price.")
            continue

        if risk_free_rate is None or not math.isfinite(risk_free_rate):
            risk_free_rate = RISK_FREE_RATE_FALLBACK

        print(f"  Underlying Price: {underlying_price}")
        print(f"  Risk-free rate (generic 106): {risk_free_rate:.4f}")

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
            calib, updated = await calibrate_symbol(ib, contract, None, calibration_cache, CALIBRATION_SETTINGS)
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
        if not await quick_option_liquidity_check(
            ib,
            contract,
            target_exp_liq,
            strikes_for_liq,
            MIN_OPT_VOL,
            API_PAUSE_SEC,
            HOST,
            PORT,
            CLIENT_ID
        ):
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
                if not await ensure_connected(ib, HOST, PORT, CLIENT_ID):
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
            # Recalibrate if missing or lacking realized_vol_frac once IV hint is available
            if calib is None or calib.get('realized_vol_frac') is None:
                calib, updated = await calibrate_symbol(ib, contract, iv_median, calibration_cache, CALIBRATION_SETTINGS)
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
            strategy_config = {
                **STRATEGY_CONFIG,
                'MIN_ENTRY': MIN_ENTRY,
                'MAX_DEBIT': MAX_DEBIT,
                'MIN_WIDTH': MIN_WIDTH,
                'MAX_WIDTH': MAX_WIDTH,
                'MIN_OPT_VOL': MIN_OPT_VOL,
                'TIGHT_MAX_ABS': TIGHT_MAX_ABS,
                'TIGHT_MAX_REL': TIGHT_MAX_REL,
            }

            bull_call_total = count_upper_pairs(df_calls, MIN_OPT_VOL) if not df_calls.empty else 0
            bear_call_total = count_upper_pairs(df_calls, MIN_OPT_VOL) if not df_calls.empty else 0
            bull_put_total = count_lower_pairs(df_puts, MIN_OPT_VOL) if not df_puts.empty else 0
            bear_put_total = count_lower_pairs(df_puts, MIN_OPT_VOL) if not df_puts.empty else 0

            bull_call_bar = tqdm(total=bull_call_total, desc=f"{contract.symbol} {target_exp} bull calls", leave=False) if bull_call_total else None
            bear_call_bar = tqdm(total=bear_call_total, desc=f"{contract.symbol} {target_exp} bear calls", leave=False) if bear_call_total else None
            bull_put_bar = tqdm(total=bull_put_total, desc=f"{contract.symbol} {target_exp} bull puts", leave=False) if bull_put_total else None
            bear_put_bar = tqdm(total=bear_put_total, desc=f"{contract.symbol} {target_exp} bear puts", leave=False) if bear_put_total else None

            bars = {
                'bull_call': bull_call_bar,
                'bear_call': bear_call_bar,
                'bull_put': bull_put_bar,
                'bear_put': bear_put_bar,
            }

            def select_bull_call(df, bars_map):
                bar = bars_map.get('bull_call')
                dfv = df[df['volume'] >= MIN_OPT_VOL]
                for _, long_leg in dfv.iterrows():
                    short_legs = dfv[dfv['strike'] > long_leg['strike']]
                    for _, short_leg in short_legs.iterrows():
                        spread_desc = f"Bull Call {long_leg['strike']}->{short_leg['strike']}"
                        yield long_leg, short_leg, spread_desc, bar

            def select_bear_call(df, bars_map):
                bar = bars_map.get('bear_call')
                dfv = df[df['volume'] >= MIN_OPT_VOL]
                for _, short_leg in dfv.iterrows():
                    long_legs = dfv[dfv['strike'] > short_leg['strike']]
                    for _, long_leg in long_legs.iterrows():
                        spread_desc = f"Bear Call {short_leg['strike']}->{long_leg['strike']}"
                        yield long_leg, short_leg, spread_desc, bar

            def select_bull_put(df, bars_map):
                bar = bars_map.get('bull_put')
                dfv = df[df['volume'] >= MIN_OPT_VOL]
                for _, short_leg in dfv.iterrows():
                    long_legs = dfv[dfv['strike'] < short_leg['strike']]
                    for _, long_leg in long_legs.iterrows():
                        spread_desc = f"Bull Put {short_leg['strike']}->{long_leg['strike']}"
                        yield long_leg, short_leg, spread_desc, bar

            def select_bear_put(df, bars_map):
                bar = bars_map.get('bear_put')
                dfv = df[df['volume'] >= MIN_OPT_VOL]
                for _, long_leg in dfv.iterrows():
                    short_legs = dfv[dfv['strike'] < long_leg['strike']]
                    for _, short_leg in short_legs.iterrows():
                        spread_desc = f"Bear Put {long_leg['strike']}->{short_leg['strike']}"
                        yield long_leg, short_leg, spread_desc, bar

            strategies = [
                VerticalStrategy(
                    name="Bull Call",
                    right='C',
                    entry_type='Debit',
                    skew_threshold=MIN_SKEW_CAPTURE_BULL,
                    trend_gate='down',
                    select_legs=select_bull_call,
                    width_fn=lambda long_leg, short_leg: short_leg['strike'] - long_leg['strike'],
                    price_fn=lambda long_leg, short_leg: long_leg['ask'] - short_leg['bid'],
                    max_profit_fn=lambda entry, width: width - entry,
                    max_loss_fn=lambda entry, width, max_profit: entry,
                    be_fn=lambda long_leg, short_leg, entry: long_leg['strike'] + entry,
                    skew_fn=lambda long_leg, short_leg: short_leg['iv'] - long_leg['iv'],
                    label_fn=lambda long_leg, short_leg: f"{long_leg['strike']}/{short_leg['strike']} Bull Call"
                ),
                VerticalStrategy(
                    name="Bear Call",
                    right='C',
                    entry_type='Credit',
                    skew_threshold=MIN_SKEW_CAPTURE_BEAR,
                    trend_gate='up',
                    select_legs=select_bear_call,
                    width_fn=lambda long_leg, short_leg: long_leg['strike'] - short_leg['strike'],
                    price_fn=lambda long_leg, short_leg: short_leg['bid'] - long_leg['ask'],
                    max_profit_fn=lambda entry, width: entry,
                    max_loss_fn=lambda entry, width, max_profit: width - entry,
                    be_fn=lambda long_leg, short_leg, entry: short_leg['strike'] + entry,
                    skew_fn=lambda long_leg, short_leg: short_leg['iv'] - long_leg['iv'],
                    label_fn=lambda long_leg, short_leg: f"{short_leg['strike']}/{long_leg['strike']} Bear Call"
                ),
                VerticalStrategy(
                    name="Bull Put",
                    right='P',
                    entry_type='Credit',
                    skew_threshold=MIN_SKEW_CAPTURE_BULL,
                    trend_gate='down',
                    select_legs=select_bull_put,
                    width_fn=lambda long_leg, short_leg: short_leg['strike'] - long_leg['strike'],
                    price_fn=lambda long_leg, short_leg: short_leg['bid'] - long_leg['ask'],
                    max_profit_fn=lambda entry, width: entry,
                    max_loss_fn=lambda entry, width, max_profit: width - entry,
                    be_fn=lambda long_leg, short_leg, entry: short_leg['strike'] - entry,
                    skew_fn=lambda long_leg, short_leg: short_leg['iv'] - long_leg['iv'],
                    label_fn=lambda long_leg, short_leg: f"{long_leg['strike']}/{short_leg['strike']} Bull Put"
                ),
                VerticalStrategy(
                    name="Bear Put",
                    right='P',
                    entry_type='Debit',
                    skew_threshold=MIN_SKEW_CAPTURE_BEAR,
                    trend_gate='up',
                    select_legs=select_bear_put,
                    width_fn=lambda long_leg, short_leg: long_leg['strike'] - short_leg['strike'],
                    price_fn=lambda long_leg, short_leg: long_leg['ask'] - short_leg['bid'],
                    max_profit_fn=lambda entry, width: width - entry,
                    max_loss_fn=lambda entry, width, max_profit: entry,
                    be_fn=lambda long_leg, short_leg, entry: long_leg['strike'] - entry,
                    skew_fn=lambda long_leg, short_leg: short_leg['iv'] - long_leg['iv'],
                    label_fn=lambda long_leg, short_leg: f"{short_leg['strike']}/{long_leg['strike']} Bear Put"
                ),
            ]

            for strat in strategies:
                df_src = df_calls if strat.right == 'C' else df_puts
                if df_src.empty:
                    continue
                run_vertical_strategy(
                    df_src,
                    strat,
                    trend,
                    sigma_fallback,
                    target_exp,
                    T,
                    mu_param,
                    rv_frac_param,
                    risk_free_rate,
                    underlying_price,
                    opportunities,
                    mc_stats,
                    strategy_config,
                    bars
                )

            if bull_call_bar: bull_call_bar.close()
            if bear_call_bar: bear_call_bar.close()
            if bull_put_bar: bull_put_bar.close()
            if bear_put_bar: bear_put_bar.close()

        symbols_bar.update(1)

    if cache_dirty:
        save_calibration_cache(CALIBRATION_CACHE_FILE, calibration_cache)

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

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUT_DIR / f"ibkr_spreads_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_res.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
    else:
        print("\nNo spreads found matching criteria.")
        if mc_stats:
            print(f"  MC runs: {mc_stats.get('mc_runs', 0)} | accepted: {mc_stats.get('accepted', 0)} | PoP fails: {mc_stats.get('pop_fail', 0)} | EV fails: {mc_stats.get('ev_fail', 0)} | ROI fails: {mc_stats.get('roi_fail', 0)} | hold-time fails: {mc_stats.get('hold_fail', 0)}")

if __name__ == '__main__':
    asyncio.run(main())
