from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List

import numpy as np
import pandas as pd

from filters import log_filter
from strategies import evaluate_vertical_candidate
from utils import bar_update, leg_is_tight


@dataclass
class VerticalStrategy:
    name: str
    right: str
    entry_type: str  # 'Debit' or 'Credit'
    skew_threshold: float
    trend_gate: Optional[str]  # 'up', 'down', or None
    select_legs: Callable[[pd.DataFrame, pd.Series], list]
    width_fn: Callable[[pd.Series, pd.Series], float]
    price_fn: Callable[[pd.Series, pd.Series], float]
    max_profit_fn: Callable[[float, float], float]
    max_loss_fn: Callable[[float, float, float], float]
    be_fn: Callable[[pd.Series, pd.Series, float], float]
    skew_fn: Callable[[pd.Series, pd.Series], float]
    label_fn: Callable[[pd.Series, pd.Series], str]


def run_vertical_strategy(
    df: pd.DataFrame,
    strategy: VerticalStrategy,
    trend: str,
    sigma_fallback: Optional[float],
    target_exp: str,
    T: float,
    mu_param: Optional[float],
    rv_frac_param: Optional[float],
    risk_free_rate: float,
    underlying_price: float,
    opportunities: List[Dict[str, Any]],
    mc_stats: Dict[str, int],
    config: Dict[str, Any],
    bars: Dict[str, Any],
):
    if strategy.trend_gate and trend == strategy.trend_gate:
        log_filter(f"Trend {trend}; skipping {strategy.name} candidates", None, target_exp, strategy.name)
        return

    for long_leg, short_leg, spread_desc, bar in strategy.select_legs(df, bars):
        bar_update(bar)
        symbol = long_leg['contract'].symbol if 'contract' in long_leg else None

        # Tightness
        tight_long, spread_long, mid_long = leg_is_tight(long_leg, config['TIGHT_MAX_ABS'], config['TIGHT_MAX_REL'])
        tight_short, spread_short, mid_short = leg_is_tight(short_leg, config['TIGHT_MAX_ABS'], config['TIGHT_MAX_REL'])
        if not (tight_long and tight_short):
            log_filter(
                f"Bid/ask too wide: long {spread_long:.2f} (mid {mid_long:.2f}), short {spread_short:.2f} (mid {mid_short:.2f})",
                symbol, target_exp, spread_desc
            )
            continue

        width = strategy.width_fn(long_leg, short_leg)
        if width < config['MIN_WIDTH'] or width > config['MAX_WIDTH']:
            log_filter(f"Width {width:.2f} outside {config['MIN_WIDTH']}-{config['MAX_WIDTH']}", symbol, target_exp, spread_desc)
            continue
        if width <= 0:
            log_filter("Width non-positive", symbol, target_exp, spread_desc)
            continue

        entry = strategy.price_fn(long_leg, short_leg)
        if strategy.entry_type == 'Credit':
            if entry <= config['MIN_ENTRY']:
                log_filter(f"Credit {entry:.2f} <= MIN_ENTRY {config['MIN_ENTRY']}", symbol, target_exp, spread_desc)
                continue
            if entry >= width:
                log_filter(f"Credit {entry:.2f} >= width {width:.2f}", symbol, target_exp, spread_desc)
                continue
        else:
            if not (config['MIN_ENTRY'] < entry < config['MAX_DEBIT']):
                log_filter(f"Cost {entry:.2f} outside ({config['MIN_ENTRY']}, {config['MAX_DEBIT']})", symbol, target_exp, spread_desc)
                continue

        max_profit = strategy.max_profit_fn(entry, width)
        max_loss = strategy.max_loss_fn(entry, width, max_profit)
        if max_profit <= 0 or max_loss <= 0:
            log_filter(f"Non-positive max profit/loss (profit {max_profit:.2f}, loss {max_loss:.2f})", symbol, target_exp, spread_desc)
            continue

        avg_iv = (long_leg['iv'] + short_leg['iv']) / 2
        skew_capture = strategy.skew_fn(long_leg, short_leg)
        sigma_input = avg_iv if avg_iv and avg_iv > 0 else sigma_fallback
        if sigma_input is None or sigma_input <= 0:
            log_filter("No usable sigma for MC", symbol, target_exp, spread_desc)
            continue
        if skew_capture < strategy.skew_threshold:
            log_filter(f"Skew capture {skew_capture:.3f} < {strategy.skew_threshold}", symbol, target_exp, spread_desc)
            continue

        long_strike = min(long_leg['strike'], short_leg['strike']) if strategy.right == 'C' else max(long_leg['strike'], short_leg['strike'])
        short_strike = max(long_leg['strike'], short_leg['strike']) if strategy.right == 'C' else min(long_leg['strike'], short_leg['strike'])

        be_price = strategy.be_fn(long_leg, short_leg, entry)
        strategy_label = strategy.label_fn(long_leg, short_leg)

        evaluate_vertical_candidate(
            symbol=symbol,
            target_exp=target_exp,
            spread_desc=spread_desc,
            entry=entry,
            entry_type=strategy.entry_type,
            max_profit=max_profit,
            max_loss=max_loss,
            long_strike=long_strike,
            short_strike=short_strike,
            right=strategy.right,
            sigma_input=sigma_input,
            T=T,
            mu_param=mu_param,
            rv_frac_param=rv_frac_param,
            risk_free_rate=risk_free_rate,
            avg_iv=avg_iv,
            underlying_price=underlying_price,
            be_price=be_price,
            strategy_label=strategy_label,
            extra_fields=None,
            config=config,
            mc_stats=mc_stats,
            opportunities=opportunities
        )
