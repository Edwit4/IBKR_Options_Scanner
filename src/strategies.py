from typing import Dict, Any, Optional

from filters import log_filter, log_pass
from mc import simulate_vertical_pop_ev_torch


def evaluate_vertical_candidate(
    *,
    symbol: str,
    target_exp: str,
    spread_desc: str,
    entry: float,
    entry_type: str,
    max_profit: float,
    max_loss: float,
    long_strike: float,
    short_strike: float,
    right: str,
    sigma_input: float,
    T: float,
    mu_param: Optional[float],
    rv_frac_param: Optional[float],
    risk_free_rate: float,
    avg_iv: Optional[float],
    underlying_price: float,
    be_price: float,
    strategy_label: str,
    extra_fields: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    mc_stats: Dict[str, int],
    opportunities: list
) -> bool:
    """
    Run MC, apply filters, and append an opportunity row if accepted.
    Returns True if appended, False otherwise.
    """
    log_pass("Running MC", symbol, target_exp, spread_desc)
    mc_stats['mc_runs'] += 1

    sl_frac = config['MC_SL_DEBIT_FRAC'] if entry_type == 'Debit' else config['MC_SL_CREDIT_MULT']

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
        sl_frac=sl_frac,
        n_paths=config['MC_N_PATHS'],
        n_steps=config['MC_N_STEPS'],
        daytrade_threshold_days=config['MC_DAYTRADE_THRESHOLD_DAYS'],
        device=None,
        mu=mu_param,
        realized_vol_frac=rv_frac_param
    )

    if pop_mc <= config['MIN_POP']:
        mc_stats['pop_fail'] += 1
        log_filter(f"PoP {pop_mc:.3f} <= {config['MIN_POP']}", symbol, target_exp, spread_desc)
        return False

    ev = ev_mc
    if config['REQUIRE_POSITIVE_EV'] and ev <= 0:
        mc_stats['ev_fail'] += 1
        log_filter(f"EV {ev:.3f} <= 0 with positive EV required", symbol, target_exp, spread_desc)
        return False

    hold_days = avg_ht * 365.0
    if hold_days < config['MIN_HOLD_DAYS']:
        mc_stats['hold_fail'] += 1
        log_filter(f"Avg hold {hold_days:.2f}d < {config['MIN_HOLD_DAYS']}d", symbol, target_exp, spread_desc)
        return False

    risk = entry if entry_type == 'Debit' else max_loss

    roi_trade = ev / risk                 # expected return on risk over trade life
    ev_per_risk = roi_trade               # keep this metric
    roi_monthly = roi_trade * (30.0 / hold_days)

    # Convert to portfolio-level monthly return, given average risk budget
    portfolio_roi_monthly = roi_monthly * config['PORTFOLIO_RISK_BUDGET_FRAC']

    if ev < config['MIN_EV']:
        log_filter(f"EV {ev:.3f} < MIN_EV {config['MIN_EV']}", symbol, target_exp, spread_desc)
        return False
    if ev_per_risk < config['MIN_EV_PER_RISK']:
        log_filter(f"EV/risk {ev_per_risk:.3f} < {config['MIN_EV_PER_RISK']}", symbol, target_exp, spread_desc)
        return False
    if portfolio_roi_monthly < config['TARGET_MONTHLY_RETURN']:
        mc_stats['roi_fail'] += 1
        log_filter(
            f"Portfolio ROI/month {portfolio_roi_monthly:.3f} < target {config['TARGET_MONTHLY_RETURN']}",
            symbol, target_exp, spread_desc
        )
        return False

    opp = {
        'Symbol': symbol,
        'Expiry': target_exp,
        'Strategy': strategy_label,
        'Entry': round(entry, 2),
        'Entry Type': entry_type,
        'Max Profit': round(max_profit, 2),
        'Max Loss': round(max_loss, 2),
        'PoP': round(pop_mc, 3),
        'PoP_SE': round(pop_se, 4),
        'EV': round(ev, 2),
        'EV_SE': round(ev_se, 4),
        'EV_Per_Risk': round(ev_per_risk, 3),
        'AvgHold_days': round(hold_days, 1),
        'P_Daytrade': round(p_daytrade, 3),
        'IV': round(avg_iv, 3) if avg_iv is not None else None,
        'Underlying': round(underlying_price, 2),
        'BE': round(be_price, 2)
    }
    if extra_fields:
        opp.update(extra_fields)

    opportunities.append(opp)
    log_pass(f"MC accepted: PoP {pop_mc:.3f}, EV {ev:.2f}, ROI/mo {roi_monthly:.3f}", symbol, target_exp, spread_desc)
    mc_stats['accepted'] += 1
    return True
