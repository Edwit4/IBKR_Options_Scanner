import os
import torch


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
        pop_se     : scalar float (standard error of PoP)
        ev_se      : scalar float (standard error of EV)
    """
    if T <= 0 or sigma <= 0 or S0 <= 0 or max_loss <= 0 or max_profit <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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

    # Rate used for BS pricing should remain risk-free even when using drifted paths
    r_pricing = r

    tp_level = tp_frac * max_profit
    if entry_type == 'Debit':
        sl_level = -sl_frac * max_loss
    else:
        sl_level = -sl_frac * entry

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
    pop_tensor = (pl_paths > 0).double()
    pop = pop_tensor.mean()
    ev = pl_paths.mean()
    avg_ht = ht_paths.mean()

    p_daytrade = (ht_paths <= daytrade_threshold_years).double().mean()

    # Standard errors
    n = float(n_paths)

    # PoP: binomial standard error
    pop_var = pop * (1.0 - pop)
    pop_se = torch.sqrt(pop_var / n)

    # EV: standard error of the mean using sample variance
    if n_paths > 1:
        ev_var = pl_paths.var(unbiased=True)
        ev_se = torch.sqrt(ev_var / n)
    else:
        ev_se = torch.tensor(0.0, dtype=torch.float64, device=device)

    # Return as Python floats
    return (
        float(pop.item()),
        float(ev.item()),
        float(avg_ht.item()),
        float(p_daytrade.item()),
        float(pop_se.item()),
        float(ev_se.item()),
    )
