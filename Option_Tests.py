import json
import numpy as np
import pandas as pd
from math import log, sqrt, exp, erf

# =====================================================
# CONFIG
# =====================================================

SPOT_CSV = "NP.csv"                         # hourly NordPool SYS data
PARAMS_JSON = "calibrated_nordpool_sys_params.json"

RISK_FREE_RATE = 0.02                       # 2% per year
TRADING_DAYS = 252                          # days per year

# Monte Carlo settings
MC_PATHS = 20000


# =====================================================
# BASIC HELPERS
# =====================================================

def norm_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_call(S0: float, K: float, T: float, sigma: float, r: float) -> float:
    """
    Black–Scholes European call on a non-dividend underlying.
    """
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)
    vol_sqrt_T = sigma * sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


# =====================================================
# 1. LOAD SPOT & PARAMS
# =====================================================

def load_spot_series(csv_path: str = SPOT_CSV) -> pd.Series:
    """
    Load NP.csv (hourly NordPool data), strip column spaces,
    aggregate to daily average prices, return daily series.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns or "Price" not in df.columns:
        raise RuntimeError(f"Expected columns 'Date' and 'Price' in {csv_path}, "
                           f"got: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"]).sort_values("Date").set_index("Date")

    # Hourly -> daily
    daily = df["Price"].resample("D").mean().dropna()

    print(f"[LOAD] Daily SYS prices: {len(daily)} points "
          f"from {daily.index[0].date()} to {daily.index[-1].date()}")
    return daily


def load_calibrated_params(path: str = PARAMS_JSON):
    """Load calibrated OU + jump parameters from JSON."""
    with open(path, "r") as f:
        params = json.load(f)

    kappa = float(params["kappa"])
    theta = float(params["theta"])
    sigma = float(params.get("sigma_d", params.get("sigma")))          # daily diffusion sigma
    lam   = float(params.get("lambda_d", params.get("lambda")))        # daily jump intensity
    mu_J  = float(params["mu_J"])
    sig_J = float(params["sig_J"])

    print("\n[LOAD] Calibrated parameters:")
    print(f"  kappa   = {kappa:.4f}")
    print(f"  theta   = {theta:.4f}")
    print(f"  sigma_d = {sigma:.4f}  (diffusion std per day)")
    print(f"  lambda_d= {lam:.6f}  (jump intensity per day)")
    print(f"  mu_J    = {mu_J:.4f}")
    print(f"  sig_J   = {sig_J:.4f}")

    return kappa, theta, sigma, lam, mu_J, sig_J


# =====================================================
# 2. MONTE CARLO SIMULATION UNDER CALIBRATED MODEL
# =====================================================

def simulate_paths(
    S0: float,
    T_years: float,
    kappa: float,
    theta: float,
    sigma_daily: float,
    lam_daily: float,
    mu_J: float,
    sig_J: float,
    n_paths: int = MC_PATHS
):
    """
    Simulate mean-reverting jump–diffusion on *daily* steps:

      S_{t+1} = S_t + kappa (theta - S_t) dt
                + sigma * sqrt(dt) * Z
                + Jumps

    Here:
      - dt = 1 / 252 years (one trading day)
      - lam_daily is *jumps per day* (from calibration),
        so we use Poisson(lam_daily) each step (not lam_daily * dt).
    """
    n_steps = max(1, int(round(T_years * TRADING_DAYS)))
    dt = 1.0 / TRADING_DAYS
    sqrt_dt = np.sqrt(dt)

    S = np.full((n_paths,), S0, dtype=float)

    for _ in range(n_steps):
        # diffusion term
        dW = np.random.normal(0.0, 1.0, size=n_paths)
        diffusion = sigma_daily * sqrt_dt * dW

        # mean-reversion drift
        drift = kappa * (theta - S) * dt

        # jumps: lam_daily is jumps per day
        N_jumps = np.random.poisson(lam_daily, size=n_paths)
        jump_sizes = np.where(
            N_jumps > 0,
            np.random.normal(mu_J, sig_J, size=n_paths) * N_jumps,
            0.0
        )

        S = S + drift + diffusion + jump_sizes

        # prevent negative or zero prices for stability
        S = np.maximum(S, 0.01)

    return S


def price_call_mc_jump_diffusion(
    S0: float,
    K: float,
    T_years: float,
    kappa: float,
    theta: float,
    sigma_daily: float,
    lam_daily: float,
    mu_J: float,
    sig_J: float,
    r: float = RISK_FREE_RATE,
    n_paths: int = MC_PATHS
) -> float:
    """Monte Carlo European call price under calibrated model."""
    S_T = simulate_paths(
        S0, T_years, kappa, theta, sigma_daily, lam_daily, mu_J, sig_J, n_paths
    )
    payoff = np.maximum(S_T - K, 0.0)
    return np.mean(payoff) * exp(-r * T_years)


# =====================================================
# 3. METHOD 1: DISTRIBUTION / FORWARD-STYLE CHECK
# =====================================================

def method1_distribution_check(S: pd.Series,
                               kappa: float,
                               theta: float,
                               sigma_daily: float,
                               lam_daily: float,
                               mu_J: float,
                               sig_J: float):
    """
    Compare empirical vs model mean & std of price changes
    at several horizons (like a forward-style sanity check).
    """
    print("\n========== METHOD 1: DISTRIBUTION CHECK ==========")

    horizons_days = [7, 30, 90, 180, 365]   # 1w, 1m, 3m, 6m, 1y
    S_values = S.values

    for H in horizons_days:
        if H >= len(S_values) // 2:
            continue

        # empirical ΔS = S_{t+H} - S_t
        S_t = S_values[:-H]
        S_T = S_values[H:]
        dS_emp = S_T - S_t
        mean_emp = dS_emp.mean()
        std_emp = dS_emp.std(ddof=1)

        # model: simulate from last spot
        S0 = S_values[-1]
        T_years = H / TRADING_DAYS
        S_T_model = simulate_paths(
            S0, T_years,
            kappa, theta, sigma_daily,
            lam_daily, mu_J, sig_J,
            n_paths=MC_PATHS
        )
        dS_model = S_T_model - S0
        mean_mod = dS_model.mean()
        std_mod = dS_model.std(ddof=1)

        print(f"\n-- Horizon H = {H} days (~{T_years:.2f} years) --")
        print(f"Empirical ΔS: mean = {mean_emp:8.4f}, std = {std_emp:8.4f}")
        print(f"Model ΔS:     mean = {mean_mod:8.4f}, std = {std_mod:8.4f}")


# =====================================================
# 4. METHOD 2: VOLATILITY TERM STRUCTURE
# =====================================================

def realized_vol_term_structure(S: pd.Series, T_years: float) -> float:
    """
    Estimate realized annualized volatility for horizon T_years:
      - compute log returns over horizon T (overlapping windows)
      - vol_T = std(log(S_{t+T}/S_t))
      - annualize: sigma_annual = vol_T / sqrt(T_years)
    """
    T_days = int(round(T_years * TRADING_DAYS))
    if T_days <= 0 or T_days >= len(S) // 2:
        return np.nan

    S_vals = S.values
    S_t = S_vals[:-T_days]
    S_T = S_vals[T_days:]
    log_rets = np.log(S_T / S_t)
    vol_T = np.std(log_rets, ddof=1)
    return vol_T / sqrt(T_years)


def model_vol_term_structure(S0: float,
                             T_years: float,
                             kappa: float,
                             theta: float,
                             sigma_daily: float,
                             lam_daily: float,
                             mu_J: float,
                             sig_J: float) -> float:
    """
    Simulate horizon T_years many times, compute std of
    log(S_T / S0), annualized.
    """
    S_T = simulate_paths(
        S0, T_years,
        kappa, theta, sigma_daily,
        lam_daily, mu_J, sig_J,
        n_paths=MC_PATHS
    )

    S_T = np.maximum(S_T, 0.01)
    log_rets = np.log(S_T / S0)
    vol_T = np.std(log_rets, ddof=1)
    return vol_T / sqrt(T_years)


def method2_vol_curve(S: pd.Series,
                      kappa: float,
                      theta: float,
                      sigma_daily: float,
                      lam_daily: float,
                      mu_J: float,
                      sig_J: float):
    """
    Compare realized vs model-implied volatility term structure
    for multiple horizons.
    """
    print("\n========== METHOD 2: VOLATILITY TERM STRUCTURE ==========")

    horizons_years = [1/12, 0.25, 0.5, 1.0]   # 1m, 3m, 6m, 1y
    S0 = S.iloc[-1]

    print("\nT_years   Realized_AnnVol   Model_AnnVol")
    for T in horizons_years:
        sigma_real = realized_vol_term_structure(S, T)
        sigma_model = model_vol_term_structure(
            S0, T, kappa, theta, sigma_daily, lam_daily, mu_J, sig_J
        )
        print(f"{T:6.2f}   {sigma_real:15.4f}   {sigma_model:13.4f}")


# =====================================================
# 5. METHOD 3: SYNTHETIC ATM OPTION COMPARISON
# =====================================================

def method3_synthetic_options(S: pd.Series,
                              kappa: float,
                              theta: float,
                              sigma_daily: float,
                              lam_daily: float,
                              mu_J: float,
                              sig_J: float,
                              r: float = RISK_FREE_RATE):
    """
    Build synthetic ATM option "market" using:
      - S0 = last spot
      - T in {1m,3m,6m,1y}
      - Realized vol as 'market vol'
      - BS price using that vol
    Then compare to:
      - jump–diffusion Monte Carlo price using calibrated params.
    """
    print("\n========== METHOD 3: SYNTHETIC ATM OPTIONS ==========")

    S0 = S.iloc[-1]
    K = S0

    horizons_years = [1/12, 0.25, 0.5, 1.0]

    print("\nT_years   Vol_Real(Ann)   BS_Price   JD_Model_Price   (JD - BS)")
    for T in horizons_years:
        sigma_real = realized_vol_term_structure(S, T)
        if np.isnan(sigma_real) or sigma_real <= 0:
            continue

        bs_price = black_scholes_call(S0, K, T, sigma_real, r)
        jd_price = price_call_mc_jump_diffusion(
            S0, K, T, kappa, theta, sigma_daily, lam_daily, mu_J, sig_J, r,
            n_paths=MC_PATHS
        )
        diff = jd_price - bs_price

        print(f"{T:6.2f}   {sigma_real:12.4f}   {bs_price:8.4f}   "
              f"{jd_price:14.4f}   {diff:9.4f}")


# =====================================================
# MAIN
# =====================================================

def main():
    # 1) Load data & parameters
    S = load_spot_series(SPOT_CSV)
    kappa, theta, sigma_d, lam_d, mu_J, sig_J = load_calibrated_params(PARAMS_JSON)

    # 2) Method 1: distribution / forward check
    method1_distribution_check(S, kappa, theta, sigma_d, lam_d, mu_J, sig_J)

    # 3) Method 2: volatility term structure
    method2_vol_curve(S, kappa, theta, sigma_d, lam_d, mu_J, sig_J)

    # 4) Method 3: synthetic ATM options (BS vs jump–diffusion)
    method3_synthetic_options(S, kappa, theta, sigma_d, lam_d, mu_J, sig_J, RISK_FREE_RATE)


if __name__ == "__main__":
    main()

