import json
import numpy as np
from scipy.stats import norm

# =====================================================
# 1. LOAD CALIBRATED PARAMETERS (FROM REAL NORDPOOL DATA)
# =====================================================

CALIBRATED_PARAMS_FILE = "calibrated_nordpool_sys_params.json"


def load_calibrated_params():
    """
    Load calibrated parameters from JSON file created by
    Calibrate_NordPool_SYS.py.

    Expected keys:
      - kappa   (mean reversion speed)
      - theta   (long run level)
      - sigma   (diffusion std of residual, per day, in price units)
      - lambda  (jump intensity per day)
      - mu_J    (mean jump size)
      - sig_J   (std dev of jump size)
    """
    try:
        with open(CALIBRATED_PARAMS_FILE, "r") as f:
            p = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"[FATAL] Could not find {CALIBRATED_PARAMS_FILE}. "
            "Run Calibrate_NordPool_SYS.py first to generate it."
        )

    kappa = float(p["kappa"])
    theta = float(p["theta"])
    sigma_daily = float(p["sigma"])     # diffusion std (per day) in price units
    lam_daily = float(p["lambda"])      # jump intensity per day
    mu_J = float(p["mu_J"])
    sig_J = float(p["sig_J"])

    print("=== Loaded Calibrated NordPool SYS Parameters ===")
    print(f"kappa   = {kappa:.4f}")
    print(f"theta   = {theta:.4f}")
    print(f"sigma_d = {sigma_daily:.4f}  (diffusion std per day)")
    print(f"lambda_d= {lam_daily:.6f}  (jump intensity per day)")
    print(f"mu_J    = {mu_J:.4f}")
    print(f"sig_J   = {sig_J:.4f}")
    print("===============================================")

    return kappa, theta, sigma_daily, lam_daily, mu_J, sig_J


# =====================================================
# 2. MONTE CARLO SIMULATOR (MEAN-REVERTING JUMP–DIFFUSION)
# =====================================================

def simulate_electricity_paths(
    S0: float,
    kappa: float,
    theta: float,
    sigma_daily: float,
    lam_daily: float,
    mu_J: float,
    sig_J: float,
    T: float = 1.0,
    N: int = 365,
    M: int = 20000,
    random_seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate M paths of electricity spot prices under a
    mean-reverting jump–diffusion:

        dS_t = kappa (theta - S_t) dt + sigma dW_t + J_t

    Here:
      - sigma_daily is the *daily* diffusion std in price units
      - lam_daily is the jump intensity per day
      - J_t is a jump size drawn from N(mu_J, sig_J^2) when a jump occurs

    We treat T in years, N steps over [0, T].
    dt = T / N (in years).
    To keep daily interpretation consistent, we use:
       sigma = sigma_daily * sqrt(252)   (approx annualized)
       lam_annual = lam_daily * 252
    so that over dt, jump prob ~ lam_annual * dt ≈ lam_daily * (dt * 252).
    If N = 252*T, that reduces correctly to lam_daily per step.
    For N=365 and using 252 as "trading days", this is an approximation.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / N  # in years

    # Approximate annualized diffusion vol from daily sigma
    sigma_annual = sigma_daily * np.sqrt(252.0)
    # Annualized jump intensity
    lam_annual = lam_daily * 252.0

    S = np.zeros((M, N + 1), dtype=float)
    S[:, 0] = S0

    for t in range(1, N + 1):
        # Brownian shock
        Z = np.random.normal(0.0, 1.0, size=M)
        dW = sigma_annual * np.sqrt(dt) * Z

        # mean reversion term
        mr = kappa * (theta - S[:, t - 1]) * dt

        # jump indicator: Poisson approx P(jump in dt) ~ lam_annual * dt
        jump_indicator = np.random.rand(M) < lam_annual * dt
        J = np.zeros(M)
        if jump_indicator.any():
            J[jump_indicator] = np.random.normal(mu_J, sig_J, size=jump_indicator.sum())

        # Euler step
        S[:, t] = S[:, t - 1] + mr + dW + J

    return S


# =====================================================
# 3. OPTION PRICING HELPERS
# =====================================================

def price_call_mc(S_paths: np.ndarray, K: float, r: float, T: float) -> float:
    """
    Price a European call by Monte Carlo given simulated paths of S_t.

    S_paths: (M, N+1) array of spot paths
    K: strike
    r: risk-free rate
    T: maturity in years
    """
    S_T = S_paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


def bs_call(S0: float, K: float, r: float, sigma_annual: float, T: float) -> float:
    """
    Black–Scholes European call price (for comparison only).
    sigma_annual is annualized percentage volatility (e.g. 0.30 for 30%).
    """
    if sigma_annual <= 0 or T <= 0:
        return max(S0 - K, 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma_annual**2) * T) / (sigma_annual * np.sqrt(T))
    d2 = d1 - sigma_annual * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


# =====================================================
# 4. MAIN EXPERIMENTS
# =====================================================

def run_single_experiment():
    """
    Use calibrated parameters to:
      - simulate paths
      - price a 1Y ATM call under the electricity model
      - compare with a Black–Scholes price using a naive annualized vol
    """
    # Load calibrated parameters
    kappa, theta, sigma_daily, lam_daily, mu_J, sig_J = load_calibrated_params()

    # --- Option / simulation config (you can tweak for your project) ---
    S0    = theta      # start around the long-run mean
    K     = theta      # ATM call
    r     = 0.02       # 2% risk-free rate (just a simple assumption)
    T     = 1.0        # 1 year
    N     = 252        # trading days
    M     = 50000      # MC paths

    print("\n=== Mean-Reverting Jump–Diffusion Electricity Model (Calibrated) ===")
    print(f"S0={S0:.2f}, K={K:.2f}, r={r:.2%}, T={T:.2f} years")
    print(f"MC paths M={M}, steps N={N}")
    print(f"kappa={kappa:.4f}, theta={theta:.4f}, sigma_daily={sigma_daily:.4f}, "
          f"lambda_daily={lam_daily:.6f}, mu_J={mu_J:.4f}, sig_J={sig_J:.4f}")

    # --- Simulate paths under your electricity model ---
    S_paths = simulate_electricity_paths(
        S0, kappa, theta, sigma_daily, lam_daily, mu_J, sig_J,
        T=T, N=N, M=M
    )

    # --- Monte Carlo price under your model ---
    price_jump_model = price_call_mc(S_paths, K, r, T)

    # --- Black–Scholes comparison ---
    # Naive annualized vol: daily std / level * sqrt(252)
    sigma_annual_pct = (sigma_daily / S0) * np.sqrt(252.0)
    price_bs = bs_call(S0, K, r, sigma_annual_pct, T)

    print("\n=== Option Pricing Results (Calibrated) ===")
    print(f"Your electricity model price (MC): {price_jump_model:.4f}")
    print(f"Black–Scholes price (approx vol): {price_bs:.4f}")
    print(f"Difference (Model - BS):          {price_jump_model - price_bs:.4f}")


def run_lambda_sensitivity():
    """
    Show how the call price changes as we scale the calibrated jump intensity λ.
    This keeps the diffusion parameters fixed and multiplies λ_daily by a factor.
    """
    kappa, theta, sigma_daily, lam_daily, mu_J, sig_J = load_calibrated_params()

    S0    = theta
    K     = theta
    r     = 0.02
    T     = 1.0
    N     = 252
    M     = 30000

    sigma_annual_pct = (sigma_daily / S0) * np.sqrt(252.0)
    bs_price = bs_call(S0, K, r, sigma_annual_pct, T)

    print("\n=== Sensitivity to Jump Intensity λ (Calibrated Base) ===")
    print(f"Baseline daily λ (from data): {lam_daily:.6f}")
    print(f"Black–Scholes price (no jumps): {bs_price:.4f}\n")
    print("scale_factor\tlambda_daily\tModelPrice")

    # Try scaled versions of λ: from 0x to 3x the calibrated value
    scale_factors = [0.0, 0.5, 1.0, 2.0, 3.0]

    for sf in scale_factors:
        lam_scaled = lam_daily * sf
        S_paths = simulate_electricity_paths(
            S0, kappa, theta, sigma_daily, lam_scaled, mu_J, sig_J,
            T=T, N=N, M=M, random_seed=123
        )
        price_model = price_call_mc(S_paths, K, r, T)
        print(f"{sf:.1f}\t\t{lam_scaled:.6f}\t{price_model:.4f}")


# =====================================================
# 5. ENTRY POINT
# =====================================================

if __name__ == "__main__":
    run_single_experiment()
    run_lambda_sensitivity()
