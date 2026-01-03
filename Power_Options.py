import numpy as np
from scipy.stats import norm

# =========================
# 1. Monte Carlo simulator
# =========================

def simulate_electricity_paths(
    S0: float,
    kappa: float,
    theta: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sig_J: float,
    T: float = 1.0,
    N: int = 365,
    M: int = 10000,
    random_seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate M paths of an electricity spot price under a
    mean-reverting jump–diffusion:

        dS_t = kappa (theta - S_t) dt + sigma dW_t + J_t dN_t

    where N_t is Poisson with intensity lam, and J_t ~ N(mu_J, sig_J^2)
    when a jump occurs.

    Returns:
        S: array of shape (M, N+1) with simulated paths.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / N
    S = np.zeros((M, N + 1), dtype=float)
    S[:, 0] = S0

    for t in range(1, N + 1):
        Z = np.random.normal(0.0, 1.0, size=M)              # diffusion shocks
        dW = sigma * np.sqrt(dt) * Z

        # mean reversion term
        mr = kappa * (theta - S[:, t - 1]) * dt

        # jump indicator: True if a jump occurs in this dt
        jump_indicator = np.random.rand(M) < lam * dt
        # jump sizes
        J = np.zeros(M)
        J[jump_indicator] = np.random.normal(mu_J, sig_J, size=jump_indicator.sum())

        # Euler step
        S[:, t] = S[:, t - 1] + mr + dW + J

    return S


# =========================
# 2. Option pricer under the model
# =========================

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


# =========================
# 3. Black–Scholes for comparison
# =========================

def bs_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black–Scholes European call price for comparison.
    """
    if sigma <= 0 or T <= 0:
        return max(S0 - K, 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


# =========================
# 4. Test / experiment runner
# =========================

def run_single_experiment():
    # --- Model / option parameters ---
    S0    = 100.0    # starting electricity price (index level)
    K     = 100.0    # at-the-money call
    r     = 0.02     # 2% risk-free rate
    T     = 1.0      # 1 year
    N     = 365      # daily steps
    M     = 20000    # number of Monte Carlo paths

    # Mean-reverting jump–diffusion parameters
    kappa = 2.0      # speed of mean reversion
    theta = 100.0    # long-run mean price
    sigma = 20.0     # diffusion volatility (absolute, not %)
    lam   = 5.0      # expected number of jumps per year
    mu_J  = 0.0      # mean jump size
    sig_J = 40.0     # std of jump size (big spikes)

    print("=== Mean-Reverting Jump–Diffusion Electricity Model ===")
    print(f"S0={S0}, K={K}, r={r}, T={T}")
    print(f"kappa={kappa}, theta={theta}, sigma={sigma}, "
          f"lambda={lam}, mu_J={mu_J}, sig_J={sig_J}")
    print(f"MC paths M={M}, steps N={N}")

    # --- Simulate paths under the electricity model ---
    S_paths = simulate_electricity_paths(
        S0, kappa, theta, sigma, lam, mu_J, sig_J, T=T, N=N, M=M
    )

    # --- Monte Carlo price under your model ---
    price_jump_model = price_call_mc(S_paths, K, r, T)

    # --- BS price for comparison ---
    # For fairness, we can convert sigma (absolute) to "relative" vol:
    # approximate % vol as sigma / S0.
    sigma_bs = sigma / S0
    price_bs = bs_call(S0, K, r, sigma_bs, T)

    print("\n=== Option Pricing Results ===")
    print(f"Your electricity model price (MC): {price_jump_model:.4f}")
    print(f"Black–Scholes price (same base vol): {price_bs:.4f}")
    print(f"Difference (Model - BS): {price_jump_model - price_bs:.4f}")


def run_lambda_sensitivity():
    """
    Show how the option price changes as jump intensity λ increases,
    compared to the BS price which ignores jumps entirely.
    """
    S0    = 100.0
    K     = 100.0
    r     = 0.02
    T     = 1.0
    N     = 365
    M     = 20000

    kappa = 2.0
    theta = 100.0
    sigma = 20.0
    mu_J  = 0.0
    sig_J = 40.0

    sigma_bs = sigma / S0
    bs_price = bs_call(S0, K, r, sigma_bs, T)

    print("\n=== Sensitivity to Jump Intensity λ ===")
    print(f"BS price (no jumps): {bs_price:.4f}\n")
    print("lambda\tModelPrice")

    for lam in [0.0, 1.0, 3.0, 5.0, 10.0]:
        S_paths = simulate_electricity_paths(
            S0, kappa, theta, sigma, lam, mu_J, sig_J, T=T, N=N, M=M,
            random_seed=123  # fixed seed per λ for comparability
        )
        price_model = price_call_mc(S_paths, K, r, T)
        print(f"{lam:.1f}\t{price_model:.4f}")


if __name__ == "__main__":
    run_single_experiment()
    run_lambda_sensitivity()
