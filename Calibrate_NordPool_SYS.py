import numpy as np
import pandas as pd
import json

# =========================
# CONFIG – MATCHES NP.csv
# =========================

# Name of your file in the project folder
CSV_PATH = "NP.csv"     # change only if you renamed the file

# Column names BEFORE stripping spaces in the CSV
DATETIME_COL_RAW = "Date"   # first column in NP.csv
PRICE_COL_RAW    = "Price"  # we’ll strip spaces to get this

# NP.csv is already just Nord Pool prices, no area column
AREA_COL = None
SYS_CODE = None

# NP.csv is hourly → we will average to daily
DATA_IS_HOURLY = True

# Jump detection threshold (in units of residual std dev)
JUMP_SIGMA_THRESHOLD = 3.0


# =========================
# 1. LOAD & CLEAN DATA
# =========================

def load_sys_prices() -> pd.Series:
    """
    Load NP.csv (hourly data), clean columns, and convert
    to a daily average price series S_t.
    """
    df = pd.read_csv(CSV_PATH)

    # Strip whitespace from column names so " Price" -> "Price"
    df.columns = df.columns.str.strip()

    # Now these should exist:
    #   - "Date"
    #   - "Price"
    if DATETIME_COL_RAW not in df.columns:
        raise RuntimeError(f"[FATAL] Column '{DATETIME_COL_RAW}' not found in CSV. "
                           f"Available columns: {list(df.columns)}")

    if PRICE_COL_RAW not in df.columns:
        raise RuntimeError(f"[FATAL] Column '{PRICE_COL_RAW}' not found in CSV. "
                           f"Available columns: {list(df.columns)}")

    # Parse datetime
    df[DATETIME_COL_RAW] = pd.to_datetime(df[DATETIME_COL_RAW])

    # Ensure price is numeric
    df[PRICE_COL_RAW] = pd.to_numeric(df[PRICE_COL_RAW], errors="coerce")
    df = df.dropna(subset=[PRICE_COL_RAW])

    # Sort & index
    df = df.sort_values(DATETIME_COL_RAW)
    df = df.set_index(DATETIME_COL_RAW)

    # Hourly → daily average
    if DATA_IS_HOURLY:
        daily = df[PRICE_COL_RAW].resample("D").mean()
    else:
        daily = df[PRICE_COL_RAW]

    daily = daily.dropna()

    print(f"[INFO] Loaded {len(daily)} daily prices from {daily.index[0].date()} "
          f"to {daily.index[-1].date()}")

    return daily


# =========================
# 2. CALIBRATE OU PARAMETERS (κ, θ, σ)
# =========================

def calibrate_ou(S: pd.Series):
    """
    Calibrate mean-reverting OU model:
        S_{t+1} = a + b S_t + ε_t
    Mapping:
        b = exp(-kappa)
        theta = a / (1 - b)
        sigma = std(ε)
    """
    S_t   = S[:-1].values
    S_tp1 = S[1:].values

    # Regression: S_{t+1} = a + b S_t
    X = np.vstack([np.ones_like(S_t), S_t]).T
    y = S_tp1

    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    a, b = beta[0], beta[1]

    # residuals
    y_hat = X @ beta
    eps   = y - y_hat

    kappa = -np.log(b)
    theta = a / (1 - b)
    sigma_eps = np.std(eps, ddof=1)

    print("\n=== OU Calibration ===")
    print(f"a = {a:.4f}, b = {b:.4f}")
    print(f"kappa = {kappa:.4f}")
    print(f"theta = {theta:.4f}")
    print(f"sigma (residual std) = {sigma_eps:.4f}")

    return kappa, theta, sigma_eps, eps


# =========================
# 3. JUMP CALIBRATION (λ, μ_J, σ_J)
# =========================

def calibrate_jumps(eps: np.ndarray):
    """
    Identify jumps as |eps| > threshold * std.
    Fit jump size distribution.
    """
    sigma_eps = np.std(eps, ddof=1)
    threshold = JUMP_SIGMA_THRESHOLD * sigma_eps

    jump_mask = np.abs(eps) > threshold
    jump_eps  = eps[jump_mask]

    num_jumps = jump_mask.sum()
    total_days = len(eps)

    if num_jumps == 0:
        print("\n[WARN] No jumps detected with current threshold.")
        lam = 0.0
        mu_J = 0.0
        sig_J = 0.0
    else:
        lam = num_jumps / total_days      # jumps per day
        mu_J = np.mean(jump_eps)
        sig_J = np.std(jump_eps, ddof=1)

    print("\n=== Jump Calibration ===")
    print(f"Total days: {total_days}")
    print(f"Jumps found: {num_jumps}")
    print(f"lambda (per day) = {lam:.6f}")
    print(f"mu_J = {mu_J:.4f}")
    print(f"sig_J = {sig_J:.4f}")
    print(f"Jump threshold = {threshold:.4f}")

    return lam, mu_J, sig_J


# =========================
# 4. MAIN
# =========================

def main():
    S = load_sys_prices()

    kappa, theta, sigma_eps, eps = calibrate_ou(S)
    lam, mu_J, sig_J = calibrate_jumps(eps)

    print("\n=== Final Calibrated Parameters ===")
    print(f"kappa =   {kappa:.4f}")
    print(f"theta =   {theta:.4f}")
    print(f"sigma =   {sigma_eps:.4f}")
    print(f"lambda =  {lam:.6f}")
    print(f"mu_J =    {mu_J:.4f}")
    print(f"sig_J =   {sig_J:.4f}")

    params = {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma_eps),
        "lambda": float(lam),
        "mu_J": float(mu_J),
        "sig_J": float(sig_J),
    }

    with open("calibrated_nordpool_sys_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print("\n[INFO] Saved → calibrated_nordpool_sys_params.json")


if __name__ == "__main__":
    main()
