# Calibration.py
import numpy as np
import pandas as pd
import json

# =========================
# CONFIG – MATCHES NP.csv
# =========================
CSV_PATH = "NP.csv"          # file must be in the same folder as this script
DATETIME_COL_RAW = "Date"    # column name in NP.csv (after stripping spaces)
PRICE_COL_RAW = "Price"      # column name in NP.csv (after stripping spaces)
DATA_IS_HOURLY = True        # NP.csv is hourly → average to daily
JUMP_SIGMA_THRESHOLD = 3.0   # jump threshold in units of residual std dev


# =========================
# 1) LOAD & CLEAN DATA
# =========================
def load_sys_prices() -> pd.Series:
    """
    Load NP.csv, clean columns, parse datetime, and convert to a daily average
    price series S_t (one value per day).
    """
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    if DATETIME_COL_RAW not in df.columns:
        raise RuntimeError(
            f"[FATAL] Column '{DATETIME_COL_RAW}' not found. Available: {list(df.columns)}"
        )
    if PRICE_COL_RAW not in df.columns:
        raise RuntimeError(
            f"[FATAL] Column '{PRICE_COL_RAW}' not found. Available: {list(df.columns)}"
        )

    df[DATETIME_COL_RAW] = pd.to_datetime(df[DATETIME_COL_RAW], errors="coerce")
    df[PRICE_COL_RAW] = pd.to_numeric(df[PRICE_COL_RAW], errors="coerce")
    df = df.dropna(subset=[DATETIME_COL_RAW, PRICE_COL_RAW])

    df = df.sort_values(DATETIME_COL_RAW).set_index(DATETIME_COL_RAW)

    if DATA_IS_HOURLY:
        daily = df[PRICE_COL_RAW].resample("D").mean()
    else:
        daily = df[PRICE_COL_RAW]

    daily = daily.dropna()

    print(
        f"[LOAD] Daily SYS prices: {len(daily)} points from {daily.index[0].date()} to {daily.index[-1].date()}"
    )
    return daily


# =========================
# 2) OU (AR(1)) CALIBRATION
# =========================
def calibrate_ou_ar1(S: pd.Series, dt: float = 1.0):
    """
    Fit AR(1): S_{t+dt} = a + b S_t + eps_t
    Map to OU:
        b = exp(-kappa*dt)
        theta = a / (1 - b)

    Returns:
        kappa, theta, eps, a, b
    """
    if len(S) < 10:
        raise RuntimeError("[FATAL] Not enough data points to calibrate OU/AR(1).")

    S_t = S[:-1].values
    S_tp1 = S[1:].values

    X = np.vstack([np.ones_like(S_t), S_t]).T
    y = S_tp1

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    # Innovation series
    eps = y - (X @ beta)

    # Safety: OU mapping assumes 0 < b < 1
    if b <= 0 or b >= 1:
        raise RuntimeError(
            f"[FATAL] AR(1) coefficient b={b:.6f} not in (0,1). "
            "OU mapping unstable. Consider de-seasonalizing, changing sampling, or modeling log(price+c)."
        )

    kappa = -np.log(b) / dt
    theta = a / (1 - b)

    return kappa, theta, eps, a, b


def ou_sigma_from_eps(sigma_eps: float, kappa: float, dt: float = 1.0) -> float:
    """
    Convert AR(1) innovation std (sigma_eps) to OU diffusion sigma using:
      Var(eps) = (sigma^2/(2kappa)) * (1 - exp(-2kappa dt))
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    denom = (1 - np.exp(-2 * kappa * dt))
    if denom <= 0:
        raise ValueError("Invalid OU sigma mapping denominator.")
    factor = (2 * kappa) / denom
    return float(sigma_eps * np.sqrt(factor))


# =========================
# 3) JUMP CALIBRATION + DIFFUSION REFINEMENT
# =========================
def calibrate_jumps_and_refine_sigma(eps: np.ndarray, kappa: float, dt: float = 1.0):
    """
    Detect jumps from AR(1) innovations eps using a threshold rule,
    estimate jump intensity and jump-size distribution from jump residuals,
    and re-estimate diffusion sigma from non-jump residuals.

    Returns:
        lambda_d, mu_J, sig_J, sigma_d (OU diffusion per day)
    """
    sigma_eps_all = np.std(eps, ddof=1)
    threshold = JUMP_SIGMA_THRESHOLD * sigma_eps_all

    jump_mask = np.abs(eps) > threshold
    jump_eps = eps[jump_mask]
    nonjump_eps = eps[~jump_mask]

    total_days = len(eps)
    num_jumps = int(jump_mask.sum())

    lam = num_jumps / total_days  # per day

    if num_jumps > 0:
        mu_J = float(np.mean(jump_eps))
        sig_J = float(np.std(jump_eps, ddof=1)) if num_jumps > 1 else 0.0
    else:
        mu_J, sig_J = 0.0, 0.0

    # Diffusion innovation std excluding jumps
    sigma_eps_diff = float(np.std(nonjump_eps, ddof=1)) if len(nonjump_eps) > 1 else float(sigma_eps_all)
    sigma_d = ou_sigma_from_eps(sigma_eps_diff, kappa, dt=dt)

    return lam, mu_J, sig_J, sigma_d, threshold, sigma_eps_all, sigma_eps_diff, num_jumps, total_days


# =========================
# 4) MAIN
# =========================
def main():
    S = load_sys_prices()

    kappa, theta, eps, a, b = calibrate_ou_ar1(S, dt=1.0)

    (
        lam_d,
        mu_J,
        sig_J,
        sigma_d,
        threshold,
        sigma_eps_all,
        sigma_eps_diff,
        num_jumps,
        total_days,
    ) = calibrate_jumps_and_refine_sigma(eps, kappa, dt=1.0)

    print("\n[LOAD] Calibrated parameters:")
    print(f"  kappa   = {kappa:.4f}")
    print(f"  theta   = {theta:.4f}")
    print(f"  sigma_d = {sigma_d:.4f}  (diffusion sigma per day)")
    print(f"  lambda_d= {lam_d:.6f}  (jump intensity per day)")
    print(f"  mu_J    = {mu_J:.4f}")
    print(f"  sig_J   = {sig_J:.4f}")

    print("\n[DIAGNOSTICS]")
    print(f"  AR(1): a = {a:.4f}, b = {b:.6f}")
    print(f"  eps std (all)     = {sigma_eps_all:.4f}")
    print(f"  eps std (nonjump) = {sigma_eps_diff:.4f}")
    print(f"  jump threshold    = {threshold:.4f}")
    print(f"  jumps found       = {num_jumps} out of {total_days} days")

    params = {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma_d": float(sigma_d),
        "lambda_d": float(lam_d),
        "mu_J": float(mu_J),
        "sig_J": float(sig_J),
        "jump_threshold_eps": float(threshold),
    }

    with open("calibrated_nordpool_sys_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print("\n[INFO] Saved → calibrated_nordpool_sys_params.json")


if __name__ == "__main__":
    main()
