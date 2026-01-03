# Black–Scholes Extension for Electricity Options

## Overview
This project extends classical Black–Scholes option pricing to electricity markets by incorporating mean reversion and jump risk. Electricity prices are non-storable and exhibit rare but extreme spikes, causing standard geometric Brownian motion assumptions to break down. The framework implemented here provides a structurally motivated alternative that better reflects electricity market dynamics.

---

## Motivation
Black–Scholes assumes:
- Lognormal price dynamics  
- Constant volatility  
- No jumps  
- No mean reversion  

Empirical electricity prices violate all of these assumptions. Power prices display strong mean reversion, heavy tails, skewness, and infrequent but severe price spikes driven by supply constraints and demand shocks. These features motivate modeling electricity prices using a mean-reverting jump diffusion rather than a pure diffusion process.

---

## Price Dynamics
The electricity spot price is modeled as:

dSₜ = κ(θ − Sₜ) dt + σ dWₜ + J dNₜ

where:
- κ is the mean reversion speed  
- θ is the long-run mean price  
- σ is the diffusion volatility  
- Wₜ is Brownian motion  
- Nₜ is a Poisson jump process with intensity λ  
- J represents jump sizes  

This formulation preserves analytical tractability while explicitly capturing spike risk, a defining characteristic of electricity markets.

---

## Calibration
Model parameters are calibrated using historical Nord Pool system price data (`NP.csv`).

### Mean Reversion and Diffusion
An AR(1) representation of the Ornstein–Uhlenbeck process is estimated on daily average prices to obtain κ and θ. Diffusion volatility is estimated from non-jump residuals to avoid attributing spike behavior to the continuous component.

### Jump Risk
Jumps are identified using a threshold rule based on standardized residuals. Jump intensity and jump-size distribution parameters are estimated from these extreme observations.

All calibration steps are implemented using NumPy, and calibrated parameters are saved for reproducibility.

---

## Black–Scholes Benchmark
For comparison, synthetic at-the-money options are priced using the Black–Scholes formula with:
- Identical maturities  
- The same initial price  
- Volatility calibrated from realized historical returns  

This isolates the impact of model structure rather than parameter scaling.

---

## Empirical Evaluation

### 1. Distribution Across Horizons
Model-implied price change distributions are compared with empirical distributions over multiple horizons. The jump–mean-reverting model produces heavier tails and higher dispersion at longer horizons, consistent with spike-driven electricity prices.

### 2. Volatility Term Structure
Realized electricity volatility declines with maturity, while the model-implied volatility remains elevated due to persistent jump risk. This highlights a key limitation of Black–Scholes in electricity markets.

### 3. Synthetic Option Pricing
Synthetic at-the-money option prices are compared under both models. Results show:
- Black–Scholes may overprice very short-dated electricity options when volatility is driven purely by recent realized variation.
- For medium and long maturities, the jump–mean-reverting model produces systematically higher option values.
- The pricing gap widens with maturity, reflecting accumulated spike risk ignored by Black–Scholes.

---

## Key Result
Black–Scholes systematically underprices medium- and long-dated electricity options because it cannot capture jump risk and persistent mean-reverting dynamics. Incorporating these features produces higher and more realistic option values aligned with observed electricity market behavior.

---

## Project Structure
- `NP.csv`  
  Historical Nord Pool system price data.

- `Calibration.py`  
  Calibration of mean reversion, diffusion volatility, jump intensity, and jump-size parameters.

- `calibrated_nordpool_sys_params.json`  
  Stored calibrated parameters for reproducible analysis.

- `Power_Options_Model.py`  
  Implementation of the mean-reverting jump diffusion price dynamics and analytical option valuation.

- `Option_Tests.py`  
  Empirical comparisons between the extended model and Black–Scholes benchmarks.

---

## Notes
Calibration is performed under the physical measure and does not include risk-neutral adjustments or seasonal de-composition. These extensions are natural next steps but are not required to demonstrate the structural limitations of Black–Scholes in electricity markets.

---

## Disclaimer
This project is for educational and research purposes only and does not constitute financial advice or a trading recommendation.
