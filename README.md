# WQU MScFE 690 — Portfolio Optimization Capstone

## Deep Reinforcement Learning for Dynamic Portfolio Allocation with CVaR Risk Constraints

**Authors:** Innocent Ndagijimana ([innondag@gmail.com](mailto:innondag@gmail.com)) · Thu Rein Win ([hellothurein@gmail.com](mailto:hellothurein@gmail.com)) **Track:** 8 — Machine Learning (Deep Learning) Investment Strategies · Topic 2: ML Application to Portfolio Allocation

An autonomous, friction-aware asset allocation engine built on continuous Deep Reinforcement Learning using Proximal Policy Optimization (PPO). The agent dynamically rebalances capital across a diversified multi-asset ETF universe under an explicit tail-risk (CVaR) penalty and a 10 bps turnover friction cost.

## Project Architecture & State Space

Capital allocation is modelled as a continuous Markov Decision Process inside a custom Gymnasium environment (`MultiAssetPortfolioEnv`).

The agent ingests an **84-dimensional continuous feature matrix** spanning 20 years (January 2006 – May 2026), 5,112 daily observations:

- **56 asset-level features** (7 per ETF): adjusted price, 1/5/21-day log returns, 20-day annualized rolling volatility, price-to-SMA20 trend ratio, and normalized RSI. Note: the raw adjusted price is included in the state; its non-stationarity is a documented limitation.  
- **28 global correlation features:** the lower triangle of the rolling 20-day 8×8 Pearson correlation matrix, capturing changing diversification structure across regimes.

### Asset Universe

8 highly liquid global ETFs: `SPY` (US large-cap), `IWM` (US small-cap), `VGK` (European equities), `EEM` (emerging markets), `AGG` (US aggregate bonds), `TLT` (long-duration Treasuries), `GLD` (gold), `VNQ` (real estate).

## Repository Structure

| File | Description |
| :---- | :---- |
| `8_selected_ETF_Capstone_Project.ipynb` | End-to-end pipeline: feature engineering, Gymnasium environment, PPO training (500k timesteps, SEED=42), out-of-sample backtest, benchmarks, dashboards |
| `Data/etf_drl_state_space.csv` | Pre-computed 84-dimensional state-space dataset (2006-02-02 → 2026-05-29) for reproducible runs |
| `requirements.txt` | Python dependencies matching the environment of record (Google Colab, Python 3.12) |
| `README.md` | This file |

## Quick Start

git clone https://github.com/hellothurein/WQU-MScFE-690-Portfolio-Optimization.git

cd WQU-MScFE-690-Portfolio-Optimization

pip install \-r requirements.txt

Open `8_selected_ETF_Capstone_Project.ipynb` (runs directly in Google Colab via the badge in the notebook) and execute cells top to bottom. The global seed (`SEED=42`) makes results reproducible under the pinned package versions.

## Methodology Summary

- **Split:** In-sample 2006–2023 (4,508 days) · Out-of-sample 2024–2026 (604 days, walk-forward, no lookahead)  
- **Reward:** `net_return − 0.1 × |CVaR₀.₀₅(rolling 20-day returns)|`, with 10 bps transaction cost on turnover  
- **Action space:** Box(0,1,(8,)) softmax-normalized; SB3 action clipping structurally bounds each weight to ≈ \[5%, 28%\], enforcing diversification  
- **Benchmarks:** Markowitz minimum-variance (rolling 252-day covariance, SLSQP, walk-forward) and equal-weight (1/N)

## Out-of-Sample Results (2024–2026, 604 trading days)

| Metric | PPO \+ CVaR | Markowitz MV | Equal-Weight |
| :---- | :---- | :---- | :---- |
| Total return | 32.38% | 21.81% | 40.24% |
| Annualized return | 12.24% | 8.41% | 14.79% |
| Annualized volatility | 9.95% | 5.56% | 11.29% |
| Sharpe ratio | 1.23 | **1.51** | 1.31 |
| Sortino ratio | 1.68 | **2.08** | 1.79 |
| Max drawdown | −9.65% | **−3.28%** | −9.97% |
| Daily 95% VaR | −0.98% | **−0.50%** | −1.09% |

**Regime-conditional finding:** Markowitz dominates the 2024 bull market (Sharpe 1.83 vs PPO 0.62); PPO dominates the 2025–2026 volatility-shift regime (Sharpe 1.62 vs Markowitz 1.24), nearly matching equal-weight (1.63).

## Reproducibility Notes

- All results were produced on Google Colab (Python 3.12, CPU) with `SEED=42` set for `random`, `numpy`, `torch`, and the PPO constructor.  
- DRL results are seed-sensitive (observed Sharpe 1.29 vs 1.20 across seeds at 2M timesteps); single-run figures are one draw from a distribution. See the report's Limitations section.  
- Environment of record: `stable-baselines3 2.9.0`, `gymnasium 1.3.0`, `numpy 2.0.2`, `torch 2.11.0`, `pandas 2.2.2` (see `requirements.txt`).

## License / Academic Use

Submitted as part of the WorldQuant University MScFE 690 Capstone. For academic review purposes.  
