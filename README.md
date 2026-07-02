# WQU-MScFE-690-Portfolio-Optimization
 # Machine Learning (Deep Learning) Investment Strategies 
 ## Deep Reinforcement Learning for Dynamic Portfolio Allocation with CVaR Risk Constraints

An autonomous, friction-aware asset allocation engine built on continuous Deep Reinforcement Learning (DRL) using Proximal Policy Optimization (PPO). This framework dynamically rebalances capital across an institutionally diversified multi-asset universe under explicit tail-risk constraints and execution friction penalties.

## 📊 Project Architecture & State Space
The optimization engine models capital allocation as a continuous Markov Decision Process (MDP) executed within a custom OpenAI Gymnasium container. 

The engine ingests a high-fidelity **84-dimensional continuous feature matrix** spanning an 11-year dataset (February 2015 – May 2026) containing 2,848 daily trading observations:
* **56 Asset-Level Features:** 1-day, 5-day, and 21-day continuous log returns; 20-day rolling annualized volatility; price-to-SMA20 trend ratios; and normalized RSI indicators for each asset.
* **28 Global Correlation Features:** The extracted lower triangle of a moving 8×8 cross-asset Pearson correlation matrix to actively capture changing diversification boundaries across regimes.

### 🌐 Asset Universe Configuration
The engine evaluates real cross-asset allocation decisions across 8 highly liquid global Exchange-Traded Funds (ETFs):
* `SPY` (US Large-Cap Equities)
* `IWM` (US Small-Cap Equities)
* `VGK` (European Equities)
* `EEM` (Emerging Market Equities)
* `AGG` (US Aggregate Bonds)
* `TLT` (Long-Duration Treasuries)
* `GLD` (Gold Commodity)
* `VNQ` (Real Estate Infrastructure)

---

## 🗂 Repository Structure

* **`8_selected_ETF_Update_1.ipynb`** — The core Jupyter notebook housing the end-to-end software pipeline: feature engineering, custom Gymnasium environment loops, stable PPO agent training updates, out-of-sample backtesting engines, and plotting dashboards.
* **`data/etf_drl_state_space.csv`** — The pre-computed 84-dimensional continuous market signal dataset used for reproducible training runs.
* **`requirements.txt`** — The explicit Python software dependencies and verified package versions required to execute the pipeline securely.
* **`README.md`** — Comprehensive environment setup documentation, framework parameters, and institutional performance sheets.

---

## 🚀 Quick Start & Installation

### 1. Environment Setup
Clone this repository to your local machine or mount it inside your cloud server container:
```bash
git clone [https://github.com/hellothurein/WQU-MScFE-690-Portfolio-Optimization.git](https://github.com/hellothurein/WQU-MScFE-690-Portfolio-Optimization.git)
cd WQU-MScFE-690-Portfolio-Optimization
