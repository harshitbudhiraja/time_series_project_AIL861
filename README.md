# Time Series Forecasting with Chronos Models

A performance analysis and profiling project for Chronos-T5 time series forecasting models, evaluating latency, FLOPs, memory usage, and forecast quality metrics.

## Overview

This project benchmarks Chronos-T5 models (tiny, small, base) on time series data from Gurgaon and Patna cities. It measures:
- Inference latency and throughput
- FLOPs and GFLOP/s
- Memory usage (RSS)
- Hardware performance counters (via `perf`)
- Forecast quality metrics (RMSE, MAE)

## Setup

1. Create conda environment from `env.yml`:
   ```bash
   conda env create -f env.yml
   conda activate chronos
   ```

2. Install additional dependencies if needed (should be included in env.yml)

## Usage

Run the main performance analysis script:

```bash
cd scripts
SUDO_PERF=1 python main.py
```

The script will:
- Load Chronos-T5 models (tiny, small, base)
- Run forecasts on Gurgaon and Patna time series data
- Sweep context lengths and forecast horizons
- Generate performance logs and plots in `../outputs/perflogs/`

## Project Structure

```
├── data/              # Time series data files (Gurgaon, Patna)
├── scripts/            # Main analysis scripts
│   ├── main.py        # Main performance logger
│   └── ...
├── report/             # LaTeX report and generated plots
├── outputs/            # Performance logs and plots
└── env.yml            # Conda environment specification
```

## Data

- `gurgaon_clean.csv` / `df_ggn_covariates.csv` - Gurgaon time series data
- `patna_clean.csv` / `df_patna_covariates.csv` - Patna time series data

## Outputs

Results are saved to `outputs/perflogs/` including:
- CSV logs with performance metrics
- Plots for latency, FLOPs, memory, and quality metrics
- Per-API breakdowns for profiled operations

