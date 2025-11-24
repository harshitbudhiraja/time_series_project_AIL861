# compare_models_plots.py
# Creates multi-model overlay plots per city for all key metrics, including ones
# that existed only in per-model folders before (branch miss rate, cycles/instructions,
# profiler *vs horizon*, RSS delta, etc.). Aggregates to avoid zig-zag.

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERF_DIR = os.path.join(BASE_DIR, "outputs", "perflogs")
OUT_DIR  = os.path.join(PERF_DIR, "_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = ["Gurgaon", "Patna"]
MODEL_DIRS = ["t5-tiny", "t5-small", "t5-base"]  # dir names under perflogs
MODEL_LABELS = {
    "t5-tiny":  "t5-tiny",
    "t5-small": "t5-small",
    "t5-base":  "t5-base",
}

# Helper: pick first present column name variant
def first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

# Robust PMU column candidates
CYCLES  = ["cpu_core/cycles:u/", "cycles:u", "cycles"]
INSTR   = ["cpu_core/instructions:u/", "instructions:u", "instructions"]
MISS    = ["cpu_core/cache-misses:u/", "cache-misses:u", "cache-misses"]
REFS    = ["cpu_core/cache-references:u/", "cache-references:u", "cache-references"]
BR      = ["cpu_core/branches:u/", "branches:u", "branches"]
BRM     = ["cpu_core/branch-misses:u/", "branch-misses:u", "branch-misses"]

def load_city_df(model_dir, city):
    # Prefer per-city CSV produced by main.py:
    p = os.path.join(PERF_DIR, model_dir, f"chronos_perf_{city}.csv")
    if not os.path.exists(p):
        # fallback to combined per-model CSV filtered by city
        p_all = os.path.join(PERF_DIR, model_dir, "chronos_perf.csv")
        if not os.path.exists(p_all):
            return None
        df = pd.read_csv(p_all)
        df = df[df["city"] == city]
        return df if len(df) else None
    return pd.read_csv(p)

def plot_ctx_overlay(city, metric_col, y_label, title_stub, fname_stub, agg_fn="mean"):
    """
    Plot metric vs context (days) at 24h horizon for all models.
    Aggregates by context_hours to avoid zig-zag. Saves to OUT_DIR.
    """
    plt.figure()
    any_curve = False
    for mdir in MODEL_DIRS:
        df = load_city_df(mdir, city)
        if df is None or metric_col not in df.columns:
            continue
        d = df[df["horizon_hours"] == 24].copy()
        if len(d) == 0:
            continue
        g = getattr(d.groupby("context_hours")[metric_col], agg_fn)().reset_index()
        g = g.sort_values("context_hours")
        x = (g["context_hours"] / 24).astype(int)
        plt.plot(x, g[metric_col], marker="o", label=MODEL_LABELS.get(mdir, mdir))
        any_curve = True
    if not any_curve:
        plt.close()
        return
    plt.title(f"{city} - {title_stub} (all models)")
    plt.xlabel("Context length (days)")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    out = os.path.join(OUT_DIR, f"{city}_{fname_stub}_all_models.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_ctx_overlay_pmu_ratio(city, num_cands, den_cands, title_stub, y_label, fname_stub, scale=1.0, pct=False):
    """
    Plot ratio derived from PMU counters vs context (24h horizon) for all models.
    """
    plt.figure()
    any_curve = False
    for mdir in MODEL_DIRS:
        df = load_city_df(mdir, city)
        if df is None:
            continue
        num = first_present(df.columns, num_cands)
        den = first_present(df.columns, den_cands)
        if num is None or den is None:
            continue
        d = df[df["horizon_hours"] == 24].copy()
        if len(d) == 0:
            continue
        g = d.groupby("context_hours")[[num, den]].mean().reset_index().sort_values("context_hours")
        val = (g[num] / g[den].replace(0, np.nan)) * scale
        if pct:
            val = val * 100.0
        x = (g["context_hours"] / 24).astype(int)
        plt.plot(x, val, marker="o", label=MODEL_LABELS.get(mdir, mdir))
        any_curve = True
    if not any_curve:
        plt.close()
        return
    plt.title(f"{city} - {title_stub} (all models)")
    plt.xlabel("Context length (days)")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    out = os.path.join(OUT_DIR, f"{city}_{fname_stub}_all_models.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_ctx_overlay_pmu_single(city, cands, title_stub, y_label, fname_stub, scale=1.0):
    """
    Plot a single PMU counter vs context (24h horizon) overlayed for all models.
    (Used to split cycles and instructions into separate simple overlays)
    """
    plt.figure()
    any_curve = False
    for mdir in MODEL_DIRS:
        df = load_city_df(mdir, city)
        if df is None:
            continue
        col = first_present(df.columns, cands)
        if col is None:
            continue
        d = df[df["horizon_hours"] == 24].copy()
        if len(d) == 0:
            continue
        g = d.groupby("context_hours")[col].mean().reset_index().sort_values("context_hours")
        x = (g["context_hours"] / 24).astype(int)
        plt.plot(x, g[col] * scale, marker="o", label=MODEL_LABELS.get(mdir, mdir))
        any_curve = True
    if not any_curve:
        plt.close()
        return
    plt.title(f"{city} - {title_stub} (all models)")
    plt.xlabel("Context length (days)")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    out = os.path.join(OUT_DIR, f"{city}_{fname_stub}_all_models.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_hor_overlay(city, metric_col, y_label, title_stub, fname_stub, ctx_days_fixed=10, agg_fn="mean"):
    """
    Plot metric vs horizon (hours) at fixed context for all models.
    Aggregated by horizon_hours for clarity.
    """
    plt.figure()
    any_curve = False
    ctx_hours = ctx_days_fixed * 24
    for mdir in MODEL_DIRS:
        df = load_city_df(mdir, city)
        if df is None or metric_col not in df.columns:
            continue
        d = df[df["context_hours"] == ctx_hours].copy()
        if len(d) == 0:
            continue
        g = getattr(d.groupby("horizon_hours")[metric_col], agg_fn)().reset_index()
        g = g.sort_values("horizon_hours")
        plt.plot(g["horizon_hours"], g[metric_col], marker="o", label=MODEL_LABELS.get(mdir, mdir))
        any_curve = True
    if not any_curve:
        plt.close()
        return
    plt.title(f"{city} - {title_stub} (all models)")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    out = os.path.join(OUT_DIR, f"{city}_{fname_stub}_all_models.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_scatter_overlay(city, x_col, y_col, x_label, y_label, title_stub, fname_stub):
    """
    Scatter of y vs x for all models (no aggregation).
    Used for GFLOP/s vs latency.
    """
    plt.figure()
    any_points = False
    for mdir in MODEL_DIRS:
        df = load_city_df(mdir, city)
        if df is None or x_col not in df.columns or y_col not in df.columns:
            continue
        d = df.dropna(subset=[x_col, y_col]).copy()
        if len(d) == 0:
            continue
        plt.scatter(d[x_col], d[y_col], s=16, label=MODEL_LABELS.get(mdir, mdir))
        any_points = True
    if not any_points:
        plt.close()
        return
    plt.title(f"{city} - {title_stub} (all models)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    out = os.path.join(OUT_DIR, f"{city}_{fname_stub}_all_models.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()

def main():
    for city in CITIES:
        # Already present in _comparisons (kept)
        plot_ctx_overlay(city, "latency_s",      "Latency (seconds)",           "Latency vs Context",            "latency_vs_context")
        plot_hor_overlay(city, "latency_s",      "Latency (seconds)",           "Latency vs Horizon",            "latency_vs_horizon")
        plot_ctx_overlay(city, "rmse",           "RMSE (units of series)",      "RMSE vs Context",               "rmse_vs_context")
        plot_hor_overlay(city, "rmse",           "RMSE (units of series)",      "RMSE vs Horizon",               "rmse_vs_horizon")
        plot_ctx_overlay(city, "mae",            "MAE (units of series)",       "MAE vs Context",                "mae_vs_context")
        plot_hor_overlay(city, "mae",            "MAE (units of series)",       "MAE vs Horizon",                "mae_vs_horizon")
        plot_ctx_overlay(city, "prof_latency_s", "Profiler latency (seconds)",  "Profiler latency vs Context",   "prof_latency_vs_context")
        plot_ctx_overlay(city, "prof_cpu_total_s","Profiler CPU total (seconds)","Profiler CPU total vs Context","prof_cputotal_vs_context")
        plot_ctx_overlay(city, "proc_rss_bytes_after", "Process RSS after (MB)", "Process RSS (after) vs Context","rss_after_vs_context", agg_fn="mean")
        plot_scatter_overlay(city, "latency_s", "gflops_per_s", "Latency (seconds)", "Throughput (GFLOP/s)", "GFLOP/s vs Latency", "gflops_vs_latency")

        # --------- NEW: counterparts you saw missing ---------
        # Profiler vs HORIZON (fixed 10d)
        plot_hor_overlay(city, "prof_latency_s",   "Profiler latency (seconds)",    "Profiler latency vs Horizon",     "prof_latency_vs_horizon")
        plot_hor_overlay(city, "prof_cpu_total_s", "Profiler CPU total (seconds)",  "Profiler CPU total vs Horizon",   "prof_cputotal_vs_horizon")
        plot_hor_overlay(city, "proc_rss_bytes_after", "Process RSS after (MB)",    "Process RSS (after) vs Horizon",  "rss_after_vs_horizon")

        # RSS Δ vs CONTEXT
        plot_ctx_overlay(city, "proc_rss_delta_bytes", "Process RSS delta (MB)",    "Process RSS Δ vs Context",        "rss_delta_vs_context", agg_fn="mean")

        # Branch miss rate vs CONTEXT (derived)
        plot_ctx_overlay_pmu_ratio(
            city, BRM, BR, "Branch miss rate vs Context", "Branch miss rate (%)",
            "branch_missrate_vs_context", pct=True
        )

        # IPC vs CONTEXT (derived)
        plot_ctx_overlay_pmu_ratio(
            city, INSTR, CYCLES, "IPC vs Context", "IPC (instructions/cycle)",
            "ipc_vs_context", pct=False
        )

        # Miss rate vs CONTEXT (derived)
        plot_ctx_overlay_pmu_ratio(
            city, MISS, REFS, "Cache miss rate vs Context", "Cache miss rate (%)",
            "missrate_vs_context", pct=True
        )

        # Cycles/Instructions vs CONTEXT — split into two clean overlays (G = 1e9)
        plot_ctx_overlay_pmu_single(
            city, CYCLES, "Cycles vs Context", "Count (billions)", "cycles_vs_context", scale=1e-9
        )
        plot_ctx_overlay_pmu_single(
            city, INSTR,  "Instructions vs Context", "Count (billions)", "instructions_vs_context", scale=1e-9
        )

        # (If available in CSVs) IPC & Miss rate vs HORIZON at fixed 10d
        # IPC vs HORIZON
        plt.figure()
        any_curve = False
        for mdir in MODEL_DIRS:
            df = load_city_df(mdir, city)
            if df is None:
                continue
            ins = first_present(df.columns, INSTR)
            cyc = first_present(df.columns, CYCLES)
            if ins is None or cyc is None:
                continue
            dH = df[df["context_hours"] == 10 * 24]
            if len(dH) == 0:
                continue
            g = dH.groupby("horizon_hours")[[ins, cyc]].mean().reset_index().sort_values("horizon_hours")
            ipc = g[ins] / g[cyc].replace(0, np.nan)
            plt.plot(g["horizon_hours"], ipc, marker="o", label=MODEL_LABELS.get(mdir, mdir))
            any_curve = True
        if any_curve:
            plt.title(f"{city} - IPC vs Horizon (all models)")
            plt.xlabel("Forecast horizon (hours)")
            plt.ylabel("IPC (instructions/cycle)")
            plt.grid(True); plt.legend()
            out = os.path.join(OUT_DIR, f"{city}_ipc_vs_horizon_all_models.png")
            plt.savefig(out, bbox_inches="tight"); plt.close()
        else:
            plt.close()

        # Miss rate vs HORIZON
        plt.figure()
        any_curve = False
        for mdir in MODEL_DIRS:
            df = load_city_df(mdir, city)
            if df is None:
                continue
            miss = first_present(df.columns, MISS)
            refs = first_present(df.columns, REFS)
            if miss is None or refs is None:
                continue
            dH = df[df["context_hours"] == 10 * 24]
            if len(dH) == 0:
                continue
            g = dH.groupby("horizon_hours")[[miss, refs]].mean().reset_index().sort_values("horizon_hours")
            mrate = (g[miss] / g[refs].replace(0, np.nan)) * 100.0
            plt.plot(g["horizon_hours"], mrate, marker="o", label=MODEL_LABELS.get(mdir, mdir))
            any_curve = True
        if any_curve:
            plt.title(f"{city} - Miss rate vs Horizon (all models)")
            plt.xlabel("Forecast horizon (hours)")
            plt.ylabel("Cache miss rate (%)")
            plt.grid(True); plt.legend()
            out = os.path.join(OUT_DIR, f"{city}_missrate_vs_horizon_all_models.png")
            plt.savefig(out, bbox_inches="tight"); plt.close()
        else:
            plt.close()

if __name__ == "__main__":
    main()
