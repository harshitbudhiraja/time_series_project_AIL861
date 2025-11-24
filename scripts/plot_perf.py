# plot_perf.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "../outputs/perflogs/chronos_perf_compute_log.csv"
OUT_DIR  = "../outputs/perflogs"

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# light cleanups
df["context_days"] = (df["context_hours"] // 24).astype(int)

# Some rows have perf only on sampled its; create friendly derived metrics where possible
def safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    b[b == 0] = np.nan
    return a / b

# ipc and cache miss rate may already be present; if not, derive for core counters when available
if "ipc" not in df.columns and "cpu_core/instructions:u/" in df.columns and "cpu_core/cycles:u/" in df.columns:
    df["ipc"] = safe_div(df["cpu_core/instructions:u/"], df["cpu_core/cycles:u/"])

if "cache_miss_rate" not in df.columns and "cpu_core/cache-misses:u/" in df.columns and "cpu_core/cache-references:u/" in df.columns:
    df["cache_miss_rate"] = safe_div(df["cpu_core/cache-misses:u/"], df["cpu_core/cache-references:u/"])

# ---------- 1) Latency vs context (24 h horizon) ----------
d1 = (
    df[df["horizon_hours"] == 24]
    .groupby(["city", "model", "context_days"], as_index=False)["latency_s"].mean()
)

for (city, model), g in d1.groupby(["city", "model"]):
    plt.figure()
    plt.plot(g["context_days"], g["latency_s"], marker="o")
    plt.xlabel("Context length (days)")
    plt.ylabel("Mean latency (s)")
    plt.title(f"{city} - Latency vs context - {model.split('/')[-1]}")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"{city}_latency_vs_context_{model.split('/')[-1]}.png"),
                bbox_inches="tight")
    plt.close()

# ---------- 2) Latency vs horizon (10 day context) ----------
d2 = (
    df[df["context_days"] == 10]
    .groupby(["city", "model", "horizon_hours"], as_index=False)["latency_s"].mean()
)

for (city, model), g in d2.groupby(["city", "model"]):
    if len(g) == 0: 
        continue
    plt.figure()
    plt.plot(g["horizon_hours"], g["latency_s"], marker="o")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("Mean latency (s)")
    plt.title(f"{city} - Latency vs horizon - {model.split('/')[-1]}")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, f"{city}_latency_vs_horizon_{model.split('/')[-1]}.png"),
                bbox_inches="tight")
    plt.close()

# ---------- 3) GFLOP/s vs latency scatter (where flops recorded) ----------
if "gflops_per_s" in df.columns:
    d3 = df.dropna(subset=["gflops_per_s"])
    for (city, model), g in d3.groupby(["city", "model"]):
        if len(g) == 0:
            continue
        plt.figure()
        plt.scatter(g["latency_s"], g["gflops_per_s"], s=16)
        plt.xlabel("Latency (s)")
        plt.ylabel("GFLOP/s")
        plt.title(f"{city} - GFLOP/s vs latency - {model.split('/')[-1]}")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, f"{city}_gflops_vs_latency_{model.split('/')[-1]}.png"),
                    bbox_inches="tight")
        plt.close()

# ---------- 4) IPC vs context and cache miss rate vs context (24 h horizon) ----------
if "ipc" in df.columns:
    d4 = (
        df[df["horizon_hours"] == 24]
        .dropna(subset=["ipc"])
        .groupby(["city", "model", "context_days"], as_index=False)["ipc"].mean()
    )
    for (city, model), g in d4.groupby(["city", "model"]):
        if len(g) == 0:
            continue
        plt.figure()
        plt.plot(g["context_days"], g["ipc"], marker="o")
        plt.xlabel("Context length (days)")
        plt.ylabel("IPC")
        plt.title(f"{city} - IPC vs context - {model.split('/')[-1]}")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, f"{city}_ipc_vs_context_{model.split('/')[-1]}.png"),
                    bbox_inches="tight")
        plt.close()

if "cache_miss_rate" in df.columns:
    d5 = (
        df[df["horizon_hours"] == 24]
        .dropna(subset=["cache_miss_rate"])
        .groupby(["city", "model", "context_days"], as_index=False)["cache_miss_rate"].mean()
    )
    for (city, model), g in d5.groupby(["city", "model"]):
        if len(g) == 0:
            continue
        plt.figure()
        plt.plot(g["context_days"], g["cache_miss_rate"], marker="o")
        plt.xlabel("Context length (days)")
        plt.ylabel("Cache miss rate")
        plt.title(f"{city} - Miss rate vs context - {model.split('/')[-1]}")
        plt.grid(True)
        plt.savefig(os.path.join(OUT_DIR, f"{city}_missrate_vs_context_{model.split('/')[-1]}.png"),
                    bbox_inches="tight")
        plt.close()

# ---------- 5) Bars for cycles and instructions at 24 h horizon ----------
core_cycles = "cpu_core/cycles:u/"
core_instr  = "cpu_core/instructions:u/"
if core_cycles in df.columns and core_instr in df.columns:
    d6 = (
        df[df["horizon_hours"] == 24]
        .dropna(subset=[core_cycles, core_instr])
        .groupby(["city", "model", "context_days"], as_index=False)[[core_cycles, core_instr]].mean()
    )
    for (city, model), g in d6.groupby(["city", "model"]):
        if len(g) == 0:
            continue
        x = np.arange(len(g))
        w = 0.4
        plt.figure()
        plt.bar(x - w/2, g[core_cycles]/1e9, width=w, label="cycles (G)")
        plt.bar(x + w/2, g[core_instr]/1e9, width=w, label="instructions (G)")
        plt.xticks(x, g["context_days"])
        plt.xlabel("Context length (days)")
        plt.ylabel("Count (billions)")
        plt.title(f"{city} - cycles and instructions vs context - {model.split('/')[-1]}")
        plt.legend()
        plt.grid(axis="y")
        plt.savefig(os.path.join(OUT_DIR, f"{city}_cycles_instr_vs_context_{model.split('/')[-1]}.png"),
                    bbox_inches="tight")
        plt.close()

print("Saved plots to", OUT_DIR)
