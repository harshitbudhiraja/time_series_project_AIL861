# run as: python api_call_and_plot.py
# Purpose: Collect ONLY per-API (per-op) profiler breakdowns across models/contexts/horizons
# and write API-focused plots under  ../outputs/perflogs/apicalls/<model-tag>/
#
# Notes:
# - Reuses the same data/pipeline logic as main.py
# - Disables "perf" to keep this fast and deterministic
# - Forces prof_every=1 and keeps max_iters small by default (override via env)
# - All plots carry units; aggregation avoids zig-zag lines.

import os, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from chronos import BaseChronosPipeline
from torch.profiler import profile, record_function, ProfilerActivity

OUT_ROOT = "../outputs/perflogs/apicalls"

MODELS = ["amazon/chronos-t5-small", "amazon/chronos-t5-tiny", "amazon/chronos-t5-base"]
DATA = {
    "ggn": "../data/df_ggn_covariates.csv",
    "pna": "../data/df_patna_covariates.csv",
}
CONTEXT_DAYS_SWEEP  = [2, 4, 8, 10, 14]
HORIZON_HOURS_SWEEP = [4, 8, 12, 24, 48]
ROLLING_STEP_HOURS  = 24
ENV_MAX_ITERS = os.environ.get("API_MAX_ITERS")
MAX_ITERS = int(ENV_MAX_ITERS) if ENV_MAX_ITERS is not None else 3  # small, just to sample ops

os.makedirs(OUT_ROOT, exist_ok=True)

def load_series(path):
    df = pd.read_csv(path, usecols=[0, 4], parse_dates=[0])
    ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(y.values, index=ts).sort_index()
    s = s.asfreq("H").interpolate(limit_direction="both")
    return s.astype(np.float32)

def model_tag(model_name: str) -> str:
    return model_name.split("/")[-1].replace("chronos-", "").replace("t5-", "t5-")

def api_csv_path(out_dir, model_name: str, city: str) -> str:
    tag = model_tag(model_name); city_tag = city.replace(" ", "")
    md = os.path.join(out_dir, tag); os.makedirs(md, exist_ok=True)
    return os.path.join(md, f"api_breakdown_{city_tag}.csv")

def ensure_header(path: str):
    cols = [
        "ts_unix","model","city","context_hours","horizon_hours","iter_idx",
        "name","calls","self_cpu_total_us","cpu_total_us","cpu_time_avg_us","flops_total"
    ]
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        import csv
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()

def append_api_rows(path: str, meta_row: dict, per_ops: list[dict]):
    import csv
    cols = [
        "ts_unix","model","city","context_hours","horizon_hours","iter_idx",
        "name","calls","self_cpu_total_us","cpu_total_us","cpu_time_avg_us","flops_total"
    ]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        for op in per_ops:
            w.writerow({
                "ts_unix": meta_row.get("ts_unix", time.time()),
                "model": meta_row["model"],
                "city": meta_row["city"],
                "context_hours": meta_row["context_hours"],
                "horizon_hours": meta_row["horizon_hours"],
                "iter_idx": meta_row["iter_idx"],
                "name": op.get("name"),
                "calls": op.get("calls"),
                "self_cpu_total_us": op.get("self_cpu_total_us"),
                "cpu_total_us": op.get("cpu_total_us"),
                "cpu_time_avg_us": op.get("cpu_time_avg_us"),
                "flops_total": op.get("flops_total"),
            })

def one_infer_with_prof(pipe, hist, horizon):
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        with record_function("chronos_infer"):
            _, mean = pipe.predict_quantiles(context=hist, prediction_length=horizon, quantile_levels=[0.5])
    ka = prof.key_averages()
    rows = []
    for evt in ka:
        calls = int(evt.count or 0)
        cpu_total_us = float(evt.cpu_time_total or 0.0)
        self_cpu_total_us = float(evt.self_cpu_time_total or 0.0)
        cpu_time_avg_us = cpu_total_us / calls if calls > 0 else 0.0

        rows.append({
            "name": evt.key,
            "calls": calls,
            "self_cpu_total_us": self_cpu_total_us,
            "cpu_total_us": cpu_total_us,
            "cpu_time_avg_us": cpu_time_avg_us,
            "flops_total": int(getattr(evt, "flops", 0) or 0),
        })

    return rows

def plot_api_top(out_dir, model_name, city, title_suffix="overall"):
    csvp = api_csv_path(out_dir, model_name, city)
    if not os.path.exists(csvp) or os.path.getsize(csvp) == 0:
        return
    df = pd.read_csv(csvp)
    if df.empty or "cpu_total_us" not in df.columns or "name" not in df.columns:
        return
    agg = (df.groupby("name", as_index=False)["cpu_total_us"].mean()
             .sort_values("cpu_total_us", ascending=False)
             .head(15))
    if len(agg) == 0:
        return
    tag = model_tag(model_name); city_tag = city.replace(" ","")
    md = os.path.join(out_dir, tag); os.makedirs(md, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.barh(agg["name"][::-1], (agg["cpu_total_us"]/1000.0)[::-1])
    plt.xlabel("Average CPU total (ms)")    # units
    plt.ylabel("Operator / API")
    plt.title(f"{city} - Top ops by CPU time - {tag} ({title_suffix})")
    plt.tight_layout()
    out = os.path.join(md, f"{city_tag}_api_top_cpu_ms_{title_suffix}.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_api_cpu_by_context(out_dir, model_name, city):
    csvp = api_csv_path(out_dir, model_name, city)
    if not os.path.exists(csvp) or os.path.getsize(csvp) == 0:
        return
    df = pd.read_csv(csvp)
    if df.empty: return
    grp = (df.groupby(["context_hours","name"], as_index=False)["cpu_total_us"].mean())
    # keep top K ops globally so the plot is readable
    top = (df.groupby("name")["cpu_total_us"].mean().sort_values(ascending=False).head(8).index)
    grp = grp[grp["name"].isin(top)]
    if grp.empty: return
    # pivot to stacked bar: context_days vs sum(ms) per op
    grp["context_days"] = (grp["context_hours"]//24).astype(int)
    pv = grp.pivot_table(index="context_days", columns="name", values="cpu_total_us", aggfunc="mean").fillna(0)/1000.0
    pv = pv.sort_index()
    tag = model_tag(model_name); city_tag = city.replace(" ","")
    md = os.path.join(out_dir, tag); os.makedirs(md, exist_ok=True)
    ax = pv.plot(kind="bar", stacked=True, figsize=(10,5))
    ax.set_xlabel("Context length (days)")     # units
    ax.set_ylabel("CPU total (ms)")            # units
    ax.set_title(f"{city} - Per-op CPU total by context - {tag}")
    ax.grid(axis="y")
    out = os.path.join(md, f"{city_tag}_api_cpu_by_context.png")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    ggn = load_series(DATA["ggn"]); pna = load_series(DATA["pna"])

    for mdl in MODELS:
        pipe = BaseChronosPipeline.from_pretrained(mdl, device_map="cpu", dtype=torch.float32)
        for days in tqdm(CONTEXT_DAYS_SWEEP, desc=f"{mdl} context sweep", unit="cfg"):
            ctx_h = int(days)*24
            for city, ser in (("Gurgaon", ggn), ("Patna", pna)):
                out_dir = OUT_ROOT
                csvp = api_csv_path(out_dir, mdl, city)
                ensure_header(csvp)
                vals = ser.values
                start = ctx_h
                # iterate a few windows just to sample ops
                for it, t in enumerate(range(start, len(vals)-24, ROLLING_STEP_HOURS)):
                    if MAX_ITERS is not None and it >= MAX_ITERS:
                        break
                    hist = torch.tensor(vals[t-ctx_h:t], dtype=torch.float32)
                    per_ops = one_infer_with_prof(pipe, hist, 24)
                    meta = {
                        "ts_unix": time.time(),
                        "model": mdl, "city": city,
                        "context_hours": ctx_h, "horizon_hours": 24,
                        "iter_idx": it
                    }
                    if per_ops:
                        append_api_rows(csvp, meta, per_ops)

        # quick plots for API breakdowns (overall + per-context)
        for city in ("Gurgaon","Patna"):
            plot_api_top(OUT_ROOT, mdl, city, title_suffix="overall")
            plot_api_cpu_by_context(OUT_ROOT, mdl, city)

    print(f"API-only outputs saved under: {OUT_ROOT}")

if __name__ == "__main__":
    main()
