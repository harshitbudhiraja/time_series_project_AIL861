# run as: SUDO_PERF=1 python main.py
# Unified OOP framework: latency + hardware counters + FLOPs, all logged to CSV.
# Merged: plotting is built-in; writes per-model/per-city CSVs and plots.
# NEW:
#   - All plots have explicit UNITS on axes
#   - Profiler per-API (per-op) breakdown captured on sampled iterations
#     and appended to   ../outputs/perflogs/<model-tag>/api_breakdown_<City>.csv
#   - We log both wall-clock latency and profiler CPU totals
#   - A short "report preamble" prints tunable knobs and their meaning
#   - Plots added for everything we log: profiler totals/latencies, memory (RSS), raw perf counters
#   - Robust perf column resolution (handles 'cpu_core/cycles:u/' vs 'cycles' vs 'cycles:u')
#   - FIX: all “vs Context” plots aggregate by context to avoid zig-zag / back-and-forth lines
#   - QUALITY: RMSE & MAE added per-iteration (vs ground-truth window) + aggregated plots
#     (NOTE: MAPE/sMAPE intentionally commented out per request)

import os, time, subprocess, shutil, tempfile, logging, logging.handlers
import numpy as np
import pandas as pd
import torch, psutil
from tqdm.auto import tqdm
from chronos import BaseChronosPipeline
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

# ----------------------- Logging utils -----------------------
def build_logger(out_dir: str,
                 name: str = "chronos-perf",
                 level: str | int = None) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "chronos_perf.log")

    # Resolve level from env if provided
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear old handlers if reloading in notebooks
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logger initialized at {logging.getLevelName(level)}. File: {log_path}")
    return logger


class ChronosPerfLogger:
    def __init__(self,
                 models,
                 data_paths,
                 out_dir="../outputs/perflogs",
                 device="cpu",
                 dtype=torch.float32,
                 use_perf=True,
                 use_profiler=True,
                 max_iters=32,
                 rolling_step_hours=24,
                 perf_sample_every=16,
                 prof_sample_every=16,
                 log_level=None):

        self.models = models
        self.data_paths = data_paths
        self.device = device
        self.dtype = dtype
        self.out_dir = out_dir
        self.use_perf = use_perf
        self.use_profiler = use_profiler
        self.max_iters = max_iters
        self.step = rolling_step_hours
        self.perf_sample_every = perf_sample_every
        self.prof_sample_every = prof_sample_every
        self.rows = []
        self.log_level = log_level
        os.makedirs(out_dir, exist_ok=True)

        self.log = build_logger(out_dir, level=log_level)

        # quick environment summary
        self.log.info(f"Config: device={device}, dtype={dtype}, "
                      f"max_iters={max_iters}, step_hours={rolling_step_hours}, "
                      f"perf={use_perf} (every {perf_sample_every}), "
                      f"profiler={use_profiler} (every {prof_sample_every})")
        self.log.info(f"Models: {models}")
        self.log.info(f"Data: {data_paths}")

    def _model_tag(self, model_name: str) -> str:
        # folder-friendly tag, mirror earlier convention
        return model_name.split("/")[-1].replace("chronos-", "").replace("t5-", "t5-")

    # ----------------------- Tiny helpers -----------------------
    def _api_csv_path(self, model_name: str, city: str) -> str:
        tag = self._model_tag(model_name)
        city_tag = city.replace(" ", "")
        model_dir = os.path.join(self.out_dir, tag)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, f"api_breakdown_{city_tag}.csv")

    def _ensure_api_csv_header(self, model_name: str, city: str):
        """Make sure the per-API CSV exists with a header even if a sampled iteration emits 0 rows."""
        path = self._api_csv_path(model_name, city)
        cols = [
            "ts_unix", "model", "city", "context_hours", "horizon_hours", "iter_idx",
            "name", "calls",
            "self_cpu_total_us", "cpu_total_us", "cpu_time_avg_us",
            "flops_total"
        ]
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            import csv
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=cols).writeheader()

    def _append_api_breakdown(self, model_name, city, meta_row, per_ops):
        """
        per_ops: list of dicts with keys:
          name, self_cpu_total_us, cpu_total_us, cpu_time_avg_us, calls, flops_total
        meta_row: dict subset with ts, model, city, context_hours, horizon_hours, iter_idx
        """
        path = self._api_csv_path(model_name, city)
        cols = [
            "ts_unix", "model", "city", "context_hours", "horizon_hours", "iter_idx",
            "name", "calls",
            "self_cpu_total_us", "cpu_total_us", "cpu_time_avg_us",
            "flops_total"
        ]
        need_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
        import csv
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if need_header:
                w.writeheader()
            for op in per_ops:
                row = {
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
                }
                w.writerow(row)

    # ----------------------- Model inspection -----------------------
    def inspect_models(self):
        """Print param and memory stats for each model."""
        for model_name in self.models:
            print(f"\n===== {model_name} =====")
            pipe = BaseChronosPipeline.from_pretrained(model_name, device_map=self.device, dtype=self.dtype)
            m = pipe.model if hasattr(pipe, "model") else pipe  # fallback
            total_params = sum(p.numel() for p in m.parameters())
            trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            param_mem = total_params * torch.finfo(self.dtype).bits / 8 / 1e6  # MB
            print(f"Total params      : {total_params:,}")
            print(f"Trainable params  : {trainable_params:,}")
            print(f"Frozen params     : {frozen_params:,}")
            print(f"Memory (params)   : {param_mem:.2f} MB ({self.dtype})")
            # quick dummy I/O check
            dummy_in = torch.randn(1, 24, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                out = pipe.predict_quantiles(context=dummy_in, prediction_length=24, quantile_levels=[0.5])
            out_shape = tuple(o.shape for o in out) if isinstance(out, (tuple, list)) else out.shape
            print(f"Input shape       : {tuple(dummy_in.shape)}")
            print(f"Output shape      : {out_shape}")
            if hasattr(m, "device"):
                print(f"Model device      : {m.device}")
            print("========================")

    # ----------------------- Loaders -----------------------
    def load_series(self, path):
        self.log.debug(f"Loading series: {path}")
        df = pd.read_csv(path, usecols=[0, 4], parse_dates=[0])
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s = pd.Series(y.values, index=ts).sort_index()
        s = s.asfreq("H").interpolate(limit_direction="both")
        self.log.info(f"Loaded {path}: {len(s)} hourly points, "
                      f"{s.index.min()} -> {s.index.max()}")
        return s.astype(np.float32)

    # ----------------------- Quality metrics (RMSE, MAE only) -----------------------
    @staticmethod
    def _error_metrics_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute RMSE and MAE on a single forecast window.
        Returns a dict with {'rmse': float, 'mae': float}.
        NOTE: MAPE/sMAPE intentionally omitted (kept as commented placeholders below).
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        n = y_true.size
        if n == 0 or y_pred.size != n:
            return {"rmse": np.nan, "mae": np.nan}

        err = y_pred - y_true
        mse = np.mean(err ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(err))

        # mape_pct = np.mean(np.abs(err) / (np.abs(y_true) + 1e-8)) * 100.0   # <-- commented out by request
        # smape_pct = np.mean(2.0*np.abs(err)/(np.abs(y_true)+np.abs(y_pred)+1e-8)) * 100.0  # <-- commented out

        return {"rmse": float(rmse), "mae": float(mae)}

    # ----------------------- Core Ops -----------------------
    @torch.no_grad()
    def forecast_once(self, pipe, hist, horizon_hours):
        _, mean = pipe.predict_quantiles(
            inputs= hist,
            prediction_length=horizon_hours,
            quantile_levels=[0.5]
        )
        return mean

    def timed_infer_with_optional_flops(self, pipe, hist, horizon_hours, do_profile):
        """
        If do_profile:
          - run PyTorch profiler with FLOPs
          - return flops_total, prof_latency_s (same window), prof_cpu_total_s (sum of CPU time over ops)
          - also return per-op breakdown for external CSV
        """
        if not do_profile:
            t0 = time.perf_counter()
            mean = self.forecast_once(pipe, hist, horizon_hours)
            t1 = time.perf_counter()
            return mean, (t1 - t0), {}

        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            t0 = time.perf_counter()
            with record_function("chronos_infer"):
                mean = self.forecast_once(pipe, hist, horizon_hours)
            t1 = time.perf_counter()

        # Summaries
        key_avg = prof.key_averages()
        total_flops = sum(getattr(evt, "flops", 0) or 0 for evt in key_avg)
        total_cpu_us = sum((evt.cpu_time_total or 0.0) for evt in key_avg)  # microseconds
        prof_cpu_total_s = total_cpu_us / 1e6

        # Per-op breakdown (minimal but useful)
        per_ops = []
        for evt in key_avg:
            # Torch versions differ: some don't expose cpu_time_avg.
            calls = int(evt.count or 0)
            cpu_total_us = float(evt.cpu_time_total or 0.0)
            self_cpu_total_us = float(evt.self_cpu_time_total or 0.0)
            cpu_time_avg_us = cpu_total_us / calls if calls > 0 else 0.0

            per_ops.append({
                "name": evt.key,                     # op name
                "calls": calls,
                "self_cpu_total_us": self_cpu_total_us,
                "cpu_total_us": cpu_total_us,
                "cpu_time_avg_us": cpu_time_avg_us,  # computed average
                "flops_total": int(getattr(evt, "flops", 0) or 0),
            })

        return mean, (t1 - t0), {
            "flops_total": total_flops,
            "prof_latency_s": (t1 - t0),
            "prof_cpu_total_s": prof_cpu_total_s,
            "per_ops": per_ops,
        }

    # ----------------------- Perf Integration -----------------------
    def have_perf(self):
        ok = shutil.which("perf") is not None
        if not ok and self.use_perf:
            self.log.warning("perf not found in PATH, hardware counters disabled.")
        return ok

    def perf_one_call(self, model_id, hist_np, horizon):
        """Run one real inference in a child process wrapped by perf stat, then parse counters."""
        if not (self.use_perf and self.have_perf()):
            return {}

        # Locate helper alongside this file
        helper_path = os.path.join(os.path.dirname(__file__), "perf_infer_helper.py")
        if not os.path.exists(helper_path):
            self.log.warning(f"perf helper not found at {helper_path}; skipping perf.")
            return {}

        py = shutil.which("python") or "python"

        # Hardware counters (user-space only). Fallback to software counters.
        hw_events = "cycles:u,instructions:u,cache-references:u,cache-misses:u,branches:u,branch-misses:u"
        sw_events = "task-clock,context-switches,cpu-migrations,page-faults"
        sudo = "sudo -E " if os.environ.get("SUDO_PERF") == "1" else ""

        data = {}
        # Keep the temp dir alive while perf runs so ctx.npy is readable.
        with tempfile.TemporaryDirectory() as td:
            npy_path = os.path.join(td, "ctx.npy")
            np.save(npy_path, hist_np)

            env = os.environ.copy()
            env["PERF_MODEL"] = model_id
            env["PERF_HORIZON"] = str(horizon)
            env["PERF_CTX_PATH"] = npy_path

            def run_perf(evlist):
                cmd = f'{sudo}perf stat -x, -e {evlist} {py} "{helper_path}"'
                self.log.debug(f"perf exec: {cmd}")
                return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

            proc = run_perf(hw_events)
            if proc.returncode != 0:
                self.log.warning(f"perf HW counters blocked (rc={proc.returncode}). Falling back to software events.")
                proc = run_perf(sw_events)

            # Parse perf CSV-like stderr
            for line in proc.stderr.splitlines():
                parts = line.split(",")
                if len(parts) >= 3:
                    val, _, ev = parts[0], parts[1], parts[2]
                    try:
                        data[ev] = int(val.replace(",", ""))
                    except:
                        try:
                            data[ev] = float(val.replace(",", ""))
                        except:
                            pass

        # Derived (plotter resolves name variants)
        if "cycles" in data and "instructions" in data:
            data["ipc"] = data["instructions"] / max(data["cycles"], 1)
        if "cache-references" in data and "cache-misses" in data:
            data["cache_miss_rate"] = data["cache-misses"] / max(data["cache-references"], 1)
        if "branches" in data and "branch-misses" in data:
            data["branch_miss_rate"] = data["branch-misses"] / max(data["branches"], 1)
        self.log.debug(f"perf counters parsed: {data}")
        return data

    # ----------------------- Run Config -----------------------
    def run_single_cfg(self, pipe, city, series, model_name, context_hours, horizon_hours):
        vals = series.values
        n = len(vals)
        proc = psutil.Process(os.getpid())
        start = context_hours

        self.log.info(f"[RUN] model={model_name} city={city} ctx={context_hours}h hor={horizon_hours}h")

        for it, t in enumerate(range(start, n - horizon_hours, self.step)):
            if self.max_iters is not None and it >= self.max_iters:
                self.log.debug("Reached max_iters cap; stopping this config.")
                break

            hist = torch.tensor(vals[t - context_hours:t], dtype=self.dtype)

            rss_before = proc.memory_info().rss
            do_prof = self.use_profiler and (it % self.prof_sample_every == 0)

            # Ensure API CSV header exists for sampled iterations (even if no ops emitted)
            if do_prof:
                self._ensure_api_csv_header(model_name, city)

            mean, latency, prof_stats = self.timed_infer_with_optional_flops(
                pipe, hist, horizon_hours, do_prof
            )
            rss_after = proc.memory_info().rss
            delta_rss = rss_after - rss_before

            # ---- QUALITY: RMSE & MAE vs ground-truth future window
            y_true = vals[t:t + horizon_hours]                       # numpy (H,)
            y_pred = mean.detach().cpu().numpy().reshape(-1)         # numpy (H,)
            qm = self._error_metrics_rmse_mae(y_true, y_pred)        # {'rmse', 'mae'}

            row = {
                "ts_unix": time.time(),
                "city": city,
                "model": model_name,
                "context_hours": context_hours,
                "horizon_hours": horizon_hours,
                "iter_idx": it,
                "latency_s": latency,  # wall-clock (seconds)

                # ---- quality metrics (units: in target units)
                "rmse": qm["rmse"],
                "mae": qm["mae"],
                # "mape_pct": ...    # intentionally commented out
                # "smape_pct": ...   # intentionally commented out

                # profiler additions come below (may be None on non-sampled iters)
                "prof_latency_s": None,     # seconds
                "prof_cpu_total_s": None,   # seconds (sum of CPU time over ops)
                "proc_rss_bytes_before": rss_before,
                "proc_rss_bytes_after": rss_after,
                "proc_rss_delta_bytes": delta_rss,
            }

            # FLOPs and derived GFLOP/s + profiler totals
            if prof_stats:
                row["prof_latency_s"]   = prof_stats.get("prof_latency_s")
                row["prof_cpu_total_s"] = prof_stats.get("prof_cpu_total_s")
                if "flops_total" in prof_stats and latency > 0:
                    row["flops_total"] = prof_stats["flops_total"]
                    row["gflops_per_s"] = prof_stats["flops_total"] / latency / 1e9 # GFLOP/s (Throughput)
                else:
                    row["flops_total"] = None
                    row["gflops_per_s"] = None

                # Per-op API breakdown on sampled iterations
                per_ops = prof_stats.get("per_ops") or []
                if per_ops:
                    self._append_api_breakdown(model_name, city, row, per_ops)

                self.log.debug(f"[prof] it={it} FLOPs={row.get('flops_total')} "
                               f"GFLOP/s={row.get('gflops_per_s')} "
                               f"CPU_total={row.get('prof_cpu_total_s')}s")

            # Perf counters
            if self.use_perf and (it % self.perf_sample_every == 0):
                perf_stats = self.perf_one_call(model_name, hist.cpu().numpy(), horizon_hours)
                row.update(perf_stats)

            # Keep in-memory rows for end-of-run CSV + plots
            self.rows.append(row)

            # light INFO every few steps (units inline)
            if it % 8 == 0:
                self.log.info(
                    f"it={it:04d} latency={latency:.4f}s "
                    f"rssΔ={delta_rss/1e6:.2f}MB "
                    f"RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
                    f"{'(prof)' if do_prof else ''} "
                    f"{'(perf)' if (self.use_perf and it % self.perf_sample_every == 0) else ''}"
                )

    # ----------------------- Top-level Run -----------------------
    def run_all(self, context_days_sweep, horizon_hours_sweep):
        """Run the full sweep (contexts @ fixed horizon + horizons @ fixed context) for all models & cities."""
        ggn = self.load_series(self.data_paths["ggn"])
        pna = self.load_series(self.data_paths["pna"])

        for model_name in self.models:
            self.log.info(f"Loading model: {model_name}")
            pipe = BaseChronosPipeline.from_pretrained(model_name, device_map=self.device, dtype=self.dtype)

        # 1) Vary context (fixed 24h horizon)
            for days in tqdm(context_days_sweep, desc=f"{model_name} context sweep", unit="cfg"):
                ctx_h = int(days) * 24
                for city, ser in (("Gurgaon", ggn), ("Patna", pna)):
                    self.run_single_cfg(pipe, city, ser, model_name, ctx_h, 24)

        # 2) Vary horizon (fixed 10d context)
            ctx_h_fixed = 10 * 24
            for horizon in tqdm(horizon_hours_sweep, desc=f"{model_name} horizon sweep", unit="cfg"):
                for city, ser in (("Gurgaon", ggn), ("Patna", pna)):
                    self.run_single_cfg(pipe, city, ser, model_name, ctx_h_fixed, int(horizon))

        self.save_results()
        self.plot_summary()

    # ----------------------- Save + Plot -----------------------
    def save_results(self):
        df = pd.DataFrame(self.rows)

        # 0) Combined (all models, all cities)
        combined_path = os.path.join(self.out_dir, "chronos_perf_compute_log.csv")
        df.to_csv(combined_path, index=False)
        self.log.info(f"Wrote combined CSV: {combined_path}  ({len(df)} rows)")

        # 1) Per-model folders
        for model_name, g_model in df.groupby("model"):
            tag = self._model_tag(model_name)
            model_dir = os.path.join(self.out_dir, tag)
            os.makedirs(model_dir, exist_ok=True)

            # per-model CSV
            path_model = os.path.join(model_dir, "chronos_perf.csv")
            g_model.to_csv(path_model, index=False)
            self.log.info(f"[{tag}] Wrote per-model CSV: {path_model}  ({len(g_model)} rows)")

            # 2) per-model-per-city CSVs
            for city, g_city in g_model.groupby("city"):
                city_tag = city.replace(" ", "")
                path_mc = os.path.join(model_dir, f"chronos_perf_{city_tag}.csv")
                g_city.to_csv(path_mc, index=False)
                self.log.info(f"[{tag}] Wrote per-model-per-city CSV: {path_mc}  ({len(g_city)} rows)")

    # ---- Helper for robust perf column lookups (kernel/PMU names vary)
    @staticmethod
    def _first_present(columns, candidates):
        for c in candidates:
            if c in columns:
                return c
        return None

    # Modifiable for api-nums
    def _plot_api_top10(self, model_dir, city_tag, model_name, city):
        """Read api_breakdown CSV (if exists) and save bar chart of top-20 ops by avg cpu_total (ms)."""
        api_csv = self._api_csv_path(model_name, city)
        if not os.path.exists(api_csv) or os.path.getsize(api_csv) == 0:
            return
        try:
            g = pd.read_csv(api_csv)
            if g.empty or "cpu_total_us" not in g.columns or "name" not in g.columns:
                return
            agg = (g.groupby("name", as_index=False)["cpu_total_us"]
                    .mean()
                    .sort_values("cpu_total_us", ascending=False)
                    .head(20))
            if len(agg) == 0:
                return
            plt.figure(figsize=(9.5, 6))  # a bit taller for top-20
            plt.barh(agg["name"][::-1], (agg["cpu_total_us"]/1000.0)[::-1])  # ms
            plt.xlabel("Average CPU total (ms)")     # units
            plt.ylabel("Operator / API")
            plt.title(f"{city} - Top 20 ops by CPU time")
            plt.tight_layout()
            out = os.path.join(model_dir, f"{city_tag}_api_top20_cpu_ms.png")
            plt.savefig(out, bbox_inches="tight"); plt.close()
            self.log.info(f"Saved plot: {out}")
        except Exception as e:
            self.log.debug(f"API plot skipped: {repr(e)}")

    def plot_summary(self):
        try:
            df = pd.DataFrame(self.rows)
            if df.empty:
                self.log.warning("No rows to plot.")
                return

            for model_name, g_all in df.groupby("model"):
                tag = self._model_tag(model_name)
                model_dir = os.path.join(self.out_dir, tag)
                os.makedirs(model_dir, exist_ok=True)

                # perf column name variants (resolve per group to handle sampled columns)
                CYCLES_CANDS = ["cpu_core/cycles:u/", "cycles:u", "cycles"]
                INSTR_CANDS  = ["cpu_core/instructions:u/", "instructions:u", "instructions"]
                MISS_CANDS   = ["cpu_core/cache-misses:u/", "cache-misses:u", "cache-misses"]
                REFS_CANDS   = ["cpu_core/cache-references:u/", "cache-references:u", "cache-references"]
                BR_CANDS     = ["cpu_core/branches:u/", "branches:u", "branches"]
                BRM_CANDS    = ["cpu_core/branch-misses:u/", "branch-misses:u", "branch-misses"]

                cyc_col  = self._first_present(df.columns, CYCLES_CANDS)
                ins_col  = self._first_present(df.columns, INSTR_CANDS)
                miss_col = self._first_present(df.columns, MISS_CANDS)
                refs_col = self._first_present(df.columns, REFS_CANDS)
                br_col   = self._first_present(df.columns, BR_CANDS)
                brm_col  = self._first_present(df.columns, BRM_CANDS)

                for city, g in g_all.groupby("city"):
                    city_tag = city.replace(" ", "")

                    # ---------- Aggregated (no zig-zag) ----------
                    d1 = g[g.horizon_hours == 24]
                    if len(d1):
                        d1m = (d1.groupby("context_hours", as_index=False)["latency_s"]
                               .mean()
                               .sort_values("context_hours"))
                        plt.figure()
                        plt.plot((d1m.context_hours/24).astype(int), d1m.latency_s, marker="o")
                        plt.title(f"{city} - Latency vs Context - {tag}")
                        plt.xlabel("Context length (days)")      # units
                        plt.ylabel("Latency (seconds)")           # units
                        plt.grid(True)
                        out = os.path.join(model_dir, f"{city_tag}_latency_vs_context.png")
                        plt.savefig(out, bbox_inches="tight"); plt.close()

                    if "prof_latency_s" in g.columns and len(d1) and d1["prof_latency_s"].notna().any():
                        d1p = (d1.dropna(subset=["prof_latency_s"])
                                 .groupby("context_hours", as_index=False)["prof_latency_s"]
                                 .mean()
                                 .sort_values("context_hours"))
                        if len(d1p):
                            plt.figure()
                            plt.plot((d1p.context_hours/24).astype(int), d1p.prof_latency_s, marker="o")
                            plt.title(f"{city} - Profiler latency vs Context - {tag}")
                            plt.xlabel("Context length (days)")
                            plt.ylabel("Profiler latency (seconds)")
                            plt.grid(True)
                            out = os.path.join(model_dir, f"{city_tag}_prof_latency_vs_context.png")
                            plt.savefig(out, bbox_inches="tight"); plt.close()

                    if "prof_cpu_total_s" in g.columns and len(d1) and d1["prof_cpu_total_s"].notna().any():
                        d1c = (d1.dropna(subset=["prof_cpu_total_s"])
                                 .groupby("context_hours", as_index=False)["prof_cpu_total_s"]
                                 .mean()
                                 .sort_values("context_hours"))
                        if len(d1c):
                            plt.figure()
                            plt.plot((d1c.context_hours/24).astype(int), d1c.prof_cpu_total_s, marker="o")
                            plt.title(f"{city} - Profiler CPU total vs Context - {tag}")
                            plt.xlabel("Context length (days)")
                            plt.ylabel("Profiler CPU total (seconds)")
                            plt.grid(True)
                            out = os.path.join(model_dir, f"{city_tag}_prof_cputotal_vs_context.png")
                            plt.savefig(out, bbox_inches="tight"); plt.close()

                    if len(d1):
                        d1rssA = (d1.groupby("context_hours", as_index=False)["proc_rss_bytes_after"]
                                   .mean().sort_values("context_hours"))
                        plt.figure()
                        plt.plot((d1rssA.context_hours/24).astype(int),
                                 d1rssA.proc_rss_bytes_after/1e6, marker="o")
                        plt.title(f"{city} - Process RSS (after) vs Context - {tag}")
                        plt.xlabel("Context length (days)")
                        plt.ylabel("Process RSS after (MB)")
                        plt.grid(True)
                        out = os.path.join(model_dir, f"{city_tag}_rss_after_vs_context.png")
                        plt.savefig(out, bbox_inches="tight"); plt.close()

                        d1rssD = (d1.groupby("context_hours", as_index=False)["proc_rss_delta_bytes"]
                                   .mean().sort_values("context_hours"))
                        plt.figure()
                        plt.plot((d1rssD.context_hours/24).astype(int),
                                 d1rssD.proc_rss_delta_bytes/1e6, marker="o")
                        plt.title(f"{city} - Process RSS Δ vs Context - {tag}")
                        plt.xlabel("Context length (days)")
                        plt.ylabel("Process RSS delta (MB)")
                        plt.grid(True)
                        out = os.path.join(model_dir, f"{city_tag}_rss_delta_vs_context.png")
                        plt.savefig(out, bbox_inches="tight"); plt.close()

                    d2 = g[g.context_hours == 10 * 24]
                    if len(d2):
                        d2m = (d2.groupby("horizon_hours", as_index=False)["latency_s"].mean()
                                .sort_values("horizon_hours"))
                        plt.figure()
                        plt.plot(d2m.horizon_hours, d2m.latency_s, marker="o")
                        plt.title(f"{city} - Latency vs Horizon - {tag}")
                        plt.xlabel("Forecast horizon (hours)")
                        plt.ylabel("Latency (seconds)")
                        plt.grid(True)
                        out = os.path.join(model_dir, f"{city_tag}_latency_vs_horizon.png")
                        plt.savefig(out, bbox_inches="tight"); plt.close()

                        if "prof_latency_s" in g.columns and g["prof_latency_s"].notna().any():
                            d2lp = (d2.dropna(subset=["prof_latency_s"])
                                     .groupby("horizon_hours", as_index=False)["prof_latency_s"].mean()
                                     .sort_values("horizon_hours"))
                            if len(d2lp):
                                plt.figure()
                                plt.plot(d2lp.horizon_hours, d2lp.prof_latency_s, marker="o")
                                plt.title(f"{city} - Profiler latency vs Horizon - {tag}")
                                plt.xlabel("Forecast horizon (hours)")
                                plt.ylabel("Profiler latency (seconds)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_prof_latency_vs_horizon.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                        if "prof_cpu_total_s" in g.columns and g["prof_cpu_total_s"].notna().any():
                            d2cp = (d2.dropna(subset=["prof_cpu_total_s"])
                                     .groupby("horizon_hours", as_index=False)["prof_cpu_total_s"].mean()
                                     .sort_values("horizon_hours"))
                            if len(d2cp):
                                plt.figure()
                                plt.plot(d2cp.horizon_hours, d2cp.prof_cpu_total_s, marker="o")
                                plt.title(f"{city} - Profiler CPU total vs Horizon - {tag}")
                                plt.xlabel("Forecast horizon (hours)")
                                plt.ylabel("Profiler CPU total (seconds)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_prof_cputotal_vs_horizon.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                        d2m_mem = (d2.groupby("horizon_hours", as_index=False)["proc_rss_bytes_after"].mean()
                                     .sort_values("horizon_hours"))
                        plt.figure()
                        plt.plot(d2m_mem.horizon_hours, d2m_mem.proc_rss_bytes_after / 1e6, marker="o")
                        plt.title(f"{city} - Process RSS (after) vs Horizon - {tag}")
                        plt.xlabel("Forecast horizon (hours)")
                        plt.ylabel("Process RSS after (MB)")
                        plt.grid(True)
                        out = os.path.join(model_dir, f"{city_tag}_rss_after_vs_horizon.png")
                        plt.savefig(out, bbox_inches="tight"); plt.close()

                    # GFLOP/s vs Latency
                    if "gflops_per_s" in g.columns and g["gflops_per_s"].notna().any():
                        dfl = g.dropna(subset=["gflops_per_s"])
                        if len(dfl):
                            plt.figure()
                            plt.scatter(dfl.latency_s, dfl.gflops_per_s, s=16)
                            plt.title(f"{city} - GFLOP/s vs Latency - {tag}")
                            plt.xlabel("Latency (seconds)")
                            plt.ylabel("Throughput (GFLOP/s)")
                            plt.grid(True)
                            out = os.path.join(model_dir, f"{city_tag}_gflops_vs_latency.png")
                            plt.savefig(out, bbox_inches="tight"); plt.close()

                    # ---------- QUALITY plots (RMSE/MAE only; aggregated) ----------
                    # vs CONTEXT at fixed 24h horizon
                    if len(d1):
                        if "rmse" in d1.columns and d1["rmse"].notna().any():
                            dmc = (d1.dropna(subset=["rmse"])
                                     .groupby("context_hours", as_index=False)["rmse"]
                                     .mean()
                                     .sort_values("context_hours"))
                            if len(dmc):
                                plt.figure()
                                plt.plot((dmc.context_hours/24).astype(int), dmc["rmse"], marker="o")
                                plt.title(f"{city} - RMSE vs Context - {tag}")
                                plt.xlabel("Context length (days)")
                                plt.ylabel("RMSE (target units)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_rmse_vs_context.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                        if "mae" in d1.columns and d1["mae"].notna().any():
                            dmc = (d1.dropna(subset=["mae"])
                                     .groupby("context_hours", as_index=False)["mae"]
                                     .mean()
                                     .sort_values("context_hours"))
                            if len(dmc):
                                plt.figure()
                                plt.plot((dmc.context_hours/24).astype(int), dmc["mae"], marker="o")
                                plt.title(f"{city} - MAE vs Context - {tag}")
                                plt.xlabel("Context length (days)")
                                plt.ylabel("MAE (target units)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_mae_vs_context.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                    # vs HORIZON at fixed 10d context
                    dH = g[g.context_hours == 10 * 24]
                    if len(dH):
                        if "rmse" in dH.columns and dH["rmse"].notna().any():
                            dmh = (dH.dropna(subset=["rmse"])
                                     .groupby("horizon_hours", as_index=False)["rmse"]
                                     .mean()
                                     .sort_values("horizon_hours"))
                            if len(dmh):
                                plt.figure()
                                plt.plot(dmh.horizon_hours, dmh["rmse"], marker="o")
                                plt.title(f"{city} - RMSE vs Horizon - {tag}")
                                plt.xlabel("Forecast horizon (hours)")
                                plt.ylabel("RMSE (target units)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_rmse_vs_horizon.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                        if "mae" in dH.columns and dH["mae"].notna().any():
                            dmh = (dH.dropna(subset=["mae"])
                                     .groupby("horizon_hours", as_index=False)["mae"]
                                     .mean()
                                     .sort_values("horizon_hours"))
                            if len(dmh):
                                plt.figure()
                                plt.plot(dmh.horizon_hours, dmh["mae"], marker="o")
                                plt.title(f"{city} - MAE vs Horizon - {tag}")
                                plt.xlabel("Forecast horizon (hours)")
                                plt.ylabel("MAE (target units)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_mae_vs_horizon.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                    # Perf: IPC & miss/branch rates vs context (24h), aggregated
                    d24 = d1
                    if d24 is not None and len(d24):
                        if ins_col in d24.columns and cyc_col in d24.columns:
                            d_ipc = (d24.groupby("context_hours", as_index=False)[[ins_col, cyc_col]].mean()
                                     .sort_values("context_hours"))
                            d_ipc["ipc"] = d_ipc[ins_col] / d_ipc[cyc_col].replace(0, np.nan)
                            if d_ipc["ipc"].notna().any():
                                plt.figure()
                                plt.plot((d_ipc.context_hours/24).astype(int), d_ipc.ipc, marker="o")
                                plt.title(f"{city} - IPC vs Context - {tag}")
                                plt.xlabel("Context length (days)")
                                plt.ylabel("IPC (instructions/cycle)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_ipc_vs_context.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                        if miss_col in d24.columns and refs_col in d24.columns:
                            d_mr = (d24.groupby("context_hours", as_index=False)[[miss_col, refs_col]].mean()
                                     .sort_values("context_hours"))
                            d_mr["miss_rate"] = d_mr[miss_col] / d_mr[refs_col].replace(0, np.nan)
                            if d_mr["miss_rate"].notna().any():
                                plt.figure()
                                plt.plot((d_mr.context_hours/24).astype(int), d_mr.miss_rate * 100.0, marker="o")
                                plt.title(f"{city} - Miss rate vs Context - {tag}")
                                plt.xlabel("Context length (days)")
                                plt.ylabel("Cache miss rate (%)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_missrate_vs_context.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                        if br_col in d24.columns and brm_col in d24.columns:
                            d_br = (d24.groupby("context_hours", as_index=False)[[br_col, brm_col]].mean()
                                     .sort_values("context_hours"))
                            d_br["br_miss_rate"] = d_br[brm_col] / d_br[br_col].replace(0, np.nan)
                            if d_br["br_miss_rate"].notna().any():
                                plt.figure()
                                plt.plot((d_br.context_hours/24).astype(int), d_br.br_miss_rate*100.0, marker="o")
                                plt.title(f"{city} - Branch miss rate vs Context - {tag}")
                                plt.xlabel("Context length (days)")
                                plt.ylabel("Branch miss rate (%)")
                                plt.grid(True)
                                out = os.path.join(model_dir, f"{city_tag}_branch_missrate_vs_context.png")
                                plt.savefig(out, bbox_inches="tight"); plt.close()

                        # Raw counters bars at 24h (cycles & instructions)
                        if cyc_col in d24.columns and ins_col in d24.columns:
                            dCI = (d24.groupby("context_hours", as_index=False)[[cyc_col, ins_col]].mean()
                                    .sort_values("context_hours"))
                            x = np.arange(len(dCI))
                            w = 0.45
                            plt.figure()
                            plt.bar(x - w/2, dCI[cyc_col]/1e9, width=w, label="cycles (G)")
                            plt.bar(x + w/2, dCI[ins_col]/1e9, width=w, label="instructions (G)")
                            plt.xticks(x, (dCI.context_hours/24).astype(int))
                            plt.xlabel("Context length (days)")
                            plt.ylabel("Count (billions)")
                            plt.title(f"{city} - cycles/instructions vs Context - {tag}")
                            plt.legend(); plt.grid(axis="y")
                            out = os.path.join(model_dir, f"{city_tag}_cycles_instr_vs_context.png")
                            plt.savefig(out, bbox_inches="tight"); plt.close()

                    # Per-API TOP-20 (reads api_breakdown CSV)
                    self._plot_api_top10(model_dir, city_tag, model_name, city)

        except Exception as e:
            self.log.warning(f"Plotting skipped: {repr(e)}")


# ----------------------- Run-config preamble (human-friendly) -----------------------
def print_run_preamble(models, data_paths, context_days_sweep, horizon_hours_sweep,
                       rolling_step_hours, perf_every, prof_every, max_iters, device, dtype, out_dir):
    """Prints a short report describing what will run and what each knob means."""
    knobs = f"""
=== Chronos Perf Run: Configuration Preamble ===

Models: {models}
Data paths:
  - Gurgaon: {data_paths.get('ggn')}
  - Patna  : {data_paths.get('pna')}

Output directory:
  {out_dir}

Tunable knobs (what they mean):
  CONTEXT_DAYS_SWEEP = {context_days_sweep}
    -> Sweep the *history length* (context) in DAYS (×24 = hours).

  HORIZON_HOURS_SWEEP = {horizon_hours_sweep}
    -> With a fixed 10-day context, sweep the *forecast horizon* in HOURS.

  rolling_step_hours = {rolling_step_hours}
    -> Sliding-window stride across the time series (HOURS).

  max_iters = {max_iters}
    -> Cap on iterations per (model, city, context/horizon) config (None = all).

  perf_sample_every = {perf_every}
    -> Linux 'perf stat' sampling cadence (Nth iteration).

  prof_sample_every = {prof_every}
    -> PyTorch profiler sampling cadence (Nth iteration).

Runtime device / dtype:
  device = {device}
  dtype  = {dtype}

Recorded metrics (per iteration):
  - latency_s            [seconds]
  - prof_latency_s       [seconds]
  - prof_cpu_total_s     [seconds]
  - flops_total          [FLOPs]
  - gflops_per_s         [GFLOP/s]
  - proc_rss_*           [bytes]
  - perf counters (sampled): cycles, instructions, cache-*, branches, branch-misses, task-clock, context-switches, cpu-migrations, page-faults
    * Derived: ipc [instructions/cycle], cache_miss_rate [%], branch_miss_rate [%]
  - quality: RMSE / MAE (target units)
    # (MAPE/sMAPE intentionally commented out)
================================================
"""
    print(knobs)


if __name__ == "__main__":
    # ---------- CONFIG: you can tweak these knobs ----------
    MODELS = ["amazon/chronos-t5-tiny", "amazon/chronos-t5-small", "amazon/chronos-t5-base"]
    DATA = {
        "ggn": "../data/df_ggn_covariates.csv",
        "pna": "../data/df_patna_covariates.csv",
    }

    CONTEXT_DAYS_SWEEP  = [2, 4, 8, 10, 14]
    HORIZON_HOURS_SWEEP = [4, 8, 12, 24, 48]

    env_max_iters = os.environ.get("MAX_ITERS")
    max_iters = int(env_max_iters) if env_max_iters is not None else 16

    out_dir = "../outputs/perflogs"
    device  = "cpu"
    dtype   = torch.float32
    rolling_step_hours = 24
    perf_every = 16     # perf every 16th call
    prof_every = 16     # FLOPs/per-op every 16th call
    log_level = "DEBUG"

    print_run_preamble(
        models=MODELS,
        data_paths=DATA,
        context_days_sweep=CONTEXT_DAYS_SWEEP,
        horizon_hours_sweep=HORIZON_HOURS_SWEEP,
        rolling_step_hours=rolling_step_hours,
        perf_every=perf_every,
        prof_every=prof_every,
        max_iters=max_iters,
        device=device,
        dtype=dtype,
        out_dir=out_dir
    )

    logger = ChronosPerfLogger(
        models=MODELS,
        data_paths=DATA,
        out_dir=out_dir,
        device=device,
        dtype=dtype,
        use_perf=True,
        use_profiler=True,
        max_iters=max_iters,
        rolling_step_hours=rolling_step_hours,
        perf_sample_every=perf_every,
        prof_sample_every=prof_every,
        log_level=log_level
    )

    logger.run_all(CONTEXT_DAYS_SWEEP, HORIZON_HOURS_SWEEP)
