# run as: SUDO_PERF=1 python 003.py
# Unified OOP framework: latency + hardware counters + FLOPs, all logged to CSV.

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
        return model_name.split("/")[-1].replace("chronos-", "").replace("t5-", "t5-")
    
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

    # ----------------------- Core Ops -----------------------
    @torch.no_grad()
    def forecast_once(self, pipe, hist, horizon_hours):
        _, mean = pipe.predict_quantiles(
            context=hist,
            prediction_length=horizon_hours,
            quantile_levels=[0.5]
        )
        return mean

    def timed_infer_with_optional_flops(self, pipe, hist, horizon_hours, do_profile):
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
        total_flops = sum(getattr(evt, "flops", 0) for evt in prof.key_averages())
        return mean, (t1 - t0), {"flops_total": total_flops}

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

        import tempfile, shutil, subprocess, os, numpy as np

        with tempfile.TemporaryDirectory() as td:
            npy_path = os.path.join(td, "ctx.npy")
            np.save(npy_path, hist_np)

            env = os.environ.copy()
            env["PERF_MODEL"] = model_id
            env["PERF_HORIZON"] = str(horizon)      # o/p
            env["PERF_CTX_PATH"] = npy_path         # i/p

            py = shutil.which("python") or "python" # runs bash: which python

            # ---------------- PERF EVENT SETS ----------------
            # Hardware counters (':u' = user-space only)
            #   - cycles: Total CPU clock elapsed, instructions: Total CPU instructions retired        → basic IPC metrics
            #   - cache-{references,misses}  → memory locality
            #   - branches,branch-misses     → control-flow efficiency
            # These stay compatible with perf_event_paranoid=1 (non-root).
            # user space only, compatible with perf_event_paranoid=1
            hw_events = "cycles:u,instructions:u,cache-references:u,cache-misses:u,branches:u,branch-misses:u"

            # Software (kernel-derived) counters:
            #   - task-clock         → total CPU time consumed
            #   - context-switches   → scheduler preemptions
            #   - cpu-migrations     → thread moved across cores
            #   - page-faults        → memory paging activity
            # Used as fallback when HW counters are restricted/unavailable.
            sw_events = "task-clock,context-switches,cpu-migrations,page-faults"

            # allow optional sudo via env if needed
            sudo = "sudo -E " if os.environ.get("SUDO_PERF") == "1" else "" # -E preserves all (existing) env variables. usually just "sudo" omits all env vars.

            # evlist: event list
            def run_perf(evlist):
                cmd = f"{sudo}perf stat -x, -e {evlist} {py} perf_infer_helper.py"
                #   Basically runs: sudo -E perf stat -x, -e cycles:u,instructions:u,cache-misses:u python perf_infer_helper.py
                
                #   sudo -E perf stat -x, -e cycles:u,instructions:u,cache-misses:u
                #
                #   -> Runs Linux 'perf stat' to collect low-level CPU counters.
                #      - 'sudo -E' preserves environment vars while running with privileges.
                #      - '-x,' outputs comma-separated stats for easy parsing.
                #      - '-e ...' selects hardware/software events to monitor # specifies which performance events to monitor

                self.log.debug(f"perf exec: {cmd}")
                return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
                # Run the perf command as a subprocess:
                # - cmd: the full "perf stat ..." command string to execute
                # - shell=True: run via system shell so pipes/wildcards work
                # - capture_output=True: capture both stdout and stderr for later parsing: NOTE: Cause perf to write to stderr, not stdout
                # - text=True: return output as decoded text instead of bytes
                # - env=env: use custom environment vars (model ID, horizon, context path)

            # try HW counters first
            proc = run_perf(hw_events)
            if proc.returncode != 0:
                self.log.warning(f"perf HW counters blocked (rc={proc.returncode}). Falling back to software events.")
                proc = run_perf(sw_events)

            data = {}

            # This part extracts the relevant performance metrics from the stderr output of the perf command.
            # And logs them in a structured dictionary format.
            # This cause perf write to stderr, not stdout
            for line in proc.stderr.splitlines():
                parts = line.split(",")
                if len(parts) >= 3:
                    val, _, ev = parts[0], parts[1], parts[2]
                    # try int, then float
                    try:
                        data[ev] = int(val.replace(",", ""))
                    except:
                        try:
                            data[ev] = float(val.replace(",", ""))
                        except:
                            pass

            if "cycles" in data and "instructions" in data:
                data["ipc"] = data["instructions"] / max(data["cycles"], 1)
            if "cache-references" in data and "cache-misses" in data:
                data["cache_miss_rate"] = data["cache-misses"] / max(data["cache-references"], 1)

            self.log.debug(f"perf counters parsed: {data}")
            return data

    # ----------------------- Run Config -----------------------
    def run_single_cfg(self, pipe, city, series, model_name, context_hours, horizon_hours):
        vals = series.values
        n = len(vals)
        proc = psutil.Process(os.getpid())
        start = context_hours

        self.log.info(f"[RUN] model={model_name} city={city} "
                      f"ctx={context_hours}h hor={horizon_hours}h")

        for it, t in enumerate(range(start, n - horizon_hours, self.step)):

            # We gotta keep this optional 
            if self.max_iters is not None and it >= self.max_iters:
                self.log.debug("Reached max_iters cap; stopping this config.")
                break

            hist = torch.tensor(vals[t - context_hours:t], dtype=self.dtype)

            rss_before = proc.memory_info().rss
            do_prof = self.use_profiler and (it % self.prof_sample_every == 0)
            mean, latency, flop_stats = self.timed_infer_with_optional_flops(
                pipe, hist, horizon_hours, do_prof
            )
            rss_after = proc.memory_info().rss
            delta_rss = rss_after - rss_before

            row = {
                "city": city,
                "model": model_name,
                "context_hours": context_hours,
                "horizon_hours": horizon_hours,
                "iter_idx": it,
                "latency_s": latency,
                "proc_rss_bytes_before": rss_before,
                "proc_rss_bytes_after": rss_after,
                "proc_rss_delta_bytes": delta_rss,
            }

            # FLOPs and derived GFLOP/s
            if flop_stats:
                row.update(flop_stats)
                if "flops_total" in flop_stats and latency > 0:
                    row["gflops_per_s"] = flop_stats["flops_total"] / latency / 1e9 # Throughput = FLOPs / time (or, work_done / time_taken)
                self.log.debug(f"[prof] it={it} FLOPs={row.get('flops_total')} "
                               f"GFLOP/s={row.get('gflops_per_s')}")

            # Perf counters
            if self.use_perf and (it % self.perf_sample_every == 0):
                perf_stats = self.perf_one_call(model_name, hist.cpu().numpy(), horizon_hours)
                row.update(perf_stats)

            self.rows.append(row)

            # light INFO every few steps
            if it % 8 == 0:
                self.log.info(f"it={it:04d} latency={latency:.4f}s "
                              f"rssΔ={delta_rss/1e6:.2f}MB "
                              f"{'(prof)' if do_prof else ''} "
                              f"{'(perf)' if (self.use_perf and it % self.perf_sample_every == 0) else ''}")

    # ----------------------- Top-level Run -----------------------
    def run_all(self, context_days_sweep, horizon_hours_sweep):
        ggn = self.load_series(self.data_paths["ggn"])
        pna = self.load_series(self.data_paths["pna"])

        for model_name in self.models:
            self.log.info(f"Loading model: {model_name}")
            pipe = BaseChronosPipeline.from_pretrained(model_name, device_map=self.device, dtype=self.dtype)

            # vary context, fixed horizon
            for days in tqdm(context_days_sweep, desc=f"{model_name} context sweep", unit="cfg"):
                ctx_h = days * 24
                for city, ser in [("Gurgaon", ggn), ("Patna", pna)]:
                    self.run_single_cfg(pipe, city, ser, model_name, ctx_h, 24)

            # fixed context, vary horizon
            ctx_h_fixed = 10 * 24
            for horizon in tqdm(horizon_hours_sweep, desc=f"{model_name} horizon sweep", unit="cfg"):
                for city, ser in [("Gurgaon", ggn), ("Patna", pna)]:
                    self.run_single_cfg(pipe, city, ser, model_name, ctx_h_fixed, horizon)

        self.save_results()
        self.plot_summary()

    # ----------------------- Save + Plot -----------------------
    def save_results(self):
        df = pd.DataFrame(self.rows)

        # 0) Combined (all models, all cities) — keep as a convenience
        combined_path = os.path.join(self.out_dir, "chronos_perf_compute_log.csv")
        df.to_csv(combined_path, index=False)
        self.log.info(f"Wrote combined CSV: {combined_path}  ({len(df)} rows)")

        # 1) Per-model folders, overwrite files if they exist
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

    def plot_summary(self):
        try:
            df = pd.DataFrame(self.rows)
            if df.empty:
                self.log.warning("No rows to plot.")
                return

            # group by model → write all plots into that model’s folder
            for model_name, g_all in df.groupby("model"):
                tag = self._model_tag(model_name)
                model_dir = os.path.join(self.out_dir, tag)
                os.makedirs(model_dir, exist_ok=True)

                for city, g in g_all.groupby("city"):
                    city_tag = city.replace(" ", "")

                    # ---- Latency vs Context (fixed 24h)
                    d1 = g[g.horizon_hours == 24]
                    if len(d1):
                        plt.figure()
                        plt.plot(d1.context_hours / 24, d1.latency_s, marker="o")
                        plt.title(f"{city} - Latency vs Context - {tag}")
                        plt.xlabel("Context (days)")
                        plt.ylabel("Latency (s)")
                        plt.grid(True)
                        out = os.path.join(model_dir, f"{city_tag}_latency_vs_context.png")
                        plt.savefig(out, bbox_inches="tight"); plt.close()
                        self.log.info(f"Saved plot: {out}")

                    # ---- Latency vs Horizon (fixed 10d)
                    d2 = g[g.context_hours == 10 * 24]
                    if len(d2):
                        d2m = (d2.groupby("horizon_hours", as_index=False)["latency_s"].mean()
                                .sort_values("horizon_hours"))
                        plt.figure()
                        plt.plot(d2m.horizon_hours, d2m.latency_s, marker="o")
                        plt.title(f"{city} - Latency vs Horizon - {tag}")
                        plt.xlabel("Forecast horizon (hours)")
                        plt.ylabel("Latency (s)")
                        plt.grid(True)
                        out = os.path.join(model_dir, f"{city_tag}_latency_vs_horizon.png")
                        plt.savefig(out, bbox_inches="tight"); plt.close()
                        self.log.info(f"Saved plot: {out}")

                    # ---- GFLOP/s vs Latency (if recorded)
                    if "gflops_per_s" in g.columns and g["gflops_per_s"].notna().any():
                        dfl = g.dropna(subset=["gflops_per_s"])
                        if len(dfl):
                            plt.figure()
                            plt.scatter(dfl.latency_s, dfl.gflops_per_s, s=16)
                            plt.title(f"{city} - GFLOP/s vs Latency - {tag}")
                            plt.xlabel("Latency (s)")
                            plt.ylabel("GFLOP/s")
                            plt.grid(True)
                            out = os.path.join(model_dir, f"{city_tag}_gflops_vs_latency.png")
                            plt.savefig(out, bbox_inches="tight"); plt.close()
                            self.log.info(f"Saved plot: {out}")

                    # ---- Perf: IPC & miss rate vs context (24h) if counters exist
                    core_cycles = "cpu_core/cycles:u/"
                    core_instr  = "cpu_core/instructions:u/"
                    miss        = "cpu_core/cache-misses:u/"
                    refs        = "cpu_core/cache-references:u/"
                    d24 = g[g.horizon_hours == 24]

                    if core_instr in d24.columns and core_cycles in d24.columns and len(d24):
                        d_ipc = (d24.groupby("context_hours", as_index=False)[[core_instr, core_cycles]].mean())
                        d_ipc["ipc"] = d_ipc[core_instr] / d_ipc[core_cycles].replace(0, np.nan)
                        if d_ipc["ipc"].notna().any():
                            plt.figure()
                            plt.plot(d_ipc.context_hours / 24, d_ipc.ipc, marker="o")
                            plt.title(f"{city} - IPC vs Context - {tag}")
                            plt.xlabel("Context (days)")
                            plt.ylabel("IPC")
                            plt.grid(True)
                            out = os.path.join(model_dir, f"{city_tag}_ipc_vs_context.png")
                            plt.savefig(out, bbox_inches="tight"); plt.close()
                            self.log.info(f"Saved plot: {out}")

                    if miss in d24.columns and refs in d24.columns and len(d24):
                        d_mr = (d24.groupby("context_hours", as_index=False)[[miss, refs]].mean())
                        d_mr["miss_rate"] = d_mr[miss] / d_mr[refs].replace(0, np.nan)
                        if d_mr["miss_rate"].notna().any():
                            plt.figure()
                            plt.plot(d_mr.context_hours / 24, d_mr.miss_rate, marker="o")
                            plt.title(f"{city} - Miss rate vs Context - {tag}")
                            plt.xlabel("Context (days)")
                            plt.ylabel("Cache miss rate")
                            plt.grid(True)
                            out = os.path.join(model_dir, f"{city_tag}_missrate_vs_context.png")
                            plt.savefig(out, bbox_inches="tight"); plt.close()
                            self.log.info(f"Saved plot: {out}")

        except Exception as e:
            self.log.warning(f"Plotting skipped: {repr(e)}")

if __name__ == "__main__":
    MODELS = ["amazon/chronos-t5-small", "amazon/chronos-t5-tiny", "amazon/chronos-t5-base"]  # add base, tiny as needed
    DATA = {
        "ggn": "../data/df_ggn_covariates.csv",
        "pna": "../data/df_patna_covariates.csv",
    }

    CONTEXT_DAYS_SWEEP  = [2, 4, 8, 10, 14]
    HORIZON_HOURS_SWEEP = [4, 8, 12, 24, 48]

    # env override for quick dev runs
    env_max_iters = os.environ.get("MAX_ITERS")
    max_iters = int(env_max_iters) if env_max_iters is not None else 16
    
    logger = ChronosPerfLogger(
        models=MODELS,
        data_paths=DATA,
        out_dir="../outputs/perflogs",
        use_perf=True,
        use_profiler=True,
        max_iters=max_iters,
        rolling_step_hours=24,
        perf_sample_every=16,   # perf every 16th call
        prof_sample_every=16,    # FLOPs every 16th call
        log_level="DEBUG"
    )

    logger.run_all(CONTEXT_DAYS_SWEEP, HORIZON_HOURS_SWEEP)
