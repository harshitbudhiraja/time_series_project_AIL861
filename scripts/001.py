# chronos_eval.py
import math, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline  # or ChronosBoltPipeline
from tqdm.auto import tqdm

MODELS = [
            # "amazon/chronos-t5-large", 
            "amazon/chronos-t5-base",
            "amazon/chronos-t5-small", 
            "amazon/chronos-t5-tiny"
        ]
DEVICE = "cpu"
DTYPE = torch.float32

CONTEXT_DAYS_SWEEP  = [2, 4, 8, 10, 14]     # Days
HORIZON_HOURS_SWEEP = [4, 8, 12, 24, 48]    # Hours
ROLLING_STEP_HOURS  = 24#1
WARMUP_HOURS        = (1*24) #atleast 

def load_series(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.iloc[:,0], errors="coerce")
    y  = df.iloc[:,4].astype(float)
    s = pd.Series(y.values, index=ts).sort_index()
    s = s.asfreq("H").interpolate(limit_direction="both")
    return s

def avg_rmse(pipeline, series, context_hours, horizon_hours):
    vals = series.values.astype(np.float32)
    n = len(vals)
    # print(n, "\n", vals)
    rmses = []
    # start = max(WARMUP_HOURS, context_hours)
    start = context_hours
    for it, t in tqdm(enumerate(range(start, n - horizon_hours, ROLLING_STEP_HOURS)), desc=f"ctx={context_hours} | hor={horizon_hours}", leave=False, unit="step"):
        hist = torch.tensor(vals[t-context_hours:t], dtype=DTYPE)
        # print(it, t, hist.shape, hist)

        # pipeline returns mean forecast [1, horizon]
        quantile_pred, mean = pipeline.predict_quantiles(
            context=hist,
            prediction_length=horizon_hours, #mean is done across each prediction_length 
            quantile_levels=[0.5]
        )

        # print(f"hist shape:{hist.shape}, quantile_pred shape: {quantile_pred.shape}, mean shape: {mean.shape}")
        pred    = mean[0].cpu().numpy()
        truth   = vals[t:t+horizon_hours]
        rmse    = math.sqrt(((pred - truth) ** 2).mean())
        rmses.append(rmse)
    return float(np.mean(rmses)) if rmses else float("nan")

def main():
    ggn = load_series("../data/df_ggn_covariates.csv")
    pna = load_series("../data/df_patna_covariates.csv")
    # print(ggn.head(), ggn.shape)
    # print(pna.head(), pna.shape)

    def run_model_eval(MODEL):
        MODEL_NAME = MODEL.split("/")[-1]
        print(f"\nEvaluating model: {MODEL} ...")
        # build once
        pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map=DEVICE, dtype=DTYPE)

        # # Smoke test
        # ctx_h = 2 * 24
        # for city, ser in [("Gurgaon", ggn), ("Patna", pna)]:
        #     rmse = avg_rmse(pipe, ser, ctx_h, 24)
        #     res.append({"curve": "rmse_vs_context", "model": MODEL, "city": city,
        #                     "context_days": 2, "horizon_hours": 24, "avg_rmse": rmse})

        # Plot 1 - RMSE vs context for 24h horizon, largest model you can run
        res = []
        for days in tqdm(CONTEXT_DAYS_SWEEP, desc="context/days_sweep", unit="cfg"):
            ctx_h = days * 24

            for city, ser in tqdm([("Gurgaon", ggn), ("Patna", pna)],
                                    desc=f"context_window - {days} days/city", 
                                    leave=False, unit="city"):
                rmse = avg_rmse(pipe, ser, ctx_h, 24)
                res.append({"curve": "rmse_vs_context", "model": MODEL, "city": city,
                            "context_days": days, "horizon_hours": 24, "avg_rmse": rmse})
        print(len(res))

        # Plot 2 - RMSE vs horizon for 10-day context
        ctx_h_fixed = 10 * 24
        for horizon in HORIZON_HOURS_SWEEP:
            for city, ser in [("Gurgaon", ggn), ("Patna", pna)]:
                rmse = avg_rmse(pipe, ser, ctx_h_fixed, horizon)
                res.append({"curve": "rmse_vs_horizon", "model": MODEL, "city": city,
                            "context_days": 10, "horizon_hours": horizon, "avg_rmse": rmse})

        df = pd.DataFrame(res)
        # df = pd.read_csv("../outputs/chronos_rmse_results.csv")
        df.to_csv(f"../outputs/{MODEL_NAME}_chronos_rmse_results.csv", index=False)

        # simple plotting block
        import matplotlib.pyplot as plt
        for city in ["Gurgaon", "Patna"]:
            d1 = df[(df.curve=="rmse_vs_context") & (df.city==city)]
            plt.figure(); plt.plot(d1.context_days, d1.avg_rmse, marker="o")
            plt.title(f"{city} - RMSE vs context - 24h horizon - {MODEL.split('/')[-1]}")
            plt.xlabel("Context length (days)"); plt.ylabel("Average RMSE"); plt.grid(True)
            plt.savefig(f"../outputs/{MODEL_NAME}_{city}_rmse_vs_context_{MODEL.split('/')[-1]}.png", bbox_inches="tight"); plt.close()

            d2 = df[(df.curve=="rmse_vs_horizon") & (df.city==city)]
            plt.figure(); plt.plot(d2.horizon_hours, d2.avg_rmse, marker="o")
            plt.title(f"{city} - RMSE vs horizon - 10-day context - {MODEL.split('/')[-1]}")
            plt.xlabel("Forecast horizon (hours)"); plt.ylabel("Average RMSE"); plt.grid(True)
            plt.savefig(f"../outputs/{MODEL_NAME}_{city}_rmse_vs_horizon_{MODEL.split('/')[-1]}.png", bbox_inches="tight"); plt.close()

    for MODEL in MODELS:
        run_model_eval(MODEL)

if __name__ == "__main__":
    main()
