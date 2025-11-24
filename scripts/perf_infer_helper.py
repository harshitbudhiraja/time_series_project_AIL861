# perf_infer_helper.py
import os, numpy as np, torch
from chronos import BaseChronosPipeline

model_id = os.environ["PERF_MODEL"]
horizon  = int(os.environ["PERF_HORIZON"])
ctx_path = os.environ["PERF_CTX_PATH"]

pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", dtype=torch.float32)
hist = torch.tensor(np.load(ctx_path), dtype=torch.float32)

with torch.no_grad():
    _, _ = pipe.predict_quantiles(context=hist, prediction_length=horizon, quantile_levels=[0.5])
