import torch
import numpy as np
from chronos import BaseChronosPipeline  # original Chronos
# For Chronos-Bolt, import ChronosBoltPipeline instead

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",                # stick to CPU for the assignment
    dtype=torch.float32              # keep it simple on CPU
)

# 10-day hourly context -> 240 points
context = torch.tensor(100 + 10*np.sin(np.linspace(0, 6*np.pi, 240)), dtype=torch.float32)

# Forecast 24 hours ahead - returns mean and chosen quantiles
quantiles, mean = pipeline.predict_quantiles(
    context=context,
    prediction_length=24, #hours #window_size
    quantile_levels=[0.1, 0.5, 0.9]
)

print("Mean forecast shape:", mean.shape)          # [1, 24]
print("Quantiles shape:", quantiles.shape)         # [1, 24, 3]
# print("Median first 5:", quantiles[0, :5, 1])
print("Median first 5:", quantiles)

