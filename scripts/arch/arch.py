import torch
import pandas as pd
from chronos import BaseChronosPipeline

def summarize_model(model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    # Load Chronos pipeline
    pipe = BaseChronosPipeline.from_pretrained(model_name, device_map=device, dtype=dtype)

    # Try to find the actual model inside the pipeline
    model = None
    if hasattr(pipe, "model"):
        model = pipe.model
    elif hasattr(pipe, "chronos_model"):
        model = pipe.chronos_model
    elif hasattr(pipe, "backbone"):
        model = pipe.backbone
    else:
        raise AttributeError(f"Cannot find model attribute in pipeline of type {type(pipe)}")

    # Extract parameter names, shapes, and number of parameters
    rows = []
    for name, param in model.named_parameters():
        rows.append({
            "Layer": name,
            "Shape": list(param.shape),
            "Params": param.numel()
        })

    # Create a DataFrame for tabular display
    df = pd.DataFrame(rows)
    total_params = df["Params"].sum()
    df.loc[len(df)] = ["TOTAL", "-", total_params]

    # Print table
    print(f"\n=== {model_name} Architecture Summary ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    for model_name in [
        "amazon/chronos-t5-tiny",
        "amazon/chronos-t5-small",
        "amazon/chronos-t5-base"
    ]:
        summarize_model(model_name)
