# play_profiler_flops.py
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# ---- toy model ----
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1, bias=True)
        # self.act = nn.ReLU()
        # self.fc2 = nn.Linear(16, 8, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.act(x)
        # x = self.fc2(x)
        return x

def main():
    torch.manual_seed(0)
    net = TinyNet()            # ~couple of small matmuls
    x = torch.randn(1, 1)     # batch 1

    # Warmup to avoid lazy init skewing numbers
    _ = net(x)

    # Profile a single forward pass
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,         # ask profiler to estimate FLOPs per op
    ) as prof:
        with record_function("tiny_forward"):
            y = net(x)

    # Per-op summary, sorted by FLOPs
    print("\n=== Per-op (sorted by FLOPs) ===")
    print(prof.key_averages().table())#sort_by="flops"))#, row_limit=50))

    # # Programmatic access
    # events = prof.key_averages()
    # total_flops = sum(getattr(e, "flops", 0) for e in events)
    # print(f"\nTotal FLOPs (approx): {int(total_flops):,}")

    # for e in prof.key_averages():
    #     # if e.flops > 0:
    #     print(f"{e.key:25s}  flops={int(e.flops):,}  calls={e.count}  shapes={e.input_shapes}")

    # # Show top few ops with their FLOPs
    # top = sorted(events, key=lambda e: getattr(e, "flops", 0), reverse=True)[:5]
    # print("\nTop ops:")
    # for e in top:
    #     fl = getattr(e, "flops", 0)
    #     print(f"  {e.key:25s}  flops={int(fl):,}  calls={e.count}  shapes={e.input_shapes}")

if __name__ == "__main__":
    main()
