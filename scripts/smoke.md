# -------- Config --------
PY        ?= python
PIP       ?= pip
ENV_NAME  ?= chronos
OUT_DIR   ?= outputs
DATA_DIR  ?= data
SCRIPT    ?= chronos_perf_compute_eval.py

# -------- Phony --------
.PHONY: help env deps perf perf-test profiler-test verify run run-fast clean superclean

help:
	@echo "Targets:"
	@echo "  env             - show active Python and env info"
	@echo "  deps            - install Python deps for Chronos perf"
	@echo "  perf            - install Linux perf tools and set perf_event_paranoid"
	@echo "  perf-test       - quick perf sanity test"
	@echo "  profiler-test   - quick torch.profiler FLOPs sanity test"
	@echo "  verify          - run both perf-test and profiler-test"
	@echo "  run             - run the full perf+compute eval script"
	@echo "  run-fast        - run with fewer iters via env var MAX_ITERS=8"
	@echo "  clean           - remove generated plots and logs"
	@echo "  superclean      - clean plus pip cache purge"

env:
	@echo "Python: $$($(PY) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Conda env (if any): $$CONDA_DEFAULT_ENV"
	@echo "Script: $(SCRIPT)"
	@echo "Data dir: $(DATA_DIR)"
	@echo "Out dir:  $(OUT_DIR)"

deps:
	$(PIP) install --upgrade pip
	$(PIP) install torch numpy pandas tqdm psutil matplotlib
	$(PIP) install "transformers>=4.40.0" sentencepiece

perf:
	@echo "Installing Linux perf tools (Debian Ubuntu)..."
	sudo apt update
	sudo apt install -y linux-tools-common linux-tools-generic linux-tools-`uname -r` || true
	@echo "Setting kernel.perf_event_paranoid to 1 (may prompt for sudo)..."
	sudo sysctl -w kernel.perf_event_paranoid=1
	@echo "perf version:"
	-perf --version || true

perf-test:
	@echo "Running perf stat smoke test..."
	-perf stat -x, -e cycles,instructions,cache-references,cache-misses,branches,branch-misses $(PY) - << 'PY'
print("hello from perf test")
PY

profiler-test:
	@echo "Running torch.profiler FLOPs smoke test..."
	$(PY) - << 'PY'
import torch
from torch.profiler import profile, record_function, ProfilerActivity
x = torch.randn(1000,1000)
y = torch.randn(1000,1000)
with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
    with record_function("matmul"):
        z = x @ y
total_flops = sum(getattr(e, "flops", 0) for e in prof.key_averages())
print("Estimated FLOPs:", int(total_flops))
PY

verify: perf-test profiler-test
	@echo "Both perf and profiler smoke tests executed."

run:
	@mkdir -p $(OUT_DIR)
	$(PY) $(SCRIPT)

# Run with a lighter pass by limiting iterations per config
run-fast:
	@mkdir -p $(OUT_DIR)
	MAX_ITERS=8 $(PY) $(SCRIPT)

clean:
	@echo "Cleaning outputs..."
	@rm -f $(OUT_DIR)/*.png $(OUT_DIR)/*.csv

superclean: clean
	@echo "Purging pip cache..."
	-$(PIP) cache purge
