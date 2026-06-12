# RapidDoc Usage Examples (DeepX)

These examples assume your virtual environment and models/env variables are already set.

## Prerequisites

```shell
# 1) Activate your venv
source venv311/bin/activate

# 2) Install RapidDoc dependencies and editable package
pip install -r requirements.deepx.txt
pip install -e .

# 3) Download the NPU models (ONNX + DXNN) — see "Model & Environment Setup" below
./setup.sh

# 4) Set required environment variables for DX-RT
source ./deepx_scripts/set_env.sh 1 2 1 3 2 4
#   CUSTOM_INTER_OP_THREADS_COUNT=1
#   CUSTOM_INTRA_OP_THREADS_COUNT=2
#   DXRT_DYNAMIC_CPU_THREAD=1
#   DXRT_TASK_MAX_LOAD=3          (NPU I/O buffer depth; presence-only check, any value accepted)
#   NFH_INPUT_WORKER_THREADS=2
#   NFH_OUTPUT_WORKER_THREADS=4

# 5) (Optional) Specify NPU devices — auto-detected if not set
export DXNN_DEVICES=0          # Single NPU
# export DXNN_DEVICES=0,1,2,3  # Multi-NPU
```

---

## Model & Environment Setup (`./setup.sh`)

RapidDoc does **not** require you to compile any model yourself. The bundled
`./setup.sh` provisions every model the pipeline needs — run it once after installing
the dependencies:

```shell
./setup.sh                          # host mode (auto-detected)
./setup.sh --force-remove-models    # re-download, replacing existing models
./setup.sh --docker_volume_path=/path/to/volume   # container mode (required inside Docker)
```

What it does (`setup_assets()` → `setup_sample_models.sh`):

1. Checks for the model payload directories `dxnn_models/` and `onnx_models/`.
2. If either is missing, runs `./setup_sample_models.sh`, which **downloads the prebuilt
   ONNX + DXNN models** into those directories. If both already exist, the download is
   skipped (use `--force-remove-models` to force a refresh).
3. In container mode (`--docker_volume_path`), models are symlinked into the Docker volume.

> **Do NOT hand-compile the models with `dxcom`.** The DXNN binaries are already built
> for the DX-M1 NPU and are fetched by `./setup.sh` — this is the single, foreground
> provisioning step. After it completes, source `deepx_scripts/set_env.sh` (step 4 above)
> and you are ready to run `demo/demo_offline.py`.

---

## CLI Options Reference

### Pipeline Mode (mutually exclusive)

| Option | Description |
|:---|:---|
| `--finegrained` | **(Default)** 7-stage per-page streaming pipeline. Stages (Layout → Plan → Formula → PDF-det → OCR-det → Table → OCR-rec) run concurrently with inter-stage queues. |
| `--use-async` | Batch async pipeline (TrueAsyncPipeline). Processes all pages through each stage before moving to the next. |
| `--no-async` | Sequential sync pipeline. Processes pages one at a time through all stages. Slowest but simplest. |

### Parse Method

| Option | Description |
|:---|:---|
| `--parse-method auto` | **(Default)** Classify PDF → digital PDFs use text layer, scanned PDFs use OCR. Per-page decision: only pages without text are OCR'd. |
| `--parse-method txt` | Use PDF text layer only. No OCR at all. Fastest (~5s for 12-page digital PDF). Pages without embedded text return empty results. |
| `--parse-method ocr` | Force OCR on all pages. Ignores PDF text layer. Use for scanned PDFs or accuracy evaluation (~200s for 12 pages). |

### Inference Options

| Option | Description |
|:---|:---|
| `--no-formula` | Skip formula recognition. Formula regions kept as cropped images instead of LaTeX. |
| `--force-ocr` | (Deprecated) Alias for `--parse-method ocr`. |
| `--hybrid` | Multi-NPU mode: assigns each model to a dedicated NPU device for physical parallelism. Requires 2+ NPUs. |

### Other Options

| Option | Description |
|:---|:---|
| `PATH` | PDF files or directories to process. Defaults to `test_files/` if omitted. |
| `--output-dir DIR` | Output directory. Defaults to `demo/output-offline-{mode}/`. |

---

## Quick Start Examples

```shell
# Basic: process all PDFs in test_files/ with finegrained pipeline
python demo/demo_offline.py test_files --finegrained

# Fast text extraction (digital PDF, no OCR)
python demo/demo_offline.py test_files/document.pdf --parse-method txt

# Force OCR mode (scanned PDF)
python demo/demo_offline.py test_files/scanned.pdf --parse-method ocr

# Multi-NPU with hybrid device allocation
export DXNN_DEVICES=0,1,2,3
python demo/demo_offline.py test_files --finegrained --hybrid

# With NPU utilization monitoring
python run_with_npu_monitor.py python demo/demo_offline.py test_files --finegrained
```

---

## NPU Monitoring

Use `run_with_npu_monitor.py` to measure real-time NPU utilization during pipeline execution:

```shell
python run_with_npu_monitor.py [--interval 1.0] <command...>
```

Example output:
```
┌─────────────────────────────────────────────────────┐
│              NPU Utilization Summary                  │
├─────────────────────────────────────────────────────┤
│  Devices: 1    Samples:  210  (1.0s interval)         │
│  Overall avg:  27.7%    max: 100.0%                   │
│  Idle (0%):    18 samples (8.6%)                      │
│   Core:0  avg= 39.6%  max=100.0%  min=  0.0%          │
│   Core:1  avg= 23.4%  max=100.0%  min=  0.0%          │
│   Core:2  avg= 20.2%  max=100.0%  min=  0.0%          │
└─────────────────────────────────────────────────────┘
```

The monitor auto-detects the number of NPU devices and cores via `dxtop`.

---

## Performance Summary

After each run, a `performance_summary_YYYYMMDD_HHMMSS.md` file is saved in the output directory with:
- Per-stage latency and throughput (Layout, OCR-det, OCR-rec, Table, etc.)
- Per-document elapsed time breakdown

---

## Environment Variables

| Variable | Default | Description |
|:---|:---|:---|
| `DXNN_DEVICES` | Auto-detect | Comma-separated NPU device IDs (e.g., `0,1,2,3`). Auto-detected via DX-RT API or `/dev/dxrt*` scan if not set. |
| `DXRT_TASK_MAX_LOAD` | 3 | NPU I/O buffer depth. Higher values queue more requests but require more NPU memory. Max usable depends on how many models are loaded. |
| `CUSTOM_INTER_OP_THREADS_COUNT` | 1 | Inter-op thread count for DX-RT inference. |
| `CUSTOM_INTRA_OP_THREADS_COUNT` | 2 | Intra-op thread count for DX-RT inference. |

---

## Architecture Notes

- **OCR Preprocessing**: Detection uses pad-to-square with gray(114,114,114) before resize (preserves aspect ratio). Recognition uses gray(114) padding for width normalization. Both match the C++ reference implementation.
- **Multi-model OCR**: Detection uses 4 aspect-ratio models (64×640, 160×640, 320×640, 640×640). Recognition uses 6 width-ratio models (ratio 3/5/10/15/25/35).
- **Hybrid mode**: `DeviceAllocator` distributes models across NPUs — Layout/Table/OCR-det/OCR-rec each get dedicated hardware for true pipeline parallelism.

---

## Other Demos

All demo apps share the same `--finegrained` (default) and hybrid auto-detection optimizations as `demo_offline.py`.

### Offline API Server (FastAPI)
```shell
# Start server (default: finegrained pipeline, deepx engines, hybrid auto-detected)
python demo/app_offline.py

# Customize pipeline mode and hybrid behavior
python demo/app_offline.py --pipeline-mode finegrained --hybrid
python demo/app_offline.py --pipeline-mode sync --no-hybrid
python demo/app_offline.py --no-deepx-default      # Use ONNX engines by default

# Smoke test
curl http://localhost:8888/health | jq          # Shows pipeline mode + hybrid status
python demo/test_api_offline.py
```

CLI options: `--host`, `--port`, `--pipeline-mode {finegrained,async,sync}`, `--hybrid`/`--no-hybrid`, `--deepx-default`/`--no-deepx-default`.

### Gradio Web UI
```shell
python demo/gradio_app.py  # http://localhost:7860
```
Uses `finegrained` pipeline by default; hybrid mode auto-enabled when ≥2 NPUs detected.

### Standalone Benchmark
```shell
# Default: finegrained pipeline
python demo/benchmark_pipeline.py --pdf test_files/sample.pdf -n 3

# Compare modes
python demo/benchmark_pipeline.py --pdf doc.pdf --legacy-async    # AsyncPipeline
python demo/benchmark_pipeline.py --pdf doc.pdf --no-async        # Sync

# Multi-NPU
python demo/benchmark_pipeline.py --pdf doc.pdf --hybrid --json-report bench.json
```

### Minimal Python API Example
See `demo/demo.py` for a minimal `do_parse()` usage example using ONNX models from `onnx_models/`.

---

## Environment Variable Validation

All apps relax `DXRT_TASK_MAX_LOAD` to **presence-only** (any value accepted; recommended `3` for M1PCIe). Other variables in `set_env.sh 1 2 1 3 2 4` must match exactly.
