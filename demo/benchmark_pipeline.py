"""
RapidDoc Pipeline Benchmark Tool

Metrics collected:
  - Per-model inference time + FPS (layout, ocr_det, ocr_rec, table, formula, pdf_det)
  - NPU clock utilisation per device (via dxrt-cli --monitor)
  - PDF content statistics (pages, text/title blocks, tables, formulas, figures)
  - End-to-end wall-clock throughput (pages/sec, docs/sec)

Usage:
    source ./deepx_scripts/set_env.sh 1 2 1 3 2 4
    python demo/benchmark_pipeline.py --pdf test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf
    python demo/benchmark_pipeline.py --pdf a.pdf b.pdf --iterations 3 --output ./bench_out
    python demo/benchmark_pipeline.py --pdf doc.pdf --no-async --layout-engine onnxruntime
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure the project root is importable when this script is run from demo/
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Closed-network guard (same as demo_offline.py)
# ---------------------------------------------------------------------------
os.environ['MINERU_MODEL_SOURCE'] = 'local'

import requests
import urllib.request
import urllib.error

def _blocked(*args, **kwargs):
    raise ConnectionError("Network access blocked in benchmark mode")

requests.get = _blocked
urllib.request.urlopen = _blocked
urllib.request.urlretrieve = _blocked
# ---------------------------------------------------------------------------

from loguru import logger


# ============================================================================
# NPU Monitor  (dxtop primary → dxrt-cli fallback)
# ============================================================================

class NpuMonitor(threading.Thread):
    """Background thread that samples dxtop (via pty) for per-core NPU stats.
    Falls back to dxrt-cli --monitor when dxtop is unavailable.

    summary() returns::
        {device_id: {'avg_util_pct': float, 'cores': {core_id: {...}}}}
    """

    _ANSI_RE = re.compile(
        rb'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|[()][0-9A-Za-z]|[=>])'
        rb'|[\x0e\x0f]'
    )
    # dxtop: "  Core :0   Util:   5.3%   Temp: 44 °C   Voltage: 750 mV   Clock: 1000 MHz"
    _CORE_RE = re.compile(
        r'Core\s*:(\d+)\s+Util:\s*([\d.]+)%'
        r'.*?Temp:\s*(\d+)'
        r'.*?Voltage:\s*(\d+)\s*mV'
        r'.*?Clock:\s*(\d+)\s*MHz',
        re.IGNORECASE,
    )
    _DEV_DXT = re.compile(r'Device\s*:(\d+)', re.IGNORECASE)
    # dxrt-cli fallback
    _DXRTCLI = re.compile(
        r'NPU\s+(\d+):\s+voltage\s+(\d+)\s+mV,\s+clock\s+(\d+)\s+MHz,'
        r'\s+temperature\s+(\d+)'
    )
    _DEV_CLI = re.compile(r'Device\s+(\d+)')

    def __init__(self, interval: int = 1):
        super().__init__(daemon=True, name='NpuMonitor')
        self.interval  = max(1, int(interval))
        self._stop_event = threading.Event()
        self._proc     = None
        self._lock     = threading.Lock()
        # {device_id: {core_id: [(util_pct, temp_c, clock_mhz, voltage_mv)]}}
        self.samples: Dict[int, Dict[int, list]] = defaultdict(lambda: defaultdict(list))

    def run(self) -> None:
        if shutil.which('dxtop'):
            self._run_dxtop()
        elif shutil.which('dxrt-cli'):
            self._run_dxrt_cli()
        else:
            logger.warning('NpuMonitor: neither dxtop nor dxrt-cli found — monitoring disabled.')

    # ── dxtop (primary) ──────────────────────────────────────────────────────

    def _run_dxtop(self) -> None:
        import pty
        import select as _sel
        try:
            master_fd, slave_fd = pty.openpty()
        except OSError as exc:
            logger.warning(f'NpuMonitor: pty.openpty() failed: {exc}')
            self._run_dxrt_cli()
            return

        env = {**os.environ, 'TERM': 'vt100', 'LINES': '60', 'COLUMNS': '160'}
        try:
            self._proc = subprocess.Popen(
                ['dxtop'],
                stdin=slave_fd, stdout=slave_fd, stderr=subprocess.DEVNULL,
                env=env, close_fds=True,
            )
        except FileNotFoundError:
            os.close(master_fd); os.close(slave_fd)
            self._run_dxrt_cli()
            return
        os.close(slave_fd)

        buf        = b''
        last_parse = time.monotonic()
        while not self._stop_event.is_set():
            try:
                r, _, _ = _sel.select([master_fd], [], [], 0.1)
            except (ValueError, OSError):
                break
            if r:
                try:
                    buf += os.read(master_fd, 8192)
                except OSError:
                    break
            if time.monotonic() - last_parse >= self.interval:
                clean = self._ANSI_RE.sub(b'', buf).decode('utf-8', errors='replace')
                self._parse_dxtop(clean)
                buf        = b''
                last_parse = time.monotonic()

        try:
            os.write(master_fd, b'q')
        except OSError:
            pass
        try:
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
        try:
            os.close(master_fd)
        except OSError:
            pass

    def _parse_dxtop(self, text: str) -> None:
        current_dev = 0
        for line in text.splitlines():
            m = self._DEV_DXT.search(line)
            if m:
                current_dev = int(m.group(1))
                continue
            m = self._CORE_RE.search(line)
            if m:
                core_id    = int(m.group(1))
                util_pct   = float(m.group(2))
                temp_c     = int(m.group(3))
                voltage_mv = int(m.group(4))
                clock_mhz  = int(m.group(5))
                with self._lock:
                    self.samples[current_dev][core_id].append(
                        (util_pct, temp_c, clock_mhz, voltage_mv)
                    )

    # ── dxrt-cli (fallback) ──────────────────────────────────────────────────

    def _run_dxrt_cli(self) -> None:
        try:
            self._proc = subprocess.Popen(
                ['dxrt-cli', '--monitor', str(self.interval)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except FileNotFoundError:
            logger.warning('NpuMonitor: dxrt-cli not found — NPU monitoring disabled.')
            return
        current_dev = 0
        for line in self._proc.stdout:
            if self._stop_event.is_set():
                break
            m = self._DEV_CLI.search(line)
            if m:
                current_dev = int(m.group(1))
                continue
            m = self._DXRTCLI.search(line)
            if m:
                _npu, voltage_mv, clock_mhz, temp_c = (int(x) for x in m.groups())
                util = 100.0 if clock_mhz > 0 else 0.0
                with self._lock:
                    self.samples[current_dev][0].append(
                        (util, temp_c, clock_mhz, voltage_mv)
                    )

    # ─────────────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass

    def summary(self) -> Dict:
        """Return {device_id: {'avg_util_pct': float, 'cores': {core_id: {...}}}}"""
        with self._lock:
            result: Dict = {}
            for dev, cores in sorted(self.samples.items()):
                all_utils: list = []
                core_data: Dict[int, Dict] = {}
                for core_id, samps in sorted(cores.items()):
                    if not samps:
                        continue
                    utils  = [s[0] for s in samps]
                    temps  = [s[1] for s in samps]
                    clocks = [s[2] for s in samps]
                    all_utils.extend(utils)
                    core_data[core_id] = {
                        'samples':       len(samps),
                        'util_pct':      round(sum(utils)  / len(utils),  1),
                        'avg_temp_c':    round(sum(temps)  / len(temps),  1),
                        'max_temp_c':    max(temps),
                        'avg_clock_mhz': round(sum(clocks) / len(clocks), 1),
                        'max_clock_mhz': max(clocks),
                    }
                if core_data:
                    result[dev] = {
                        'avg_util_pct': round(sum(all_utils) / len(all_utils), 1),
                        'cores': core_data,
                    }
            return result


# ============================================================================
# Content stats from middle_json
# ============================================================================

def _count_content(middle_json: dict) -> Dict:
    """Count text blocks, tables, formulas, figures from a middle_json dict."""
    TEXT_CATEGORY_IDS = {0, 1, 4, 6, 7, 15, 16}   # Title, Text, captions, ocr text
    TABLE_CATEGORY_IDS = {5}                         # TableBody
    FORMULA_CATEGORY_IDS = {8, 9, 13, 14}            # interline/inline equations
    FIGURE_CATEGORY_IDS = {3}                        # ImageBody

    counts = {
        'pages': 0,
        'text_blocks': 0,
        'table_blocks': 0,
        'formula_blocks': 0,
        'figure_blocks': 0,
    }

    pdf_info = middle_json.get("pdf_info", [])
    counts['pages'] = len(pdf_info)
    for page in pdf_info:
        for block in page.get("preproc_blocks", []):
            cat = block.get("type", -1)
            # layout_dets use category_id; preproc_blocks use "type" as string
            # Handle both numeric and string types
            if isinstance(cat, str):
                cat_lower = cat.lower()
                if cat_lower in ("title", "text", "caption"):
                    counts['text_blocks'] += 1
                elif cat_lower in ("table",):
                    counts['table_blocks'] += 1
                elif cat_lower in ("formula", "equation", "interlineequation"):
                    counts['formula_blocks'] += 1
                elif cat_lower in ("figure", "image"):
                    counts['figure_blocks'] += 1
            else:
                if cat in TEXT_CATEGORY_IDS:
                    counts['text_blocks'] += 1
                elif cat in TABLE_CATEGORY_IDS:
                    counts['table_blocks'] += 1
                elif cat in FORMULA_CATEGORY_IDS:
                    counts['formula_blocks'] += 1
                elif cat in FIGURE_CATEGORY_IDS:
                    counts['figure_blocks'] += 1

        # Also check para_blocks if preproc_blocks is empty
        for block in page.get("para_blocks", []):
            btype = block.get("type", "")
            if isinstance(btype, str):
                btype_l = btype.lower()
                if btype_l in ("title", "text", "caption"):
                    counts['text_blocks'] += 1
                elif btype_l == "table":
                    counts['table_blocks'] += 1
                elif btype_l in ("formula", "equation", "interlineequation"):
                    counts['formula_blocks'] += 1
                elif btype_l in ("figure", "image"):
                    counts['figure_blocks'] += 1

    return counts


def _count_from_model_json(model_json: list) -> Dict:
    """Fall-back: count directly from layout detection results (model_json)."""
    # category_id mapping (from CategoryId class)
    TEXT_IDS   = {0, 1, 4, 6, 7, 15, 16}
    TABLE_IDS  = {5}
    FORMULA_IDS= {8, 9, 13, 14}
    FIGURE_IDS = {3}

    counts = {
        'pages':          len(model_json),
        'text_blocks':    0,
        'table_blocks':   0,
        'formula_blocks': 0,
        'figure_blocks':  0,
    }
    for page in model_json:
        for det in page.get("layout_dets", []):
            cid = det.get("category_id", -1)
            if cid in TEXT_IDS:
                counts['text_blocks'] += 1
            elif cid in TABLE_IDS:
                counts['table_blocks'] += 1
            elif cid in FORMULA_IDS:
                counts['formula_blocks'] += 1
            elif cid in FIGURE_IDS:
                counts['figure_blocks'] += 1
    return counts


# ============================================================================
# Core benchmark function
# ============================================================================

def run_benchmark(
    pdf_paths: List[str],
    output_dir: str,
    iterations: int = 1,
    layout_engine: str = "dxengine",
    ocr_engine: str = "dxengine",
    formula_engine: str = "onnxruntime",
    table_engine: str = "dxengine",
    formula_enable: bool = True,
    table_enable: bool = True,
    formula_rec_enable: bool = True,
    use_async: bool = True,
    npu_monitor_interval: int = 1,
    json_report: Optional[str] = None,
) -> Dict:
    """Execute the pipeline benchmark and return a structured report dict."""
    # Late imports to benefit from the network-block patch above
    from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze

    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
    from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType
    from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
    from rapid_doc.model.formula.rapid_formula_self.utils.typings import EngineType as FormulaEngineType
    from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

    project_root = Path(__file__).parent.parent.absolute()
    onnx_models_dir = project_root / "onnx_models"
    dxnn_models_dir = project_root / "dxnn_models"

    # ---- Layout config -------------------------------------------------------
    layout_config = {"model_type": LayoutModelType.PP_DOCLAYOUT_L}
    if layout_engine.lower() == "dxengine":
        layout_config["engine_type"] = LayoutEngineType.DXENGINE
        layout_config["model_dir_or_path"] = str(dxnn_models_dir / "pp_doclayout_l_part1.dxnn")
        layout_config["sub_model_path"]    = str(onnx_models_dir  / "pp_doclayout_l_part2.onnx")
    elif layout_engine.lower() == "openvino":
        layout_config["engine_type"] = LayoutEngineType.OPENVINO
        layout_config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")
    else:
        layout_config["engine_type"] = LayoutEngineType.ONNXRUNTIME
        layout_config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")

    # ---- OCR config ----------------------------------------------------------
    ocr_config = {
        "use_det_mode": "auto",
        "engine_type": ocr_engine,
        "Det.model_path": str(dxnn_models_dir / "det_v5_640_640.dxnn"),
        "Rec.model_path": str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
        "char_dict_path": str(project_root / "value_compare" / "recognition" / "character_dict_from_onnx.txt"),
        "use_multi_det_model": True,
        "Det.model_paths": {
            1:  str(dxnn_models_dir / "det_v5_640_640.dxnn"),
            2:  str(dxnn_models_dir / "det_v5_320_640.dxnn"),
            4:  str(dxnn_models_dir / "det_v5_160_640.dxnn"),
            10: str(dxnn_models_dir / "det_v5_64_640.dxnn"),
        },
        "use_multi_rec_model": True,
        "Rec.model_paths": {
            3:  str(dxnn_models_dir / "rec_v5_ratio_3.dxnn"),
            5:  str(dxnn_models_dir / "rec_v5_ratio_5.dxnn"),
            10: str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
            15: str(dxnn_models_dir / "rec_v5_ratio_15.dxnn"),
            25: str(dxnn_models_dir / "rec_v5_ratio_25.dxnn"),
            35: str(dxnn_models_dir / "rec_v5_ratio_35.dxnn"),
        },
        "save_debug_images": False,
        "debug_save_dir": os.path.join(output_dir, "ocr_debug"),
    }

    # ---- Formula config ------------------------------------------------------
    formula_config = {"model_type": FormulaModelType.PP_FORMULANET_PLUS_M}
    if not formula_rec_enable:
        formula_config["formula_rec_enable"] = False
    if formula_engine.lower() == "openvino":
        formula_config["engine_type"] = FormulaEngineType.OPENVINO
        formula_config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")
    else:
        formula_config["engine_type"] = FormulaEngineType.ONNXRUNTIME
        formula_config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")

    # ---- Table config --------------------------------------------------------
    table_config = {"model_type": TableModelType.UNET}
    if table_engine.lower() == "dxengine":
        table_config["engine_type"] = "dxengine"
        table_config["model_dir_or_path"] = str(dxnn_models_dir / "unet.dxnn")
    else:
        table_config["engine_type"] = "onnxruntime"
        table_config["model_dir_or_path"] = str(onnx_models_dir / "unet.onnx")

    checkbox_config = {"checkbox_enable": False}

    # ---- Load PDFs -----------------------------------------------------------
    file_names: List[str] = []
    pdf_bytes_list: List[bytes] = []
    for p in pdf_paths:
        file_names.append(Path(p).stem)
        pdf_bytes_list.append(Path(p).read_bytes())

    # ---- Iteration loop ------------------------------------------------------
    iteration_results: List[Dict] = []

    for it in range(1, iterations + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Benchmark iteration {it}/{iterations}")
        logger.info(f"{'='*80}")

        # Fresh copy for each iteration
        fresh_pdf_bytes = list(pdf_bytes_list)

        # Start NPU monitor
        npu_mon = NpuMonitor(interval=npu_monitor_interval)
        npu_mon.start()

        wall_start = time.perf_counter()

        (
            infer_results,
            all_image_lists,
            all_page_dicts,
            lang_list,
            ocr_enabled_list,
            all_pdf_perf_stats,
            _model_load_times,
        ) = pipeline_doc_analyze(
            fresh_pdf_bytes,
            parse_method="auto",
            formula_enable=formula_enable,
            table_enable=table_enable,
            layout_config=layout_config,
            ocr_config=ocr_config,
            formula_config=formula_config,
            table_config=table_config,
            checkbox_config=checkbox_config,
            use_async_pipeline=use_async,
        )

        wall_elapsed = time.perf_counter() - wall_start

        npu_mon.stop()
        npu_mon.join(timeout=5)
        npu_stats = npu_mon.summary()

        # Collect total page count
        total_pages = sum(len(imgs) for imgs in all_image_lists)

        # ---- Content stats: count directly from layout detection results -----
        # (avoids calling pipeline_result_to_middle_json which triggers an OCR
        # model reload in SYNC mode and causes NPU OOM over multiple iterations)
        content_stats_per_pdf: List[Dict] = [
            {'pdf': file_names[idx], **_count_from_model_json(model_list)}
            for idx, model_list in enumerate(infer_results)
        ]

        # ---- Aggregate model-level perf stats across all PDFs ----------------
        agg: Dict[str, Dict] = defaultdict(lambda: {'time': 0.0, 'count': 0})
        for pdf_idx, pdf_stats in all_pdf_perf_stats.items():
            for model_key, ms in pdf_stats.items():
                agg[model_key]['time']  += ms['time']
                agg[model_key]['count'] += ms['count']

        iteration_results.append({
            'iteration':        it,
            'wall_time_s':      round(wall_elapsed, 3),
            'total_pages':      total_pages,
            'pages_per_sec':    round(total_pages / max(wall_elapsed, 0.001), 3),
            'model_stats':      {k: dict(v) for k, v in agg.items()},
            'content_stats':    content_stats_per_pdf,
            'npu_stats':        npu_stats,
        })

    # ---- Build cross-iteration summary ---------------------------------------
    n = len(iteration_results)
    avg_wall    = sum(r['wall_time_s'] for r in iteration_results) / n
    avg_pps     = sum(r['pages_per_sec'] for r in iteration_results) / n
    total_pages = iteration_results[0]['total_pages']  # same across iters

    # Averaged model stats
    avg_model: Dict[str, Dict] = defaultdict(lambda: {'time': 0.0, 'count': 0})
    for r in iteration_results:
        for k, ms in r['model_stats'].items():
            avg_model[k]['time']  += ms['time'] / n
            avg_model[k]['count'] += ms['count'] // n  # integer average

    # Averaged NPU stats
    avg_npu: Dict[int, Dict] = {}
    all_npu = [r['npu_stats'] for r in iteration_results if r['npu_stats']]
    if all_npu:
        for dev in all_npu[0]:
            dev_stats = [it_npu[dev] for it_npu in all_npu if dev in it_npu]
            if not dev_stats:
                continue
            avg_dev_util = sum(d.get('avg_util_pct', 0) for d in dev_stats) / len(dev_stats)
            avg_cores: Dict[int, Dict] = {}
            for core_id in dev_stats[0].get('cores', {}):
                core_list = [
                    d['cores'][core_id]
                    for d in dev_stats
                    if core_id in d.get('cores', {})
                ]
                if core_list:
                    avg_cores[core_id] = {
                        k: round(sum(c[k] for c in core_list) / len(core_list), 1)
                        for k in ('util_pct', 'avg_temp_c', 'max_temp_c', 'avg_clock_mhz', 'max_clock_mhz')
                        if k in core_list[0]
                    }
            avg_npu[dev] = {
                'avg_util_pct': round(avg_dev_util, 1),
                'cores': avg_cores,
            }

    report = {
        'config': {
            'pdf_files':       [str(p) for p in pdf_paths],
            'iterations':      iterations,
            'layout_engine':   layout_engine,
            'ocr_engine':      ocr_engine,
            'formula_engine':  formula_engine,
            'table_engine':    table_engine,
            'formula_enable':  formula_enable,
            'formula_rec_enable': formula_rec_enable,
            'table_enable':    table_enable,
            'use_async':       use_async,
        },
        'summary': {
            'total_pages':         total_pages,
            'avg_wall_time_s':     round(avg_wall, 3),
            'avg_pages_per_sec':   round(avg_pps, 3),
            'avg_model_stats':     {
                k: {
                    'time_s':     round(v['time'], 3),
                    'count':      v['count'],
                    'fps':        round(v['count'] / max(v['time'], 0.001), 2),
                    'ms_per_item': round(v['time'] / max(v['count'], 1) * 1000, 2),
                }
                for k, v in avg_model.items()
            },
            'avg_npu_stats': avg_npu,
        },
        'iterations': iteration_results,
    }

    # ---- Save JSON report ----------------------------------------------------
    if json_report:
        Path(json_report).parent.mkdir(parents=True, exist_ok=True)
        with open(json_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Benchmark report saved → {json_report}")

    return report


# ============================================================================
# Pretty-printing
# ============================================================================

def _bar(fraction: float, width: int = 20) -> str:
    filled = int(round(fraction * width))
    return '█' * filled + '░' * (width - filled)


def print_report(report: Dict) -> None:
    cfg  = report['config']
    summ = report['summary']
    pdfs = cfg['pdf_files']

    print()
    print("╔" + "═" * 78 + "╗")
    print("║{:^78}║".format("RapidDoc Pipeline Benchmark Report"))
    print("╟" + "─" * 78 + "╢")
    print(f"║  PDFs      : {', '.join(Path(p).name for p in pdfs):<63}║")
    print(f"║  Engines   : layout={cfg['layout_engine']:<10} ocr={cfg['ocr_engine']:<10} "
          f"table={cfg['table_engine']:<10}  ║")
    print(f"║  Formula   : {'enabled' if cfg['formula_enable'] else 'disabled':<5}  "
          f"Table: {'enabled' if cfg['table_enable'] else 'disabled':<5}  "
          f"Async: {'yes' if cfg['use_async'] else 'no':<3}  "
          f"Iterations: {cfg['iterations']:<5}                   ║")
    print("╠" + "═" * 78 + "╣")

    # Throughput
    print(f"║  Total pages   : {summ['total_pages']:<60}║")
    print(f"║  Avg wall time : {summ['avg_wall_time_s']:.3f}s{'':<57}║")
    print(f"║  Avg throughput: {summ['avg_pages_per_sec']:.3f} pages/sec{'':<51}║")
    print("╠" + "═" * 78 + "╣")

    # Model stats table
    print("║{:^78}║".format("Per-Model Inference Statistics (averaged)"))
    print("╟" + "─" * 78 + "╢")
    header = f"  {'Model':<12} {'Time(s)':>8}  {'Count':>6}  {'FPS':>7}  {'ms/item':>8}  {'% of total':>10}"
    print(f"║{header:<78}║")
    print("╟" + "─" * 78 + "╢")

    model_order = ['layout', 'formula', 'pdf_det', 'ocr_det', 'table', 'ocr_rec']
    ms_stats    = summ['avg_model_stats']
    total_model_time = sum(v['time_s'] for v in ms_stats.values()) if ms_stats else 0

    for key in model_order:
        if key not in ms_stats:
            continue
        s = ms_stats[key]
        pct = s['time_s'] / total_model_time * 100 if total_model_time > 0 else 0
        bar = _bar(pct / 100)
        row = (f"  {key:<12} {s['time_s']:>8.3f}  {s['count']:>6}  {s['fps']:>7.2f}"
               f"  {s['ms_per_item']:>8.2f}  {pct:>9.1f}%  {bar}")
        print(f"║{row:<78}║")

    print("╠" + "═" * 78 + "╣")

    # Content stats (first iteration, first PDF)
    if report['iterations'] and report['iterations'][0]['content_stats']:
        print("║{:^78}║".format("PDF Content Statistics (iteration 1)"))
        print("╟" + "─" * 78 + "╢")
        for cs in report['iterations'][0]['content_stats']:
            row = (f"  {cs['pdf']:<20}  pages={cs['pages']:<4}  "
                   f"text={cs['text_blocks']:<5}  tables={cs['table_blocks']:<4}  "
                   f"formulas={cs['formula_blocks']:<4}  figures={cs['figure_blocks']:<4}")
            print(f"║{row:<78}║")
        print("╠" + "═" * 78 + "╣")

    # NPU stats
    npu = summ.get('avg_npu_stats', {})
    if npu:
        print("║{:^78}║".format("NPU Core Utilisation (averaged over iterations)"))
        print("╟" + "─" * 78 + "╢")
        for dev_id, dev_data in sorted(npu.items()):
            avg_util = dev_data.get('avg_util_pct', 0)
            util_bar = _bar(avg_util / 100)
            hdr = f"  Device {dev_id}:  avg util={avg_util:5.1f}%  {util_bar}"
            print(f"║{hdr:<78}║")
            for core_id, cs in sorted(dev_data.get('cores', {}).items()):
                core_bar = _bar(cs.get('util_pct', 0) / 100)
                row = (f"    Core {core_id}:  {cs.get('util_pct', 0):5.1f}%  {core_bar}"
                       f"  {cs.get('avg_clock_mhz', 0):.0f}MHz"
                       f"  {cs.get('avg_temp_c', 0):.1f}°C")
                print(f"║{row:<78}║")
    else:
        print("║{:^78}║".format("NPU: no samples collected (dxtop/dxrt-cli unavailable)"))

    print("╚" + "═" * 78 + "╝")
    print()


# ============================================================================
# CLI
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RapidDoc pipeline benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf", nargs="+", required=True, metavar="FILE",
                   help="One or more PDF files to benchmark")
    p.add_argument("--output", "-o", default="./benchmark_out",
                   help="Output directory for pipeline artefacts")
    p.add_argument("--iterations", "-n", type=int, default=1,
                   help="Number of benchmark iterations")
    p.add_argument("--layout-engine",  default="dxengine",
                   choices=["dxengine", "onnxruntime", "openvino"])
    p.add_argument("--ocr-engine",     default="dxengine",
                   choices=["dxengine", "onnxruntime", "openvino", "torch", "paddle"])
    p.add_argument("--formula-engine", default="onnxruntime",
                   choices=["onnxruntime", "openvino"])
    p.add_argument("--table-engine",   default="dxengine",
                   choices=["dxengine", "onnxruntime", "torch"])
    p.add_argument("--no-formula", dest="formula_enable", action="store_false",
                   help="Disable formula processing (skip formula regions entirely)")
    p.add_argument("--no-formula-rec", dest="formula_rec_enable", action="store_false",
                   help="Skip ONNX formula inference; keep formula regions as images")
    p.add_argument("--no-table",   dest="table_enable",   action="store_false",
                   help="Disable table processing")
    p.add_argument("--no-async",   dest="use_async",      action="store_false",
                   help="Use synchronous pipeline instead of async")
    p.add_argument("--npu-interval", type=int, default=1, metavar="SEC",
                   help="dxrt-cli --monitor sampling interval in seconds")
    p.add_argument("--json-report", metavar="FILE",
                   help="Save full benchmark report as JSON to this path")
    return p


def _check_env() -> None:
    """Warn if mandatory environment variables are missing."""
    required = {
        'CUSTOM_INTER_OP_THREADS_COUNT': '1',
        'CUSTOM_INTRA_OP_THREADS_COUNT': '2',
        'DXRT_DYNAMIC_CPU_THREAD':       '1',
        'DXRT_TASK_MAX_LOAD':            '3',
        'NFH_INPUT_WORKER_THREADS':      '2',
        'NFH_OUTPUT_WORKER_THREADS':     '4',
    }
    issues = [
        f"{k}={os.environ.get(k)!r} (expected {v!r})"
        for k, v in required.items()
        if os.environ.get(k) != v
    ]
    if issues:
        logger.warning("Environment variables not set correctly:")
        for msg in issues:
            logger.warning(f"  {msg}")
        logger.warning("Run: source ./deepx_scripts/set_env.sh 1 2 1 3 2 4")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _check_env()

    report = run_benchmark(
        pdf_paths=args.pdf,
        output_dir=args.output,
        iterations=args.iterations,
        layout_engine=args.layout_engine,
        ocr_engine=args.ocr_engine,
        formula_engine=args.formula_engine,
        table_engine=args.table_engine,
        formula_enable=args.formula_enable,
        table_enable=args.table_enable,
        formula_rec_enable=args.formula_rec_enable,
        use_async=args.use_async,
        npu_monitor_interval=args.npu_interval,
        json_report=args.json_report,
    )

    print_report(report)


if __name__ == "__main__":
    main()
