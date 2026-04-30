#!/usr/bin/env python3
"""
RapidDoc 파이프라인 Async 벤치마크
====================================

Sync(BatchAnalyze) vs Async(TrueAsyncPipeline)를 동일 입력으로 비교하여
스테이지별 소요 시간·처리량을 측정한다.

사용법:
    # 기본 실행 (test_files/ 사용, dxengine)
    python dxnn_benchmark/benchmark_pipeline_async.py

    # 옵션 지정
    python dxnn_benchmark/benchmark_pipeline_async.py \
        --pdf test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf \
        --engine dxengine \
        --warmup 1 \
        --runs 3 \
        --no-formula \
        --no-table

출력 예시:
    ┌─────────────┬──────────┬──────────┬──────────┬──────────────────┐
    │  Stage      │  Sync    │  Async   │  Speedup │  Engine          │
    ├─────────────┼──────────┼──────────┼──────────┼──────────────────┤
    │  Layout     │  1.234 s │  0.987 s │  1.25x   │  dxengine        │
    │  OCR-det    │  0.890 s │  0.456 s │  1.95x   │  dxengine        │
    │  OCR-rec    │  0.678 s │  0.432 s │  1.57x   │  dxengine        │
    │  Table      │  2.100 s │  1.890 s │  1.11x   │  dxengine        │
    │  TOTAL      │  5.234 s │  4.012 s │  1.30x   │                  │
    └─────────────┴──────────┴──────────┴──────────┴──────────────────┘
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# ─── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─── 폐쇄망 설정 ──────────────────────────────────────────────────────────
os.environ.setdefault('MINERU_MODEL_SOURCE', 'local')


# ─────────────────────────────────────────────────────────────────────────────
# 환경 변수 확인
# ─────────────────────────────────────────────────────────────────────────────

def check_env() -> bool:
    required = {
        'CUSTOM_INTER_OP_THREADS_COUNT': '1',
        'CUSTOM_INTRA_OP_THREADS_COUNT': '2',
        'DXRT_DYNAMIC_CPU_THREAD': '1',
        'DXRT_TASK_MAX_LOAD': '3',
        'NFH_INPUT_WORKER_THREADS': '2',
        'NFH_OUTPUT_WORKER_THREADS': '4',
    }
    missing = [k for k in required if k not in os.environ]
    if missing:
        logger.warning(
            f"⚠  환경 변수 미설정 (DX Engine 비활성화 시 무시 가능): {missing}\n"
            "    설정하려면: source ./deepx_scripts/set_env.sh 1 2 1 3 2 4"
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PDF → 페이지 이미지 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_pages(pdf_path: str, start: int = 0, end: Optional[int] = None):
    """
    PDF에서 페이지 이미지와 텍스트 딕셔너리를 로드하여
    파이프라인 입력 형식 (images_with_extra_info) 으로 변환한다.
    """
    from rapid_doc.utils.pdf_image_tools import load_images_from_pdf
    from rapid_doc.utils.pdf_classify import classify
    from rapid_doc.utils.pdf_text_tool import get_page
    from rapid_doc.utils.enum_class import ImageType
    from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2

    logger.info(f"PDF 로드: {pdf_path}")
    pdf_bytes = Path(pdf_path).read_bytes()

    # 페이지 범위 자르기
    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start, end)

    # OCR 필요 여부 판단
    if classify(pdf_bytes) == 'ocr':
        ocr_enable = True
        logger.info("  → 스캔 PDF (OCR 강제 활성화)")
    else:
        ocr_enable = False
        logger.info("  → 텍스트 PDF (PDF-det 우선)")

    images_list, pdf_doc_list = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)

    all_pdf_dict = []
    page_ocr_flags = []
    for pdf_doc in pdf_doc_list:
        page_dict = get_page(pdf_doc)
        has_text = bool(page_dict.get('blocks'))
        page_ocr_flags.append(not has_text)
        if has_text:
            from rapid_doc.utils.pdf_image_tools import get_ori_image
            page_dict['ori_image_list'] = get_ori_image(pdf_doc)
        else:
            page_dict['ori_image_list'] = []
        pdf_doc.close()
        all_pdf_dict.append(page_dict)

    pdf_force_ocr = ocr_enable or any(page_ocr_flags)

    images_with_extra_info = []
    for page_idx, (img_dict, page_dict) in enumerate(zip(images_list, all_pdf_dict)):
        needs_ocr = page_ocr_flags[page_idx] if page_idx < len(page_ocr_flags) else False
        images_with_extra_info.append((
            img_dict['img_pil'],
            img_dict['scale'],
            pdf_force_ocr or needs_ocr,
            "ch",       # language
            page_dict,
            0,          # pdf_idx
            page_idx,
        ))

    logger.info(f"  → {len(images_with_extra_info)} 페이지 로드 완료")
    return images_with_extra_info


# ─────────────────────────────────────────────────────────────────────────────
# 모델 설정 빌더
# ─────────────────────────────────────────────────────────────────────────────

def build_configs(engine: str, project_root: Path, formula_enable: bool, table_enable: bool):
    """엔진 이름(dxengine / onnxruntime)에 따라 모델 설정 딕셔너리를 반환한다."""
    onnx_dir = project_root / "onnx_models"
    dxnn_dir = project_root / "dxnn_models"

    # ── Layout ────────────────────────────────────────────────────────────
    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
    from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType

    if engine == "dxengine":
        layout_config = {
            "model_type": LayoutModelType.PP_DOCLAYOUT_L,
            "engine_type": LayoutEngineType.DXENGINE,
            "model_dir_or_path": str(dxnn_dir / "pp_doclayout_l_part1.dxnn"),
            "sub_model_path": str(onnx_dir / "pp_doclayout_l_part2.onnx"),
        }
    else:
        layout_config = {
            "model_type": LayoutModelType.PP_DOCLAYOUT_L,
            "engine_type": LayoutEngineType.ONNXRUNTIME,
            "model_dir_or_path": str(onnx_dir / "pp_doclayout_l.onnx"),
        }

    # ── OCR ───────────────────────────────────────────────────────────────
    char_dict = str(project_root / "value_compare" / "recognition" / "character_dict_from_onnx.txt")

    if engine == "dxengine":
        ocr_config = {
            "engine_type": "dxengine",
            "Det.model_path": str(dxnn_dir / "det_v5_640_640.dxnn"),
            "Rec.model_path": str(dxnn_dir / "rec_v5_ratio_10.dxnn"),
            "char_dict_path": char_dict,
            "use_det_mode": "auto",
            "use_multi_det_model": True,
            "Det.model_paths": {
                1: str(dxnn_dir / "det_v5_640_640.dxnn"),
                2: str(dxnn_dir / "det_v5_320_640.dxnn"),
                4: str(dxnn_dir / "det_v5_160_640.dxnn"),
                10: str(dxnn_dir / "det_v5_64_640.dxnn"),
            },
            "use_multi_rec_model": True,
            "Rec.model_paths": {
                3: str(dxnn_dir / "rec_v5_ratio_3.dxnn"),
                5: str(dxnn_dir / "rec_v5_ratio_5.dxnn"),
                10: str(dxnn_dir / "rec_v5_ratio_10.dxnn"),
                15: str(dxnn_dir / "rec_v5_ratio_15.dxnn"),
                25: str(dxnn_dir / "rec_v5_ratio_25.dxnn"),
                35: str(dxnn_dir / "rec_v5_ratio_35.dxnn"),
            },
        }
    else:
        from rapidocr import EngineType as OCREngineType
        ocr_config = {
            "Det.engine_type": OCREngineType.ONNXRUNTIME,
            "Rec.engine_type": OCREngineType.ONNXRUNTIME,
            "Det.model_path": str(onnx_dir / "ch_PP-OCRv5_server_det.onnx"),
            "Rec.model_path": str(onnx_dir / "ch_PP-OCRv5_rec_server_infer.onnx"),
            "use_det_mode": "auto",
        }

    # ── Formula ───────────────────────────────────────────────────────────
    from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
    from rapid_doc.model.formula.rapid_formula_self.utils.typings import EngineType as FormulaEngineType

    formula_config = {
        "model_type": FormulaModelType.PP_FORMULANET_PLUS_M,
        "engine_type": FormulaEngineType.ONNXRUNTIME,   # Formula는 onnxruntime 고정
        "model_dir_or_path": str(onnx_dir / "pp_formulanet_plus_m.onnx"),
        "enable": formula_enable,
    }

    # ── Table ─────────────────────────────────────────────────────────────
    from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

    if engine == "dxengine":
        table_config = {
            "model_type": TableModelType.UNET,
            "engine_type": "dxengine",
            "model_dir_or_path": str(dxnn_dir / "unet.dxnn"),
            "enable": table_enable,
        }
    else:
        table_config = {
            "model_type": TableModelType.UNET,
            "engine_type": "onnxruntime",
            "model_dir_or_path": str(onnx_dir / "unet.onnx"),
            "enable": table_enable,
        }

    return layout_config, ocr_config, formula_config, table_config


# ─────────────────────────────────────────────────────────────────────────────
# Sync 파이프라인 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_sync_pipeline(
    images_with_extra_info,
    formula_enable: bool,
    table_enable: bool,
    layout_config: dict,
    ocr_config: dict,
    formula_config: dict,
    table_config: dict,
) -> Tuple[List, Dict, float]:
    """BatchAnalyze 기반 동기 파이프라인을 실행하고 (결과, perf_stats, elapsed) 를 반환한다."""
    from rapid_doc.backend.pipeline.pipeline_analyze import batch_image_analyze

    t0 = time.perf_counter()
    results, perf_stats = batch_image_analyze(
        images_with_extra_info,
        formula_enable=formula_enable,
        table_enable=table_enable,
        layout_config=layout_config,
        ocr_config=ocr_config,
        formula_config=formula_config,
        table_config=table_config,
        checkbox_config={"checkbox_enable": False},
    )
    elapsed = time.perf_counter() - t0
    return results, perf_stats, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Async 파이프라인 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_async_pipeline(
    images_with_extra_info,
    formula_enable: bool,
    table_enable: bool,
    layout_config: dict,
    ocr_config: dict,
    formula_config: dict,
    table_config: dict,
) -> Tuple[List, Dict, float]:
    """TrueAsyncPipeline을 실행하고 (결과, perf_stats, elapsed) 를 반환한다."""
    from rapid_doc.backend.pipeline.async_pipeline import async_batch_image_analyze

    t0 = time.perf_counter()
    results, perf_stats = async_batch_image_analyze(
        images_with_extra_info,
        formula_enable=formula_enable,
        table_enable=table_enable,
        layout_config=layout_config,
        ocr_config=ocr_config,
        formula_config=formula_config,
        table_config=table_config,
        checkbox_config={"checkbox_enable": False},
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    return results, perf_stats, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# 결과 리포터
# ─────────────────────────────────────────────────────────────────────────────

STAGE_ORDER = ['layout', 'formula', 'pdf_det', 'ocr_det', 'table', 'ocr_rec']
STAGE_LABELS = {
    'layout':  '📊 Layout  ',
    'formula': '📐 Formula ',
    'pdf_det': '📄 PDF-det ',
    'ocr_det': '🔍 OCR-det ',
    'table':   '📋 Table   ',
    'ocr_rec': '✍️  OCR-rec ',
}


def _fmt(sec: Optional[float]) -> str:
    if sec is None:
        return "  —      "
    return f"{sec:7.3f}s"


def print_comparison(
    sync_stats: Dict,
    async_stats: Dict,
    sync_total: float,
    async_total: float,
    n_pages: int,
    engine: str,
    n_runs: int,
) -> None:
    """Sync vs Async 비교표를 출력한다."""
    col = 14
    sep = "─" * (col * 4 + 24)

    logger.info("")
    logger.info("=" * (col * 4 + 24))
    logger.info("  Sync vs Async 파이프라인 비교")
    logger.info(f"  엔진: {engine}  |  페이지: {n_pages}  |  실행 횟수: {n_runs}")
    logger.info("=" * (col * 4 + 24))
    header = (
        f"{'Stage':<14}  {'Sync':>{col - 1}}  {'Async':>{col - 1}}  "
        f"{'Speedup':>{col - 2}}  {'Engine'}"
    )
    logger.info(header)
    logger.info(sep)

    for key in STAGE_ORDER:
        label = STAGE_LABELS.get(key, key)
        s_time = sync_stats.get(key, {}).get('time')
        a_time = async_stats.get(key, {}).get('time')

        if s_time is None and a_time is None:
            continue

        if s_time and a_time and a_time > 0:
            speedup = f"{s_time / a_time:.2f}x"
        else:
            speedup = "  —  "

        row = (
            f"{label:<14}  {_fmt(s_time):>{col - 1}}  {_fmt(a_time):>{col - 1}}  "
            f"{speedup:>{col - 2}}  {engine}"
        )
        logger.info(row)

    logger.info(sep)
    total_speedup = f"{sync_total / async_total:.2f}x" if async_total > 0 else "—"
    logger.info(
        f"{'TOTAL':<14}  {_fmt(sync_total):>{col - 1}}  {_fmt(async_total):>{col - 1}}  "
        f"{total_speedup:>{col - 2}}"
    )
    logger.info(
        f"{'Pages/sec':<14}  "
        f"{n_pages / sync_total:>{col - 2}.2f} it/s  "
        f"{n_pages / async_total:>{col - 2}.2f} it/s"
    )
    logger.info("=" * (col * 4 + 24))


def check_output_consistency(sync_results: List, async_results: List) -> bool:
    """
    Sync vs Async 결과의 페이지 수와 검출 항목 수가 일치하는지 확인한다.
    박스 좌표 등 수치 결과는 모델 비결정성으로 약간 다를 수 있으므로
    여기서는 구조적 정합성만 체크한다.
    """
    if len(sync_results) != len(async_results):
        logger.error(
            f"❌ 페이지 수 불일치: sync={len(sync_results)}, async={len(async_results)}"
        )
        return False

    ok = True
    for i, (s_page, a_page) in enumerate(zip(sync_results, async_results)):
        s_cat = sorted([item.get('category_id', -1) for item in s_page])
        a_cat = sorted([item.get('category_id', -1) for item in a_page])
        if s_cat != a_cat:
            logger.warning(
                f"  Page {i}: category_id 분포 차이 "
                f"sync={len(s_cat)}개, async={len(a_cat)}개"
            )
            ok = False

    if ok:
        logger.info("✅ 출력 정합성 검사 통과 (category_id 분포 일치)")
    else:
        logger.warning("⚠  일부 페이지에서 출력 차이 감지 (허용 범위일 수 있음)")
    return ok


def save_report(
    sync_stats: Dict,
    async_stats: Dict,
    sync_total: float,
    async_total: float,
    n_pages: int,
    engine: str,
    output_path: str,
) -> None:
    """벤치마크 결과를 JSON 파일로 저장한다."""
    report = {
        "engine": engine,
        "n_pages": n_pages,
        "sync": {
            "total_sec": sync_total,
            "pages_per_sec": n_pages / sync_total if sync_total > 0 else 0,
            "stages": sync_stats,
        },
        "async": {
            "total_sec": async_total,
            "pages_per_sec": n_pages / async_total if async_total > 0 else 0,
            "stages": async_stats,
        },
        "speedup": sync_total / async_total if async_total > 0 else 0,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"📄 리포트 저장: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _find_pdf(pdf_arg: Optional[str], project_root) -> str:
    """PDF 경로를 결정한다. 미지정 시 test_files/ 의 첫 번째 파일을 반환한다."""
    if pdf_arg:
        return pdf_arg
    pdf_dir = project_root / "demo" / "pdfs"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(f"test_files/ 에 PDF 파일이 없습니다: {pdf_dir}")
        sys.exit(1)
    pdf_path = str(pdfs[0])
    logger.info(f"자동 선택된 PDF: {pdf_path}")
    return pdf_path


def _log_settings(pdf_path, n_pages, engine, formula_enable, table_enable, warmup, runs):
    logger.info("=" * 80)
    logger.info("벤치마크 설정")
    logger.info(f"  PDF      : {pdf_path}")
    logger.info(f"  Pages    : {n_pages}")
    logger.info(f"  Engine   : {engine}")
    logger.info(f"  Formula  : {formula_enable}")
    logger.info(f"  Table    : {table_enable}")
    logger.info(f"  Warmup   : {warmup}회")
    logger.info(f"  Runs     : {runs}회")
    logger.info("=" * 80)


def _run_sync_benchmark(args, images_with_extra_info, formula_enable, table_enable,
                        layout_config, ocr_config, formula_config, table_config):
    """Sync 파이프라인 워밍업 + 측정을 수행하고 (results, stats, elapsed_list)를 반환한다."""
    n_pages = len(images_with_extra_info)
    pipeline_args = (
        images_with_extra_info, formula_enable, table_enable,
        layout_config, ocr_config, formula_config, table_config,
    )
    logger.info(f"\n[Sync] 워밍업 ({args.warmup}회) ...")
    for _ in range(args.warmup):
        run_sync_pipeline(*pipeline_args)

    results_last, stats_last = None, {}
    elapsed_list = []
    logger.info(f"[Sync] 측정 ({args.runs}회) ...")
    for r in range(args.runs):
        results, stats, elapsed = run_sync_pipeline(*pipeline_args)
        elapsed_list.append(elapsed)
        results_last, stats_last = results, stats
        logger.info(f"  Run {r + 1}/{args.runs}: {elapsed:.3f}s ({n_pages / elapsed:.2f} it/s)")

    best = min(elapsed_list)
    logger.info(f"[Sync] Best: {best:.3f}s | Mean: {np.mean(elapsed_list):.3f}s")
    return results_last, stats_last, elapsed_list


def _run_async_benchmark(args, images_with_extra_info, formula_enable, table_enable,
                         layout_config, ocr_config, formula_config, table_config):
    """Async 파이프라인 워밍업 + 측정을 수행하고 (results, stats, elapsed_list)를 반환한다."""
    n_pages = len(images_with_extra_info)
    pipeline_args = (
        images_with_extra_info, formula_enable, table_enable,
        layout_config, ocr_config, formula_config, table_config,
    )
    logger.info(f"\n[Async] 워밍업 ({args.warmup}회) ...")
    for _ in range(args.warmup):
        run_async_pipeline(*pipeline_args)

    results_last, stats_last = None, {}
    elapsed_list = []
    logger.info(f"[Async] 측정 ({args.runs}회) ...")
    for r in range(args.runs):
        results, stats, elapsed = run_async_pipeline(*pipeline_args)
        elapsed_list.append(elapsed)
        results_last, stats_last = results, stats
        logger.info(f"  Run {r + 1}/{args.runs}: {elapsed:.3f}s ({n_pages / elapsed:.2f} it/s)")

    best = min(elapsed_list)
    logger.info(f"[Async] Best: {best:.3f}s | Mean: {np.mean(elapsed_list):.3f}s")
    return results_last, stats_last, elapsed_list


def _report(args, sync_data, async_data, n_pages):
    """비교 출력 + 정합성 검사 + JSON 저장을 수행한다."""
    def extract_stage_stats(pipeline_perf_stats: Dict) -> Dict:
        """pdf_perf_stats[0] 형식을 {stage: {time, count}} 로 변환한다."""
        pdf0 = pipeline_perf_stats.get(0, {})
        return {k: dict(v) for k, v in pdf0.items()}

    sync_results, sync_stats, sync_elapsed = sync_data
    async_results, async_stats, async_elapsed = async_data
    sync_best = min(sync_elapsed) if sync_elapsed else 0.0
    async_best = min(async_elapsed) if async_elapsed else 0.0

    if not args.skip_sync and not args.skip_async:
        sync_stage = extract_stage_stats(sync_stats)
        async_stage = extract_stage_stats(async_stats)
        print_comparison(
            sync_stage, async_stage, sync_best, async_best,
            n_pages, args.engine, args.runs,
        )
        if sync_results and async_results:
            check_output_consistency(sync_results, async_results)
        if args.output:
            save_report(sync_stage, async_stage, sync_best, async_best,
                        n_pages, args.engine, args.output)
    elif not args.skip_sync:
        logger.info(f"[Sync Only] Best={sync_best:.3f}s | {n_pages / max(sync_best, 1e-9):.2f} pages/sec")
    else:
        logger.info(f"[Async Only] Best={async_best:.3f}s | {n_pages / max(async_best, 1e-9):.2f} pages/sec")


# == 메인 ==


def main():
    parser = argparse.ArgumentParser(description="RapidDoc 파이프라인 Sync vs Async 벤치마크")
    parser.add_argument("--pdf", type=str, default=None,
                        help="입력 PDF 경로. 미지정 시 test_files/ 의 첫 번째 PDF 사용")
    parser.add_argument("--engine", type=str, choices=["dxengine", "onnxruntime"],
                        default="dxengine", help="추론 엔진 (기본값: dxengine)")
    parser.add_argument("--start-page", type=int, default=0, help="시작 페이지 ID")
    parser.add_argument("--end-page", type=int, default=None, help="종료 페이지 ID (None=전체)")
    parser.add_argument("--warmup", type=int, default=1, help="워밍업 실행 횟수")
    parser.add_argument("--runs", type=int, default=3, help="측정 실행 횟수")
    parser.add_argument("--no-formula", action="store_true", help="수식 인식 비활성화")
    parser.add_argument("--no-table", action="store_true", help="테이블 인식 비활성화")
    parser.add_argument("--skip-sync", action="store_true", help="Sync 파이프라인 건너뜀")
    parser.add_argument("--skip-async", action="store_true", help="Async 파이프라인 건너뜀")
    parser.add_argument("--output", type=str, default=None,
                        help="결과 JSON 저장 경로 (미지정 시 저장 안 함)")
    args = parser.parse_args()

    if args.engine == "dxengine":
        check_env()

    project_root = _ROOT
    pdf_path = _find_pdf(args.pdf, project_root)
    formula_enable = not args.no_formula
    table_enable = not args.no_table

    layout_config, ocr_config, formula_config, table_config = build_configs(
        args.engine, project_root, formula_enable, table_enable
    )
    images_with_extra_info = load_pages(pdf_path, args.start_page, args.end_page)
    n_pages = len(images_with_extra_info)
    _log_settings(pdf_path, n_pages, args.engine, formula_enable, table_enable,
                  args.warmup, args.runs)

    common_kwargs = {
        'formula_enable': formula_enable, 'table_enable': table_enable,
        'layout_config': layout_config, 'ocr_config': ocr_config,
        'formula_config': formula_config, 'table_config': table_config,
    }
    sync_data = (
        _run_sync_benchmark(args, images_with_extra_info, **common_kwargs)
        if not args.skip_sync
        else (None, {}, [])
    )
    if args.skip_sync:
        logger.info("[Sync] 건너뜀 (--skip-sync)")

    async_data = (
        _run_async_benchmark(args, images_with_extra_info, **common_kwargs)
        if not args.skip_async
        else (None, {}, [])
    )
    if args.skip_async:
        logger.info("[Async] 건너뜀 (--skip-async)")

    _report(args, sync_data, async_data, n_pages)


if __name__ == "__main__":
    main()
