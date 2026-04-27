# Copyright (c) Opendatalab. All rights reserved.
"""
RapidDoc demo for closed networks.
Parses PDFs using only locally stored ONNX models.

Usage:
    # 1. Run once via CLI
    python demo_offline.py

    # 2. Run the API server (stays alive)
    python demo/app_offline.py

    # Example curl call (server must be running):
    curl -X POST "http://localhost:8888/file_parse" \\
      -F "files=@demo/pdfs/example.pdf" \\
      -F "output_dir=./output-api" \\
      -F "formula_enable=true" \\
      -F "table_enable=true" \\
      -F "layout_engine=dxengine" \\
      -F "ocr_engine=dxengine" \\
      -F "return_md=true"

Notes:
    - This file (demo_offline.py): one-shot CLI script
    - app_offline.py: API server; keep one server up and send multiple requests
    - For detailed API usage, see demo/README_API_OFFLINE.md
"""
import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# 환경 변수 체크 (./deepx_scripts/set_env.sh 1 2 1 3 2 4 설정 필요)
# =============================================================================
required_env_vars = {
    'CUSTOM_INTER_OP_THREADS_COUNT': '1',
    'CUSTOM_INTRA_OP_THREADS_COUNT': '2',
    'DXRT_DYNAMIC_CPU_THREAD': '1',
    'DXRT_TASK_MAX_LOAD': '3',
    'NFH_INPUT_WORKER_THREADS': '2',
    'NFH_OUTPUT_WORKER_THREADS': '4'
}

missing_vars = []
incorrect_vars = []

for var_name, expected_value in required_env_vars.items():
    actual_value = os.environ.get(var_name)
    if actual_value is None:
        missing_vars.append(var_name)
    elif actual_value != expected_value:
        incorrect_vars.append(f"{var_name}={actual_value} (expected: {expected_value})")

if missing_vars or incorrect_vars:
    print("=" * 80)
    print("Error: required environment variables are not set correctly.")
    print("-" * 80)
    if missing_vars:
        print(f"Missing variables: {', '.join(missing_vars)}")
    if incorrect_vars:
        print(f"Variables with unexpected values: {', '.join(incorrect_vars)}")
    print("-" * 80)
    print("Please run the following command and try again:")
    print("  source ./deepx_scripts/set_env.sh 1 2 1 3 2 4")
    print("=" * 80)
    import sys
    sys.exit(1)

# =============================================================================
# 폐쇄망 설정: 모델 다운로드 차단
# =============================================================================
os.environ['MINERU_MODEL_SOURCE'] = 'local'  # 로컬 모델만 사용

# requests와 urllib를 차단하여 외부 다운로드 시도 완전 방지
import requests
import urllib.request
import urllib.error

_original_requests_get = requests.get
_original_urllib_urlopen = urllib.request.urlopen
_original_urllib_urlretrieve = urllib.request.urlretrieve

def _blocked_requests_get(*args, **kwargs):
    raise requests.exceptions.ConnectionError("Network access blocked in closed environment")

def _blocked_urllib_urlopen(*args, **kwargs):
    raise urllib.error.URLError("Network access blocked in closed environment")

def _blocked_urllib_urlretrieve(*args, **kwargs):
    raise urllib.error.URLError("Network access blocked in closed environment")

requests.get = _blocked_requests_get
urllib.request.urlopen = _blocked_urllib_urlopen
urllib.request.urlretrieve = _blocked_urllib_urlretrieve
# =============================================================================

from loguru import logger

from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType
from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType
from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
from rapid_doc.model.formula.rapid_formula_self.utils.typings import EngineType as FormulaEngineType
from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType


def _build_perf_summary_md(
    all_pdf_perf_stats: dict,
    pdf_file_names: list[str],
    wall_time: float,
    total_pages: int,
    pipeline_mode: str = "",
    model_load_times: dict = None,
) -> str:
    """Performance Summary를 마크다운 문자열로 생성한다."""
    stage_order = ['layout', 'formula', 'pdf_det', 'ocr_det', 'table', 'ocr_rec']
    stage_labels = {
        'layout':  'Layout',
        'formula': 'Formula',
        'pdf_det': 'PDF-det',
        'ocr_det': 'OCR-det',
        'table':   'Table',
        'ocr_rec': 'OCR-rec',
    }

    lines: list[str] = []
    title = f"{pipeline_mode} Performance Summary" if pipeline_mode else "Performance Summary"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if pipeline_mode:
        lines.append(f"- **Pipeline Mode**: {pipeline_mode}")
    lines.append(f"- **Total Files**: {len(pdf_file_names)}")
    lines.append(f"- **Total Pages**: {total_pages}")
    lines.append(f"- **Total Wall Time**: {wall_time:.2f} s")
    if total_pages > 0 and wall_time > 0:
        lines.append(f"- **Overall Throughput**: {total_pages / wall_time:.1f} pages/s")
    lines.append("")

    # Model Loading 섹션
    if model_load_times:
        total_load = sum(model_load_times.values())
        lines.append("## Model Loading")
        lines.append("")
        lines.append("| Model | Load Time |")
        lines.append("|:---|---:|")
        for name, elapsed in model_load_times.items():
            lines.append(f"| {name} | {elapsed:.2f} s |")
        lines.append(f"| **Total** | **{total_load:.2f} s** |")
        lines.append("")

    # 전체 집계 Pipeline Steps (Overall)
    agg: dict[str, dict] = {}
    for pdf_stats in all_pdf_perf_stats.values():
        for key, s in pdf_stats.items():
            if key not in agg:
                agg[key] = {'time': 0.0, 'count': 0}
            agg[key]['time'] += s['time']
            agg[key]['count'] += s['count']

    total_stage_time = sum(s['time'] for s in agg.values())

    lines.append("## Overall Pipeline Performance")
    lines.append("")
    lines.append("| Pipeline Step | Avg Latency | Throughput |")
    lines.append("|:---|---:|---:|")
    for key in stage_order:
        if key not in agg or agg[key]['time'] <= 0:
            continue
        s = agg[key]
        t, c = s['time'], s['count']
        avg_ms = (t / max(c, 1)) * 1000
        fps = c / max(t, 0.001)
        label = stage_labels.get(key, key)
        lines.append(f"| {label} | {avg_ms:.2f} ms | {fps:.1f} FPS |")
    lines.append("")
    lines.append(f"- **Total Stages**: {total_stage_time:.2f} s")
    lines.append("")

    # Per-Document Elapsed Time 테이블
    if len(all_pdf_perf_stats) > 1:
        lines.append("## Per-Document Elapsed Time")
        lines.append("")
        lines.append("| Document | Total Time (s) | Pages |")
        lines.append("|:---|---:|---:|")
        for pdf_idx in sorted(all_pdf_perf_stats.keys()):
            pdf_stats = all_pdf_perf_stats[pdf_idx]
            doc_total = sum(s['time'] for s in pdf_stats.values())
            page_count = max(
                (s['count'] for s in pdf_stats.values()), default=0
            )
            pdf_name = pdf_file_names[pdf_idx] if pdf_idx < len(pdf_file_names) else f"PDF #{pdf_idx}"
            lines.append(f"| {pdf_name} | {doc_total:.2f} | {page_count} |")
        lines.append("")

    # Per-PDF 상세 성능
    for pdf_idx in sorted(all_pdf_perf_stats.keys()):
        pdf_stats = all_pdf_perf_stats[pdf_idx]
        pdf_name = pdf_file_names[pdf_idx] if pdf_idx < len(pdf_file_names) else f"PDF #{pdf_idx}"
        total_pdf_stage_time = sum(s['time'] for s in pdf_stats.values())

        lines.append(f"## {pdf_name}")
        lines.append("")
        lines.append("| Pipeline Step | Avg Latency | Throughput |")
        lines.append("|:---|---:|---:|")

        for key in stage_order:
            if key not in pdf_stats or pdf_stats[key]['time'] <= 0:
                continue
            s = pdf_stats[key]
            t, c = s['time'], s['count']
            avg_ms = (t / max(c, 1)) * 1000
            fps = c / max(t, 0.001)
            label = stage_labels.get(key, key)
            lines.append(f"| {label} | {avg_ms:.2f} ms | {fps:.1f} FPS |")

        lines.append("")
        lines.append(f"- **Total Stage Time**: {total_pdf_stage_time:.2f} s")
        layout_count = pdf_stats.get('layout', {}).get('count', 0)
        if layout_count > 0:
            lines.append(f"- **Pages**: {layout_count}")
            lines.append(f"- **Avg per Page**: {total_pdf_stage_time / layout_count:.2f} s")
        lines.append("")

    return "\n".join(lines)


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing (model_type 명시로 해결)
    table_enable=True,  # Enable table parsing (UNET 모델 사용 - paddle_cls.onnx 불필요)
    # Engine 선택 플래그
    layout_engine="dxengine",  # "onnxruntime", "dxengine", "openvino"
    ocr_engine="dxengine",  # "onnxruntime", "dxengine", "openvino", "torch", "paddle"
    formula_engine="onnxruntime",  # "onnxruntime", "dxengine", "openvino"
    table_engine="dxengine",  # "onnxruntime", "torch"
    formula_rec_enable=True,  # False: skip ONNX formula inference, keep regions as images
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
    use_async_pipeline=True,  # Whether to use async pipeline for parallel processing
):
    # =========================================================================
    # 모델 경로 설정 (엔진별)
    # =========================================================================
    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent.absolute()
    onnx_models_dir = project_root / "onnx_models"
    dxnn_models_dir = project_root / "dxnn_models"  # DX Engine 모델 디렉토리
    
    # =========================================================================
    # Layout 모델 설정
    # =========================================================================
    layout_config = {
        "model_type": LayoutModelType.PP_DOCLAYOUT_L,
    }
    
    # Engine-specific Layout settings
    if layout_engine.lower() == "dxengine":
        layout_config["engine_type"] = LayoutEngineType.DXENGINE
        layout_config["model_dir_or_path"] = str(dxnn_models_dir / "pp_doclayout_l_part1.dxnn")
        layout_config["sub_model_path"] = str(onnx_models_dir / "pp_doclayout_l_part2.onnx")
        logger.info("Layout model: DX Engine")
    elif layout_engine.lower() == "openvino":
        layout_config["engine_type"] = LayoutEngineType.OPENVINO
        layout_config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")
        logger.info("Layout model: OpenVINO")
    else:  # onnxruntime (default)
        layout_config["engine_type"] = LayoutEngineType.ONNXRUNTIME
        layout_config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")
        logger.info("Layout model: ONNX Runtime")

    # =========================================================================
    # OCR 모델 설정
    # =========================================================================
    ocr_config = {}
    # Common OCR settings
    ocr_config.update({
        # Skip font path to prevent downloads in closed environments.
        # Fonts are only used for visualization and not required for OCR.
        # rapidocr may try to download fonts if missing, but urllib is blocked
        # above so it will error and proceed with default behavior.
        
        # Additional settings
        # "Rec.rec_batch_num": 1,
        "use_det_mode": parse_method,  # auto: PDF extraction first → OCR | txt: PDF only | ocr: always OCR
        "engine_type": "dxengine",
        
        # Default single model path (fallback - also needed when using multi-model)
        "Det.model_path": str(dxnn_models_dir / "det_v5_640_640.dxnn"),
        "Rec.model_path": str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
        
        "char_dict_path": str(project_root / "value_compare" / "recognition" / "character_dict_from_onnx.txt"),
        
        # Multi-model detection 설정 (DX OCR에서 사용)
        "use_multi_det_model": True,  # True로 설정 시 ratio 기반 multi-model detection 사용
        "Det.model_paths": {  # ratio별 모델 경로 (use_multi_det_model=True일 때만 사용)
            1: str(dxnn_models_dir / "det_v5_640_640.dxnn"),  # H/W ratio ~1.5 (기본)
            2: str(dxnn_models_dir / "det_v5_320_640.dxnn"),  # H/W ratio ~2.5
            4: str(dxnn_models_dir / "det_v5_160_640.dxnn"),  # H/W ratio ~ 7.5
            10: str(dxnn_models_dir / "det_v5_64_640.dxnn"),  # H/W ratio > 7.5
        },
        
        # Multi-model recognition 설정 (DX OCR에서 사용)
        "use_multi_rec_model": True,  # True로 설정 시 ratio 기반 multi-model 사용
        "Rec.model_paths": {  # ratio별 모델 경로 (use_multi_rec_model=True일 때만 사용)
            3: str(dxnn_models_dir / "rec_v5_ratio_3.dxnn"),
            5: str(dxnn_models_dir / "rec_v5_ratio_5.dxnn"),
            10: str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
            15: str(dxnn_models_dir / "rec_v5_ratio_15.dxnn"),
            25: str(dxnn_models_dir / "rec_v5_ratio_25.dxnn"),
            35: str(dxnn_models_dir / "rec_v5_ratio_35.dxnn"),
        },
        
        # Debug 이미지 저장 설정 (DX OCR에서 사용)
        "save_debug_images": False,  # True로 설정 시 detection input/output 저장
        "debug_save_dir": os.path.join(output_dir, "ocr_debug"),  # 저장 디렉토리
    })

    # =========================================================================
    # Formula 모델 설정
    # =========================================================================
    formula_config = {
        "model_type": FormulaModelType.PP_FORMULANET_PLUS_M,
    }
    if not formula_rec_enable:
        formula_config["formula_rec_enable"] = False
    
    # Engine-specific Formula settings
    if formula_engine.lower() == "dxengine":
        logger.error("=" * 80)
        logger.error("Error: Formula model is not supported by DX Engine.")
        logger.error("Supported engines: onnxruntime, openvino")
        logger.error("=" * 80)
        import sys
        sys.exit(1)
    elif formula_engine.lower() == "openvino":
        formula_config["engine_type"] = FormulaEngineType.OPENVINO
        formula_config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")
        logger.info("Formula model: OpenVINO")
    else:  # onnxruntime (default)
        formula_config["engine_type"] = FormulaEngineType.ONNXRUNTIME
        formula_config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")
        logger.info("Formula model: ONNX Runtime")

    # =========================================================================
    # Table model settings
    # =========================================================================
    table_config = {}
    
    # UNET_SLANET_PLUS: classify wired/wireless via paddle_cls, then process each
    # Currently wireless model also uses UNET DX Engine (can be replaced with slanet_plus ONNX later)
    table_config["model_type"] = TableModelType.UNET_SLANET_PLUS
    table_config["wireless_model_type"] = "unet"
    # cls.model_dir_or_path=None → TableCls auto-downloads from ModelScope (table_cls/models/)
    
    # Engine-specific Table settings
    if table_engine.lower() == "dxengine":
        table_config["engine_type"] = "dxengine"
        table_config["unet.model_dir_or_path"] = str(dxnn_models_dir / "unet.dxnn")
        table_config["wireless_engine_type"] = "dxengine"
        table_config["slanet_plus.model_dir_or_path"] = str(dxnn_models_dir / "unet.dxnn")
        logger.info("Table model: UNET_SLANET_PLUS (wired=DX, wireless=DX/UNET, cls=ONNX)")
    elif table_engine.lower() == "torch":
        table_config["engine_type"] = "torch"
        table_config["wireless_engine_type"] = "torch"
        logger.info("Table model: UNET_SLANET_PLUS (wired=Torch, wireless=Torch)")
    else:  # onnxruntime (default)
        table_config["engine_type"] = "onnxruntime"
        table_config["unet.model_dir_or_path"] = str(onnx_models_dir / "unet.onnx")
        table_config["wireless_engine_type"] = "onnxruntime"
        table_config["slanet_plus.model_dir_or_path"] = str(onnx_models_dir / "unet.onnx")
        logger.info("Table model: UNET_SLANET_PLUS (wired=ONNX, wireless=ONNX/UNET)")

    checkbox_config = {
        # 체크박스 인식 (OpenCV 기반, 오검출 가능성 있음)
        "checkbox_enable": False,
    }

    # 이미지 추출 설정
    image_config = {
        "extract_original_image": False,  # pypdfium2로 원본 이미지 추출
        "extract_original_image_iou_thresh": 0.5,  # IOU 임계값
    }
    # =========================================================================
    
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        pdf_bytes_list[idx] = new_pdf_bytes

    # =========================================================================
    # Measure model inference performance
    # =========================================================================
    logger.info("Model inference started")
    start_time = time.time()
    
    infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list, all_pdf_perf_stats, model_load_times = pipeline_doc_analyze(
        pdf_bytes_list, 
        parse_method=parse_method, 
        formula_enable=formula_enable,
        table_enable=table_enable,
        layout_config=layout_config, 
        ocr_config=ocr_config, 
        formula_config=formula_config, 
        table_config=table_config, 
        checkbox_config=checkbox_config,
        use_async_pipeline=use_async_pipeline,
    )
    
    wall_time = time.time() - start_time
    
    # Performance Summary를 마크다운 파일로 저장
    if all_pdf_perf_stats:
        total_pages = sum(
            s.get('layout', {}).get('count', 0)
            for s in all_pdf_perf_stats.values()
        )
        mode_labels = {False: 'sync', True: 'async', 'finegrained': 'finegrained'}
        pipeline_label = mode_labels.get(use_async_pipeline, str(use_async_pipeline))
        perf_md = _build_perf_summary_md(
            all_pdf_perf_stats, pdf_file_names, wall_time, total_pages, pipeline_label,
            model_load_times=model_load_times,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        perf_md_path = os.path.join(output_dir, f"performance_summary_{timestamp}.md")
        os.makedirs(output_dir, exist_ok=True)
        with open(perf_md_path, "w", encoding="utf-8") as f:
            f.write(perf_md)
        _perf_md_saved_path = perf_md_path
    else:
        _perf_md_saved_path = None

    if use_async_pipeline in (True, 'finegrained'):
        middle_ocr_config = {**(ocr_config or {}), 'use_async': True}
    else:
        middle_ocr_config = ocr_config

    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_dict = all_page_dicts[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_dict, image_writer, _lang, _ocr_enable, 
            formula_enable, ocr_config=middle_ocr_config, image_config=image_config
        )

        pdf_info = middle_json["pdf_info"]

        pdf_bytes = pdf_bytes_list[idx]
        if f_draw_layout_bbox:
            draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

        if f_draw_span_bbox:
            draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

        if f_dump_orig_pdf:
            md_writer.write(
                f"{pdf_file_name}_origin.pdf",
                pdf_bytes,
            )

        if f_dump_md:
            image_dir = str(os.path.basename(local_image_dir))
            md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}.md",
                md_content_str,
            )

        if f_dump_content_list:
            image_dir = str(os.path.basename(local_image_dir))
            content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}_content_list.json",
                json.dumps(content_list, ensure_ascii=False, indent=4),
            )

        if f_dump_middle_json:
            md_writer.write_string(
                f"{pdf_file_name}_middle.json",
                json.dumps(middle_json, ensure_ascii=False, indent=4),
            )

        if f_dump_model_output:
            md_writer.write_string(
                f"{pdf_file_name}_model.json",
                json.dumps(model_json, ensure_ascii=False, indent=4),
            )

        logger.info(f"local output dir is {local_md_dir}")

    return _perf_md_saved_path


def parse_doc(
        path_list: list[Path],
        output_dir,
        method="auto",
        formula_enable=True,
        table_enable=False,
        start_page_id=0,
        end_page_id=None,
        # Engine 선택
        layout_engine="dxengine",
        ocr_engine="dxengine",
        formula_engine="onnxruntime",
        table_engine="dxengine",
        use_async_pipeline=True,
        formula_rec_enable=True,
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
        formula_enable: Enable formula parsing, default is True
        table_enable: Enable table parsing, default is False
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
        return do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            parse_method=method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            layout_engine=layout_engine,
            ocr_engine=ocr_engine,
            formula_engine=formula_engine,
            table_engine=table_engine,
            formula_rec_enable=formula_rec_enable,
            use_async_pipeline=use_async_pipeline,
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    # =========================================================================
    # CLI 인자 파싱
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='RapidDoc PDF Parser - Offline Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # demo/pdfs/ 기본 디렉토리 사용
  python demo/demo_offline.py --finegrained

  # 특정 디렉토리의 모든 PDF
  python demo/demo_offline.py /path/to/pdf/folder --finegrained

  # 개별 PDF 파일 하나 또는 여러 개
  python demo/demo_offline.py file.pdf --no-async
  python demo/demo_offline.py a.pdf b.pdf c.pdf --finegrained
        """,
    )
    parser.add_argument(
        'input', nargs='*',
        help='PDF 파일 또는 디렉토리 경로 (생략 시 demo/pdfs/ 사용)',
        metavar='PATH',
    )
    pipeline_group = parser.add_mutually_exclusive_group()
    pipeline_group.add_argument(
        '--use-async', dest='pipeline_mode', action='store_const', const=True,
        help='Enable TrueAsyncPipeline (batch async mode)',
    )
    pipeline_group.add_argument(
        '--finegrained', dest='pipeline_mode', action='store_const', const='finegrained',
        help='Enable FinegrainedStreamingPipeline (7-stage per-page streaming) [default]',
    )
    pipeline_group.add_argument(
        '--no-async', dest='pipeline_mode', action='store_const', const=False,
        help='Disable async pipeline / sync mode',
    )
    parser.add_argument(
        '--output-dir', dest='output_dir', default=None,
        help='결과 저장 디렉토리 (기본: demo/output-offline-{mode}/)',
        metavar='DIR',
    )
    parser.add_argument(
        '--force-ocr', action='store_true', default=False,
        help='Ignore PDF text metadata and always use image→OCR path (for model evaluation)',
    )
    parser.add_argument(
        '--no-formula', action='store_true', default=False,
        help='Disable formula recognition (skip ONNX formula inference entirely)',
    )
    parser.set_defaults(pipeline_mode='finegrained')  # Default: finegrained (fastest pipeline)
    args = parser.parse_args()
    
    # =========================================================================
    # 모델 활성화 설정
    # =========================================================================
    FORMULA_ENABLE = not args.no_formula
    FORMULA_REC_ENABLE = not args.no_formula
    TABLE_ENABLE = True     # 표 인식 모델 사용 여부 (True/False) - UNET 모델 사용 (paddle_cls 불필요)
    
    # =========================================================================
    # 엔진 선택 설정
    # =========================================================================
    # 각 모델별로 사용할 엔진을 선택할 수 있습니다.
    # 
    # 지원 엔진:
    #   Layout  : "onnxruntime", "dxengine", "openvino"
    #   OCR     : "onnxruntime", "dxengine", "openvino", "torch", "paddle"
    #   Formula : "onnxruntime", "dxengine", "openvino"
    #   Table   : "onnxruntime", "dxengine", "torch"
    #
    # 주의사항:
    #   - dxengine 사용 시: dxnn_models/ 디렉토리에 .dxnn 파일 필요
    #   - onnxruntime 사용 시: onnx_models/ 디렉토리에 .onnx 파일 필요
    #   - openvino 사용 시: openvino 패키지 설치 필요
    #
    # 테이블 인식 관련:
    #   - 현재 ModelType.UNET 사용 (paddle_cls.onnx 불필요)
    #   - unet.onnx 모델만 있으면 됨 (유선 테이블 전용)
    #   - 무선 테이블도 인식하려면 paddle_cls.onnx + slanet_plus.onnx 필요
    # =========================================================================
    
    LAYOUT_ENGINE = "dxengine"   # Layout 모델 엔진
    OCR_ENGINE = "dxengine"      # OCR 모델 엔진
    FORMULA_ENGINE = "onnxruntime"  # Formula 모델 엔진
    TABLE_ENGINE = "dxengine"    # Table 모델 엔진
    
    # DX Engine 사용 예시 (주석 해제하여 사용)
    # LAYOUT_ENGINE = "dxengine"
    # OCR_ENGINE = "dxengine"
    # FORMULA_ENGINE = "dxengine"
    # TABLE_ENGINE = "dxengine"
    # =========================================================================
    
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]
    valid_suffixes = pdf_suffixes + image_suffixes

    # =========================================================================
    # 입력 경로 처리: 파일/디렉토리 혼합 지원
    # =========================================================================
    doc_path_list = []
    if args.input:
        for raw in args.input:
            p = Path(raw).expanduser().resolve()
            if p.is_dir():
                doc_path_list.extend(
                    child for child in sorted(p.glob('*')) if child.suffix in valid_suffixes
                )
            elif p.is_file() and p.suffix in valid_suffixes:
                doc_path_list.append(p)
            else:
                logger.warning(f"건너뜀 (파일 없음 또는 지원하지 않는 형식): {raw}")
    else:
        # 기본: demo/pdfs/ 디렉토리
        default_dir = Path(__dir__) / "pdfs"
        doc_path_list = sorted(
            p for p in default_dir.glob('*') if p.suffix in valid_suffixes
        )

    if not doc_path_list:
        logger.error("처리할 파일이 없습니다. 경로를 확인해 주세요.")
        import sys; sys.exit(1)

    # =========================================================================
    # 출력 디렉토리
    # =========================================================================
    _output_suffix = {False: 'no_async', True: 'async', 'finegrained': 'finegrained'}
    if args.output_dir:
        output_dir = str(Path(args.output_dir).expanduser().resolve())
    else:
        output_dir = os.path.join(__dir__, f"output-offline-{_output_suffix.get(args.pipeline_mode, str(args.pipeline_mode))}")

    logger.info("=" * 80)
    logger.info(f"Running in closed-network mode: processing {len(doc_path_list)} files")
    logger.info(f"Formula recognition: {'enabled' if FORMULA_ENABLE else 'disabled'}"
                + ("" if FORMULA_REC_ENABLE else " (rec disabled — image only)"))
    logger.info(f"Table recognition: {'enabled' if TABLE_ENABLE else 'disabled'}")
    logger.info(f"Parse method: {'ocr (force-ocr, no PDF metadata)' if args.force_ocr else 'auto'}")
    _mode_label = {False: 'sync', True: 'async (TrueAsyncPipeline)', 'finegrained': 'finegrained (7-stage streaming)'}
    logger.info(f"Pipeline mode: {_mode_label.get(args.pipeline_mode, str(args.pipeline_mode))}")
    logger.info(f"Output dir   : {output_dir}")
    logger.info("-" * 80)
    logger.info("Engine configuration:")
    logger.info(f"  Layout  Engine: {LAYOUT_ENGINE}")
    logger.info(f"  OCR     Engine: {OCR_ENGINE}")
    logger.info(f"  Formula Engine: {FORMULA_ENGINE}")
    logger.info(f"  Table   Engine: {TABLE_ENGINE}")
    logger.info("=" * 80)
    
    perf_md_path = parse_doc(
        doc_path_list,
        output_dir,
        method="ocr" if args.force_ocr else "auto",
        formula_enable=FORMULA_ENABLE,
        table_enable=TABLE_ENABLE,
        layout_engine=LAYOUT_ENGINE,
        ocr_engine=OCR_ENGINE,
        formula_engine=FORMULA_ENGINE,
        table_engine=TABLE_ENGINE,
        formula_rec_enable=FORMULA_REC_ENABLE,
        use_async_pipeline=args.pipeline_mode,
    )
    if perf_md_path:
        logger.info("=" * 80)
        logger.info(f"Performance summary saved: {perf_md_path}")
        logger.info("=" * 80)
