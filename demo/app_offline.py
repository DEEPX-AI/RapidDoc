# Copyright (c) Opendatalab. All rights reserved.
"""
RapidDoc FastAPI server for closed networks.
Provides the same configuration as demo_offline.py via FastAPI.

Usage:
    source venv/bin/activate
    python demo/app_offline.py

    or

    uvicorn demo.app_offline:app --host 0.0.0.0 --port 8888
"""
import argparse
import gc
import json
import os
import sys
import traceback
import tempfile
import shutil
from glob import glob
from base64 import b64encode, b64decode
from pathlib import Path
from typing import Optional, List, Dict, Any
import io

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

# =============================================================================
# 환경 변수 체크 (deepx_scripts/set_env.sh 1 2 1 3 2 4 설정 필요)
# =============================================================================
def check_environment():
    """Check that deepx_scripts/set_env.sh 1 2 1 3 2 4 was sourced and required env vars are set.
    DXRT_TASK_MAX_LOAD is checked for presence only (any value accepted).
    """
    required_env_vars = {
        'CUSTOM_INTER_OP_THREADS_COUNT': '1',
        'CUSTOM_INTRA_OP_THREADS_COUNT': '2',
        'DXRT_DYNAMIC_CPU_THREAD': '1',
        'DXRT_TASK_MAX_LOAD': None,  # presence-only (any value accepted)
        'NFH_INPUT_WORKER_THREADS': '2',
        'NFH_OUTPUT_WORKER_THREADS': '4'
    }
    
    missing_vars = []
    incorrect_vars = []
    
    for var_name, expected_value in required_env_vars.items():
        actual_value = os.environ.get(var_name)
        if actual_value is None:
            missing_vars.append(var_name)
        elif expected_value is not None and actual_value != expected_value:
            incorrect_vars.append(f"{var_name}={actual_value} (expected: {expected_value})")
    
    if missing_vars or incorrect_vars:
        logger.error("=" * 80)
        logger.error("❌ Error: required environment variables are not set correctly.")
        logger.error("-" * 80)
        if missing_vars:
            logger.error(f"Missing variables: {', '.join(missing_vars)}")
        if incorrect_vars:
            logger.error(f"Variables with unexpected values: {', '.join(incorrect_vars)}")
        logger.error("-" * 80)
        logger.error("Please run the following command and try again:")
        logger.error("  source ./deepx_scripts/set_env.sh 1 2 1 3 2 4")
        logger.error("=" * 80)
        logger.error("")
        logger.error("Exiting.")
        sys.exit(1)
    else:
        logger.info("=" * 80)
        logger.info("✓ Environment variables verified")
        logger.info("-" * 80)
        for var_name, expected_value in required_env_vars.items():
            actual = os.environ.get(var_name)
            label = "any" if expected_value is None else expected_value
            logger.info(f"  {var_name}={actual} (expected: {label})")
        logger.info("=" * 80)
    
    return True

# 환경 변수 체크 실행
env_check_passed = check_environment()

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

from rapid_doc.cli.common import aio_do_parse
from rapid_doc.utils.language import remove_invalid_surrogates
from rapid_doc.version import __version__

# 프로젝트 루트 디렉토리
__dir__ = os.path.dirname(os.path.abspath(__file__))
project_root = Path(__dir__).parent.absolute()
onnx_models_dir = project_root / "onnx_models"
dxnn_models_dir = project_root / "dxnn_models"

# =============================================================================
# 기본 설정 (demo_offline.py와 동일)
# =============================================================================
DEFAULT_FORMULA_ENABLE = True
DEFAULT_TABLE_ENABLE = True
DEFAULT_DEEPX = True  # default: use DX Engine (NPU) since this server targets DEEPX env

# Pipeline performance defaults (aligned with demo_offline.py)
#   - 'finegrained': 7-stage per-page streaming pipeline (fastest, default)
#   - True         : AsyncPipelineRapidDoc (legacy async)
#   - False        : Synchronous batch processing
DEFAULT_PIPELINE_MODE = 'finegrained'

# Hybrid device partitioning (auto-enabled when 2+ NPU devices are available)
def _autodetect_hybrid_default() -> bool:
    try:
        from rapid_doc.utils.device_utils import get_dxnn_devices
        return len(get_dxnn_devices()) >= 2
    except Exception:
        return False

DEFAULT_HYBRID = _autodetect_hybrid_default()

# deepx=False (ONNX)
DEFAULT_LAYOUT_ENGINE = "onnxruntime"
DEFAULT_OCR_ENGINE = "onnxruntime"
DEFAULT_FORMULA_ENGINE = "onnxruntime"
DEFAULT_TABLE_ENGINE = "onnxruntime"

# deepx=True (DX Engine)
DEEPX_LAYOUT_ENGINE = "dxengine"
DEEPX_OCR_ENGINE = "dxengine"
DEEPX_FORMULA_ENGINE = "onnxruntime"  # Formula는 항상 ONNX
DEEPX_TABLE_ENGINE = "dxengine"

# 지원되는 파일 확장자
pdf_suffixes = [".pdf"]
image_suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

app = FastAPI(
    title="RapidDoc Offline API",
    description="RapidDoc API for closed environments - supports ONNX/DX Engine",
    version=__version__
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/health")
async def health_check():
    env_vars = {
        "CUSTOM_INTER_OP_THREADS_COUNT": os.environ.get("CUSTOM_INTER_OP_THREADS_COUNT"),
        "CUSTOM_INTRA_OP_THREADS_COUNT": os.environ.get("CUSTOM_INTRA_OP_THREADS_COUNT"),
        "DXRT_DYNAMIC_CPU_THREAD": os.environ.get("DXRT_DYNAMIC_CPU_THREAD"),
        "DXRT_TASK_MAX_LOAD": os.environ.get("DXRT_TASK_MAX_LOAD"),
        "NFH_INPUT_WORKER_THREADS": os.environ.get("NFH_INPUT_WORKER_THREADS"),
        "NFH_OUTPUT_WORKER_THREADS": os.environ.get("NFH_OUTPUT_WORKER_THREADS"),
    }
    
    return {
        "status": "healthy",
        "version": __version__,
        "api": "RapidDoc Offline API",
        "mode": "closed_environment",
        "environment_check": env_check_passed,
        "environment_variables": env_vars,
        "pipeline": {
            "mode": DEFAULT_PIPELINE_MODE,
            "hybrid": DEFAULT_HYBRID,
        },
        "default_engines": {
            "deepx": DEFAULT_DEEPX,
            "layout": DEFAULT_LAYOUT_ENGINE,
            "ocr": DEFAULT_OCR_ENGINE,
            "formula": DEFAULT_FORMULA_ENGINE,
            "table": DEFAULT_TABLE_ENGINE,
        },
        "deepx_engines": {
            "layout": DEEPX_LAYOUT_ENGINE,
            "ocr": DEEPX_OCR_ENGINE,
            "formula": DEEPX_FORMULA_ENGINE,
            "table": DEEPX_TABLE_ENGINE,
        }
    }


def get_default_layout_config(engine: str = DEFAULT_LAYOUT_ENGINE) -> dict:
    """Default layout model configuration"""
    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
    from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType
    
    config = {
        "model_type": LayoutModelType.PP_DOCLAYOUT_L,
    }
    
    if engine.lower() == "dxengine":
        config["engine_type"] = LayoutEngineType.DXENGINE
        config["model_dir_or_path"] = str(dxnn_models_dir / "pp_doclayout_l_part1.dxnn")
        config["sub_model_path"] = str(onnx_models_dir / "pp_doclayout_l_part2.onnx")
    elif engine.lower() == "openvino":
        config["engine_type"] = LayoutEngineType.OPENVINO
        config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")
    else:  # onnxruntime
        config["engine_type"] = LayoutEngineType.ONNXRUNTIME
        config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")
    
    return config


def get_default_ocr_config(engine: str = DEFAULT_OCR_ENGINE) -> dict:
    """Default OCR model configuration"""
    from rapidocr import EngineType as OCREngineType
    
    config = {}
    
    if engine.lower() == "dxengine":
        config["engine_type"] = "dxengine"
        config["Det.model_path"] = str(dxnn_models_dir / "ch_PP-OCRv5_server_det.dxnn")
        config["Rec.model_path"] = str(dxnn_models_dir / "ch_PP-OCRv5_rec_server_infer.dxnn")
        config["char_dict_path"] = str(project_root / "value_compare" / "recognition" / "character_dict_from_onnx.txt")
    elif engine.lower() == "openvino":
        config["Det.engine_type"] = OCREngineType.OPENVINO
        config["Rec.engine_type"] = OCREngineType.OPENVINO
        config["Det.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_server_det.onnx")
        config["Rec.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_rec_server_infer.onnx")
    elif engine.lower() == "torch":
        config["Det.engine_type"] = OCREngineType.TORCH
        config["Rec.engine_type"] = OCREngineType.TORCH
        config["Det.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_server_det.onnx")
        config["Rec.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_rec_server_infer.onnx")
    elif engine.lower() == "paddle":
        config["Det.engine_type"] = OCREngineType.PADDLE
        config["Rec.engine_type"] = OCREngineType.PADDLE
    else:  # onnxruntime
        config["Det.engine_type"] = OCREngineType.ONNXRUNTIME
        config["Rec.engine_type"] = OCREngineType.ONNXRUNTIME
        config["Det.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_server_det.onnx")
        config["Rec.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_rec_server_infer.onnx")
    
    # OCR 공통 설정
    config.update({
        "use_det_mode": 'auto',
        "use_multi_det_model": True,
        "Det.model_paths": {
            1: str(dxnn_models_dir / "det_v5_640_640.dxnn"),
            2: str(dxnn_models_dir / "det_v5_320_640.dxnn"),
            4: str(dxnn_models_dir / "det_v5_160_640.dxnn"),
            10: str(dxnn_models_dir / "det_v5_64_640.dxnn"),
        },
        "use_multi_rec_model": True,
        "Rec.model_paths": {
            3: str(dxnn_models_dir / "rec_v5_ratio_3.dxnn"),
            5: str(dxnn_models_dir / "rec_v5_ratio_5.dxnn"),
            10: str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
            15: str(dxnn_models_dir / "rec_v5_ratio_15.dxnn"),
            25: str(dxnn_models_dir / "rec_v5_ratio_25.dxnn"),
            35: str(dxnn_models_dir / "rec_v5_ratio_35.dxnn"),
        },
    })
    
    return config


def get_default_formula_config(engine: str = DEFAULT_FORMULA_ENGINE) -> dict:
    """Default formula model configuration"""
    from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
    from rapid_doc.model.formula.rapid_formula_self.utils.typings import EngineType as FormulaEngineType
    
    config = {
        "model_type": FormulaModelType.PP_FORMULANET_PLUS_M,
    }
    
    if engine.lower() == "dxengine":
        config["engine_type"] = "dxengine"
        config["model_dir_or_path"] = str(dxnn_models_dir / "pp_formulanet_plus_l.dxnn")
    elif engine.lower() == "openvino":
        config["engine_type"] = FormulaEngineType.OPENVINO
        config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")
    else:  # onnxruntime
        config["engine_type"] = FormulaEngineType.ONNXRUNTIME
        config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")
    
    return config


def get_default_table_config(engine: str = DEFAULT_TABLE_ENGINE) -> dict:
    """Default table model configuration"""
    from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType
    
    config = {
        "model_type": TableModelType.UNET,
    }
    
    if engine.lower() == "dxengine":
        config["engine_type"] = "dxengine"
        config["unet.model_dir_or_path"] = str(dxnn_models_dir / "unet.dxnn")
    elif engine.lower() == "torch":
        config["engine_type"] = "torch"
    else:  # onnxruntime
        config["engine_type"] = "onnxruntime"
        config["unet.model_dir_or_path"] = str(onnx_models_dir / "unet.onnx")
    
    return config


def get_infer_result(file_suffix_identifier: str, file_name: str, parse_dir: str) -> Optional[str]:
    """Read inference result file; searches nested directories when needed."""
    # Try the official path format first
    result_file_path = os.path.join(parse_dir, f"{file_name}{file_suffix_identifier}")
    logger.info(f"Looking for result file: {result_file_path}")
    
    if os.path.exists(result_file_path):
        logger.info(f"Found result file: {result_file_path}")
        try:
            with open(result_file_path, "r", encoding="utf-8") as fp:
                content = fp.read()
                logger.info(f"Read {len(content)} characters from {result_file_path}")
                return content
        except Exception as e:
            logger.error(f"Error reading file {result_file_path}: {e}")
            return None
    
    # If the official path is missing, search recursively
    logger.warning(f"Result file not found at official path: {result_file_path}")
    logger.info("Searching recursively in subdirectories...")
    
    found_files = []
    for root, dirs, files in os.walk(parse_dir):
        for file in files:
            if file.endswith(file_suffix_identifier):
                full_path = os.path.join(root, file)
                found_files.append(full_path)
                logger.info(f"Found potential result file: {full_path}")
    
    if found_files:
        # Prefer files whose names include the target stem
        for file_path in found_files:
            if file_name in file_path:
                logger.info(f"Selected result file: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as fp:
                        content = fp.read()
                        logger.info(f"Read {len(content)} characters from {file_path}")
                        return content
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    continue
        
        # If none match the stem, use the first found file
        if found_files:
            selected_file = found_files[0]
            logger.info(f"Using first found file: {selected_file}")
            try:
                with open(selected_file, "r", encoding="utf-8") as fp:
                    content = fp.read()
                    logger.info(f"Read {len(content)} characters from {selected_file}")
                    return content
            except Exception as e:
                logger.error(f"Error reading file {selected_file}: {e}")
                return None
    
    logger.warning(f"No result file found for pattern: {file_suffix_identifier}")
    return None


def encode_image(image_path: str) -> str:
    """Base64-encode an image file."""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


# =============================================================================
# Vision API 모델
# =============================================================================

class ImageSource(BaseModel):
    """Image source"""
    imageUri: Optional[str] = Field(None, description="Cloud storage URI (gs://...) or HTTP URL")
    
class ImageContent(BaseModel):
    """Image content"""
    content: Optional[str] = Field(None, description="Base64 encoded image")
    
class Image(BaseModel):
    """Image"""
    source: Optional[ImageSource] = None
    content: Optional[str] = Field(None, description="Base64 encoded image")

class Feature(BaseModel):
    """Requested feature"""
    type: str = Field(..., description="Feature type: TEXT_DETECTION, DOCUMENT_TEXT_DETECTION, etc.")
    maxResults: Optional[int] = Field(None, description="Maximum number of results")

class AnnotateImageRequest(BaseModel):
    """Annotate image request"""
    image: Image
    features: List[Feature]
    deepx: Optional[bool] = Field(None, description="Use DeepX engine (rec/det/unet/layout)")

class AnnotateImageRequests(BaseModel):
    """List of annotate image requests"""
    requests: List[AnnotateImageRequest]
    deepx: Optional[bool] = Field(None, description="Use DeepX engine for all requests (rec/det/unet/layout)")


# =============================================================================
# 기존 함수들
# =============================================================================


@app.post(
    "/file_parse",
    tags=["projects"],
    summary="Parse files using RapidDoc (ONNX/DX Engine)",
)
async def file_parse(
    files: List[UploadFile] = File(...),
    output_dir: str = Form("./output-offline"),
    clear_output_file: bool = Form(False),
    lang_list: List[str] = Form(["ch"]),
    backend: str = Form("pipeline"),
    parse_method: str = Form("auto"),
    formula_enable: bool = Form(DEFAULT_FORMULA_ENABLE),
    table_enable: bool = Form(DEFAULT_TABLE_ENABLE),
    deepx: Optional[bool] = Form(None),
    layout_engine: Optional[str] = Form(None),
    ocr_engine: Optional[str] = Form(None),
    formula_engine: Optional[str] = Form(None),
    table_engine: Optional[str] = Form(None),
    return_md: bool = Form(True),
    return_middle_json: bool = Form(False),
    return_model_output: bool = Form(False),
    return_content_list: bool = Form(False),
    return_images: bool = Form(False),
    start_page_id: int = Form(0),
    end_page_id: int = Form(99999),
):
    """
    Parse files using RapidDoc (ONNX/DX Engine)
    
    Args:
        files: Files to parse (PDF, images)
        output_dir: Output directory
        lang_list: Languages to parse (e.g., ['ch', 'en'])
        backend: Parsing backend (pipeline)
        parse_method: Parsing method (auto, ocr, txt)
        formula_enable: Enable formula parsing
        table_enable: Enable table parsing
        deepx: Use DX Engine (True: rec/det/unet/layout via dxengine, False: onnxruntime)
        layout_engine: Layout engine (None inherits deepx selection)
        ocr_engine: OCR engine (None inherits deepx selection)
        formula_engine: Formula engine (None inherits deepx selection)
        table_engine: Table engine (None inherits deepx selection)
        return_md: Return Markdown content
        return_middle_json: Return middle JSON
        return_model_output: Return model outputs
        return_content_list: Return content list
        return_images: Return images as base64
        start_page_id: Start page ID
        end_page_id: End page ID
    """
    try:
        # 백엔드 유형 검증
        supported_backends = ["pipeline"]
        if backend not in supported_backends:
            return JSONResponse(
                content={"error": f"Unsupported backend: {backend}. Supported: {supported_backends}"},
                status_code=400,
            )

        # 파일 검증
        if not files:
            return JSONResponse(
                content={"error": "No files provided"},
                status_code=400,
            )

        # Resolve deepx default if not provided
        deepx = DEFAULT_DEEPX if deepx is None else deepx

        # deepx 옵션에 따라 엔진 설정
        if layout_engine is None:
            layout_engine = DEEPX_LAYOUT_ENGINE if deepx else DEFAULT_LAYOUT_ENGINE
        if ocr_engine is None:
            ocr_engine = DEEPX_OCR_ENGINE if deepx else DEFAULT_OCR_ENGINE
        if formula_engine is None:
            formula_engine = DEEPX_FORMULA_ENGINE if deepx else DEFAULT_FORMULA_ENGINE
        if table_engine is None:
            table_engine = DEEPX_TABLE_ENGINE if deepx else DEFAULT_TABLE_ENGINE

        # 기본 설정 생성
        layout_config = get_default_layout_config(layout_engine)
        ocr_config = get_default_ocr_config(ocr_engine)
        formula_config = get_default_formula_config(formula_engine)
        table_config = get_default_table_config(table_engine)
        checkbox_config = {"checkbox_enable": False}
        image_config = {
            "extract_original_image": False,
            "extract_original_image_iou_thresh": 0.5,
        }

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info(f"Running in closed-network mode: processing {len(files)} files")
        logger.info(f"DeepX mode: {'enabled' if deepx else 'disabled'}")
        logger.info(f"Formula recognition: {'enabled' if formula_enable else 'disabled'}")
        logger.info(f"Table recognition: {'enabled' if table_enable else 'disabled'}")
        logger.info("-" * 80)
        logger.info("Engine configuration:")
        logger.info(f"  Layout  Engine: {layout_engine}")
        logger.info(f"  OCR     Engine: {ocr_engine}")
        logger.info(f"  Formula Engine: {formula_engine}")
        logger.info(f"  Table   Engine: {table_engine}")
        logger.info("=" * 80)
        
        results = []
        
        for file in files:
            # 파일 확장자 검증
            file_suffix = Path(file.filename).suffix.lower()
            if file_suffix not in pdf_suffixes + image_suffixes:
                return JSONResponse(
                    content={"error": f"File type {file_suffix} is not supported for {file.filename}"},
                    status_code=400,
                )
            
            # 파일 내용 읽기
            content = await file.read()
            file_name = Path(file.filename).stem
            file_basename = file.filename
            
            if file_suffix in image_suffixes:
                from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes
                content = images_bytes_to_pdf_bytes(content)
            
            # 공식 API의 aio_do_parse 함수 사용
            try:
                logger.info(f"Starting to parse {file.filename} using backend {backend}")
                logger.info(f"Output directory: {output_dir}")
                logger.info(f"File name stem: {file_name}")
                
                await aio_do_parse(
                    output_dir=output_dir,
                    pdf_file_names=[file.filename],
                    pdf_bytes_list=[content],
                    p_lang_list=lang_list,
                    backend=backend,
                    parse_method=parse_method,
                    formula_enable=formula_enable,
                    table_enable=table_enable,
                    start_page_id=start_page_id,
                    end_page_id=end_page_id,
                    layout_config=layout_config,
                    ocr_config=ocr_config,
                    formula_config=formula_config,
                    table_config=table_config,
                    checkbox_config=checkbox_config,
                    image_config=image_config,
                    use_async_pipeline=DEFAULT_PIPELINE_MODE,
                    hybrid=DEFAULT_HYBRID,
                )
                
                logger.info(f"Parse completed for {file.filename}")
                
                # 결과 수집
                file_result = {"filename": file.filename}
                logger.info(f"Collecting results for {file.filename}")
                
                if return_md:
                    md_content = get_infer_result(".md", file_basename, output_dir)
                    file_result["md_content"] = md_content
                
                if return_middle_json:
                    middle_json_content = get_infer_result("_middle.json", file_basename, output_dir)
                    if middle_json_content:
                        file_result["middle_json"] = json.loads(middle_json_content)
                
                if return_model_output:
                    model_json_content = get_infer_result("_model.json", file_basename, output_dir)
                    if model_json_content:
                        file_result["model_output"] = json.loads(model_json_content)
                
                if return_content_list:
                    content_list_content = get_infer_result("_content_list.json", file_basename, output_dir)
                    if content_list_content:
                        file_result["content_list"] = json.loads(content_list_content)
                
                if return_images:
                    # 출력 디렉토리에서 이미지 파일 찾기 - 다양한 디렉토리 구조 지원
                    image_dirs = [
                        os.path.join(output_dir, file_basename, "images"),
                        os.path.join(output_dir, file_basename, "auto", "images"),
                    ]
                    
                    found_images = {}
                    for image_dir in image_dirs:
                        if os.path.exists(image_dir):
                            logger.info(f"Found image directory: {image_dir}")
                            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]:
                                image_paths = glob(os.path.join(image_dir, ext))
                                for image_path in image_paths:
                                    image_name = os.path.basename(image_path)
                                    if image_name not in found_images:
                                        found_images[image_name] = f"data:image/jpeg;base64,{encode_image(image_path)}"
                                        logger.info(f"Added image: {image_name}")
                    
                    file_result["images"] = found_images
                    logger.info(f"Total images found: {len(found_images)}")
                
                file_result["backend"] = backend
                file_result["deepx"] = deepx
                file_result["engines"] = {
                    "layout": layout_engine,
                    "ocr": ocr_engine,
                    "formula": formula_engine,
                    "table": table_engine,
                }
                results.append(file_result)
                
            except Exception as parse_error:
                tb_str = traceback.format_exc()
                logger.error(f"Error parsing {file.filename}:\n{tb_str}")
                results.append({
                    "filename": file.filename,
                    "error": str(parse_error)
                })
            
            # 파일 정리
            if clear_output_file:
                shutil.rmtree(os.path.join(output_dir, file_basename), ignore_errors=True)

        # 결과 반환
        response_data = {
            "results": results,
            "total_files": len(files),
            "successful_files": len([r for r in results if "error" not in r]),
            "mode": "closed_environment",
            "deepx": deepx,
            "engines_used": {
                "layout": layout_engine,
                "ocr": ocr_engine,
                "formula": formula_engine,
                "table": table_engine,
            }
        }
        
        return JSONResponse(response_data, status_code=200)

    except Exception as e:
        error_message = remove_invalid_surrogates(str(e))
        logger.error(f"API error: {error_message}")
        return JSONResponse(content={"error": error_message}, status_code=500)
    finally:
        gc.collect()


@app.post(
    "/v1/images:annotate",
    tags=["vision-api"],
    summary="Vision API compatible endpoint",
)
async def images_annotate(request_body: AnnotateImageRequests):
    """
        Vision API-compatible image annotation endpoint.
    
        Supported feature types:
        - TEXT_DETECTION: OCR text detection
        - DOCUMENT_TEXT_DETECTION: Document text detection (layout, formula, table included)
    
        Request body example:
        {
            "requests": [
                {
                    "image": {
                        "content": "base64_encoded_image_string",
                        // or
                        "source": {
                            "imageUri": "http://example.com/image.jpg"
                        }
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
    """
    try:
        responses = []
        
        for idx, req in enumerate(request_body.requests):
            try:
                # 이미지 데이터 추출
                image_bytes = None
                image_name = f"image_{idx}.png"
                
                # base64 content가 있는 경우
                if req.image.content:
                    image_bytes = b64decode(req.image.content)
                # imageUri가 있는 경우
                elif req.image.source and req.image.source.imageUri:
                    uri = req.image.source.imageUri
                    # HTTP/HTTPS URL 처리
                    if uri.startswith("http://") or uri.startswith("https://"):
                        # 폐쇄망 환경이므로 에러 반환
                        responses.append({
                            "error": {
                                "code": 403,
                                "message": "Network access blocked in closed environment. Please use base64 content instead.",
                                "status": "PERMISSION_DENIED"
                            }
                        })
                        continue
                    # 로컬 파일 경로 처리
                    elif os.path.exists(uri):
                        with open(uri, "rb") as f:
                            image_bytes = f.read()
                        image_name = os.path.basename(uri)
                    else:
                        responses.append({
                            "error": {
                                "code": 400,
                                "message": f"Invalid imageUri: {uri}",
                                "status": "INVALID_ARGUMENT"
                            }
                        })
                        continue
                else:
                    responses.append({
                        "error": {
                            "code": 400,
                            "message": "Either image.content or image.source.imageUri must be provided",
                            "status": "INVALID_ARGUMENT"
                        }
                    })
                    continue
                
                # Feature 타입 확인
                features = req.features
                is_full_document = any(f.type == "DOCUMENT_TEXT_DETECTION" for f in features)
                is_text_detection = any(f.type == "TEXT_DETECTION" for f in features)
                
                if not is_full_document and not is_text_detection:
                    responses.append({
                        "error": {
                            "code": 400,
                            "message": f"Unsupported feature types. Supported: TEXT_DETECTION, DOCUMENT_TEXT_DETECTION",
                            "status": "INVALID_ARGUMENT"
                        }
                    })
                    continue
                
                # 임시 출력 디렉토리 생성
                with tempfile.TemporaryDirectory() as temp_output_dir:
                    # 이미지를 PDF로 변환
                    from rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes
                    pdf_bytes = images_bytes_to_pdf_bytes(image_bytes)
                    
                    # deepx 옵션 결정 (요청별 > 전체 요청 > 기본값)
                    use_deepx = req.deepx if req.deepx is not None else (
                        request_body.deepx if request_body.deepx is not None else DEFAULT_DEEPX
                    )
                    
                    # deepx에 따라 엔진 선택
                    layout_engine = DEEPX_LAYOUT_ENGINE if use_deepx else DEFAULT_LAYOUT_ENGINE
                    ocr_engine = DEEPX_OCR_ENGINE if use_deepx else DEFAULT_OCR_ENGINE
                    formula_engine = DEEPX_FORMULA_ENGINE if use_deepx else DEFAULT_FORMULA_ENGINE
                    table_engine = DEEPX_TABLE_ENGINE if use_deepx else DEFAULT_TABLE_ENGINE
                    
                    # 엔진 설정
                    layout_config = get_default_layout_config(layout_engine)
                    ocr_config = get_default_ocr_config(ocr_engine)
                    formula_config = get_default_formula_config(formula_engine)
                    table_config = get_default_table_config(table_engine)
                    checkbox_config = {"checkbox_enable": False}
                    image_config = {
                        "extract_original_image": False,
                        "extract_original_image_iou_thresh": 0.5,
                    }
                    
                    # 파싱 실행
                    await aio_do_parse(
                        output_dir=temp_output_dir,
                        pdf_file_names=[image_name],
                        pdf_bytes_list=[pdf_bytes],
                        p_lang_list=["ch", "en"],
                        backend="pipeline",
                        parse_method="auto",
                        formula_enable=is_full_document,
                        table_enable=is_full_document,
                        start_page_id=0,
                        end_page_id=99999,
                        layout_config=layout_config,
                        ocr_config=ocr_config,
                        formula_config=formula_config,
                        table_config=table_config,
                        checkbox_config=checkbox_config,
                        image_config=image_config,
                        use_async_pipeline=DEFAULT_PIPELINE_MODE,
                        hybrid=DEFAULT_HYBRID,
                    )
                    
                    # 결과 수집
                    response_data = {}
                    
                    # TEXT_DETECTION의 경우 간단한 텍스트 반환
                    if is_text_detection and not is_full_document:
                        md_content = get_infer_result(".md", image_name, temp_output_dir)
                        if md_content:
                            response_data["textAnnotations"] = [
                                {
                                    "description": md_content,
                                    "locale": "auto"
                                }
                            ]
                            response_data["fullTextAnnotation"] = {
                                "text": md_content
                            }
                    
                    # DOCUMENT_TEXT_DETECTION의 경우 상세 정보 반환
                    if is_full_document:
                        md_content = get_infer_result(".md", image_name, temp_output_dir)
                        middle_json_content = get_infer_result("_middle.json", image_name, temp_output_dir)
                        content_list_content = get_infer_result("_content_list.json", image_name, temp_output_dir)
                        
                        response_data["fullTextAnnotation"] = {
                            "text": md_content if md_content else ""
                        }
                        
                        if middle_json_content:
                            try:
                                middle_json = json.loads(middle_json_content)
                                response_data["middleJson"] = middle_json
                            except:
                                pass
                        
                        if content_list_content:
                            try:
                                content_list = json.loads(content_list_content)
                                response_data["contentList"] = content_list
                            except:
                                pass
                        
                        # textAnnotations도 포함
                        if md_content:
                            response_data["textAnnotations"] = [
                                {
                                    "description": md_content,
                                    "locale": "auto"
                                }
                            ]
                    
                    responses.append(response_data)
                    
            except Exception as e:
                logger.error(f"Error processing request {idx}: {str(e)}")
                logger.error(traceback.format_exc())
                responses.append({
                    "error": {
                        "code": 500,
                        "message": str(e),
                        "status": "INTERNAL"
                    }
                })
        
        return JSONResponse({
            "responses": responses
        }, status_code=200)
        
    except Exception as e:
        error_message = remove_invalid_surrogates(str(e))
        logger.error(f"API error: {error_message}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "error": {
                "code": 500,
                "message": error_message,
                "status": "INTERNAL"
            }
        }, status_code=500)
    finally:
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RapidDoc Offline API server (closed environment)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port to listen on (default: 8888)",
    )
    parser.add_argument(
        "--no-gzip",
        action="store_true",
        help="Disable GZip middleware (enabled by default)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level for uvicorn (default: info)",
    )
    parser.add_argument(
        "--deepx-default",
        dest="deepx_default",
        action="store_true",
        help="Use DeepX engines by default when requests omit the deepx flag (default)",
    )
    parser.add_argument(
        "--no-deepx-default",
        dest="deepx_default",
        action="store_false",
        help="Use ONNXRuntime engines by default when requests omit the deepx flag",
    )
    parser.set_defaults(deepx_default=DEFAULT_DEEPX)
    parser.add_argument(
        "--pipeline-mode",
        choices=["finegrained", "async", "sync"],
        default=DEFAULT_PIPELINE_MODE,
        help="Pipeline execution mode (default: finegrained = 7-stage streaming, fastest)",
    )
    hybrid_group = parser.add_mutually_exclusive_group()
    hybrid_group.add_argument(
        "--hybrid", dest="hybrid", action="store_true",
        help="Force hybrid device partitioning (requires 2+ NPU devices)",
    )
    hybrid_group.add_argument(
        "--no-hybrid", dest="hybrid", action="store_false",
        help="Disable hybrid device partitioning",
    )
    parser.set_defaults(hybrid=DEFAULT_HYBRID)
    args = parser.parse_args()

    if args.no_gzip:
        # Remove GZip middleware when requested (middleware list is ordered)
        app.user_middleware = [m for m in app.user_middleware if m.cls is not GZipMiddleware]
        app.middleware_stack = app.build_middleware_stack()

    # Apply CLI-driven defaults globally
    DEFAULT_DEEPX = args.deepx_default
    DEFAULT_PIPELINE_MODE = {
        "finegrained": "finegrained", "async": True, "sync": False
    }[args.pipeline_mode]
    DEFAULT_HYBRID = args.hybrid

    logger.info("=" * 80)
    logger.info("RapidDoc Offline API Server Starting...")
    logger.info(f"Version: {__version__}")
    logger.info("Mode: Closed Environment (Network Blocked)")
    logger.info("-" * 80)
    logger.info(f"Pipeline Mode: {args.pipeline_mode}")
    logger.info(f"Hybrid Device Partitioning: {'enabled' if DEFAULT_HYBRID else 'disabled'}")
    logger.info("Default Engine Settings (deepx=False):")
    logger.info(f"  Layout Engine:  {DEFAULT_LAYOUT_ENGINE}")
    logger.info(f"  OCR Engine:     {DEFAULT_OCR_ENGINE}")
    logger.info(f"  Formula Engine: {DEFAULT_FORMULA_ENGINE}")
    logger.info(f"  Table Engine:   {DEFAULT_TABLE_ENGINE}")
    logger.info("-" * 80)
    logger.info("DeepX Engine Settings (deepx=True):")
    logger.info(f"  Layout Engine:  {DEEPX_LAYOUT_ENGINE}")
    logger.info(f"  OCR Engine:     {DEEPX_OCR_ENGINE}")
    logger.info(f"  Formula Engine: {DEEPX_FORMULA_ENGINE}")
    logger.info(f"  Table Engine:   {DEEPX_TABLE_ENGINE}")
    logger.info("=" * 80)
    logger.info(f"Default deepx mode (requests with deepx omitted): {'enabled' if DEFAULT_DEEPX else 'disabled'}")
    logger.info(f"Server will start at: http://{args.host}:{args.port}")
    logger.info(f"API Documentation: http://{args.host}:{args.port}/docs")
    logger.info("=" * 80)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
