#!/usr/bin/env python3
# Copyright (c) Opendatalab. All rights reserved.
"""
RapidDoc Gradio Web UI - For Closed Environment
DX Engine based PDF parsing web interface

Usage:
    source venv/bin/activate
    source deepx_scripts/set_env.sh 1 2 1 3 2 4
    python demo/gradio_app.py
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ["GRADIO_DEFAULT_LANGUAGE"] = "en"
os.environ.setdefault("LANG", "en_US.UTF-8")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
import gradio as gr
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

# =============================================================================
# Environment Setup Check: Verify if deepx_scripts/set_env.sh has been executed
# =============================================================================
def check_environment_setup():
    """
    Check if deepx_scripts/set_env.sh has been executed.
    DXRT_TASK_MAX_LOAD is checked for presence only (any value accepted).
    Exit with warning if required environment variables are missing.
    """
    required_env_vars = {
        "CUSTOM_INTER_OP_THREADS_COUNT": "1",
        "CUSTOM_INTRA_OP_THREADS_COUNT": "2",
        "DXRT_DYNAMIC_CPU_THREAD": "1",
        "DXRT_TASK_MAX_LOAD": None,  # presence-only check
        "NFH_INPUT_WORKER_THREADS": "2",
        "NFH_OUTPUT_WORKER_THREADS": "4",
    }
    
    missing_vars = []
    incorrect_vars = []
    for var_name, expected_value in required_env_vars.items():
        actual = os.environ.get(var_name)
        if actual is None:
            missing_vars.append(var_name)
        elif expected_value is not None and actual != expected_value:
            incorrect_vars.append(f"{var_name}={actual} (expected: {expected_value})")
    
    if missing_vars or incorrect_vars:
        logger.error("=" * 80)
        logger.error("❌ Environment setup is not complete!")
        logger.error("")
        logger.error("Please run the following command first:")
        logger.error("  $ source ./deepx_scripts/set_env.sh 1 2 1 3 2 4")
        logger.error("")
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        if incorrect_vars:
            logger.error(f"Variables with unexpected values: {', '.join(incorrect_vars)}")
        logger.error("=" * 80)
        sys.exit(1)
    
    logger.info("✅ Environment setup verified")
    for var_name, expected_value in required_env_vars.items():
        label = "any" if expected_value is None else expected_value
        logger.info(f"  {var_name}={os.environ.get(var_name)} (expected: {label})")

check_environment_setup()
# =============================================================================

# =============================================================================
# Closed Environment Setup: Block model downloads
# =============================================================================
os.environ['MINERU_MODEL_SOURCE'] = 'local'  # Use local models only

# Block requests and urllib to completely prevent external download attempts
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

# Project root directory
__dir__ = os.path.dirname(os.path.abspath(__file__))
project_root = Path(__dir__).parent.absolute()
onnx_models_dir = project_root / "onnx_models"
dxnn_models_dir = project_root / "dxnn_models"

# Default output directory
DEFAULT_OUTPUT_DIR = project_root / "demo" / "output-gradio"
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

# Pipeline / hybrid defaults (aligned with demo_offline.py optimizations)
#   - 'finegrained' : 7-stage per-page streaming pipeline (fastest, default)
#   - True          : AsyncPipelineRapidDoc (legacy async)
#   - False         : Synchronous batch processing
DEFAULT_PIPELINE_MODE = 'finegrained'

def _autodetect_hybrid_default() -> bool:
    try:
        from rapid_doc.utils.device_utils import get_dxnn_devices
        return len(get_dxnn_devices()) >= 2
    except Exception:
        return False

DEFAULT_HYBRID = _autodetect_hybrid_default()


def extract_performance_summary(perf_logs: str) -> Tuple[List[List[str]], str]:
    """
    Extract per-PDF performance statistics and return rows plus total time.

    Returns: (rows, total_time)
    rows: [[model, engine, time, percentage, items, s_per_it, it_per_s], ...]
    total_time: string like "27.31s" or ""
    """
    if not perf_logs:
        return [], ""
    
    # Find Performance Summary section only (not per-PDF statistics)
    lines = perf_logs.split('\n')
    in_summary_section = False
    summary_data = []
    total_time = ""
    
    for line in lines:
        # Detect Performance Summary section (not "Performance by PDF")
        if "Performance Summary" in line:
            in_summary_section = True
            continue
        
        # Exit if we hit "Performance by PDF" or "per-PDF" section
        if in_summary_section and ("Performance by PDF" in line):
            break
        
        if in_summary_section:
            # Extract model performance lines
            # Check if line contains performance data (has | separator and time format)
            if '|' in line and 's (' in line and 'it/s' in line:
                # Parse line format: "📊 Layout   [    dxengine] |    3.98s ( 14.6%) |   13it | 0.306 s/it |   3.26 it/s"
                try:
                    parts = line.split('|')
                    if len(parts) >= 5:
                        # Extract model name and engine
                        model_part = parts[0].strip()
                        # Remove emoji (first character if it's an emoji)
                        if model_part and not model_part[0].isalnum():
                            model_part = model_part[1:].strip()
                        
                        model_name = model_part.split('[')[0].strip()
                        
                        # Skip PDF-Det entries
                        if 'PDF-Det' in model_name or 'pdf-det' in model_name.lower():
                            continue
                        
                        engine = model_part.split('[')[1].split(']')[0].strip() if '[' in model_part else 'N/A'
                        
                        # Extract metrics
                        time_part = parts[1].strip()  # "3.98s ( 14.6%)"
                        time_match = time_part.split('s')[0].strip()
                        
                        items = parts[2].strip().split('it')[0].strip()  # "13it"
                        it_per_s = parts[4].strip().split('it/s')[0].strip()  # "3.26 it/s"
                        
                        summary_data.append({
                            'model': model_name,
                            'engine': engine,
                            'time': time_match,
                            'items': items,
                            'it_per_s': it_per_s
                        })
                except Exception as e:
                    # Log parsing error for debugging
                    logger.debug(f"Failed to parse performance line: {line}, error: {e}")
                    pass
            
            # Extract total processing time
            if "Total processing time" in line:
                try:
                    # Format: "🔥 Total processing time: 27.31s"
                    total_time = line.split(':')[1].strip()
                except:
                    pass
    
    if not summary_data:
        return [], ""

    rows = []
    for data in summary_data:
        rows.append([
            data['model'],
            data['engine'],
            data['time'],
            data['items'],
            data['it_per_s'],
        ])

    if total_time:
        rows.append(["Total", "", total_time, "", ""])
    return rows, total_time


def format_performance_markdown(perf_rows: List[List[str]], display_mode: str = "all") -> str:
    """
    Format performance data as a styled markdown card.
    
    Args:
        perf_rows: List of performance data rows
        display_mode: "all", "time", "items", or "throughput"
        
    Returns:
        Formatted markdown string
    """
    if not perf_rows:
        return "**No performance data available yet.**\n\nRun a parsing task to see performance metrics."
    
    md_lines = []
    
    # Process each row
    for row in perf_rows:
        if len(row) < 5:
            continue
            
        model, engine, time, items, it_per_s = row
        
        # Skip Total row
        if model == "Total":
            continue
        
        # Model performance row with emoji
        model_emoji = {
            'Layout': '📊',
            'OCR-Det': '🔍',
            'OCR-Rec': '✍️',
            'Formula': '📐',
            'Table': '📋'
        }.get(model, '📄')
        
        engine_badge = f"`{engine}`" if engine != 'N/A' else ""
        
        # Build metrics based on display mode
        metrics = []
        if display_mode in ["all", "time"]:
            metrics.append(f"⏱️ **{time}s**")
        if display_mode in ["all", "items"]:
            metrics.append(f"📦 {items} items")
        if display_mode in ["all", "throughput"]:
            metrics.append(f"⚡ **{it_per_s} it/s**")
        
        metrics_str = " · ".join(metrics) if metrics else "No metrics selected"
        
        md_lines.append(
            f"**{model_emoji} {model}** {engine_badge}  \n"
            f"{metrics_str}"
        )
    
    return "\n\n".join(md_lines)


def convert_md_images_for_gradio(md_content: str, image_dir: str) -> str:
    """
    Convert markdown image paths to base64-encoded data URLs.
    This is the most reliable way to display images in Gradio Markdown.
    """
    import re
    import base64
    
    def replace_image_path(match):
        img_path = match.group(1)
        
        # Determine the actual file path
        if os.path.isabs(img_path):
            full_path = img_path
        else:
            full_path = os.path.join(image_dir, img_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            logger.warning(f"Image not found: {full_path}")
            return match.group(0)  # Return original if file not found
        
        try:
            # Read image and encode to base64
            with open(full_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Determine image type from extension
            ext = os.path.splitext(full_path)[1].lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # Create data URL
            data_url = f'data:{mime_type};base64,{img_base64}'
            
            # Return HTML img tag with base64 data
            return f'<img src="{data_url}" style="max-width: 100%; height: auto;" />'
            
        except Exception as e:
            logger.error(f"Failed to encode image {full_path}: {e}")
            return match.group(0)  # Return original on error
    
    # Replace markdown images with base64-encoded HTML img tags
    md_content = re.sub(r'!\[\]\(([^)]+)\)', replace_image_path, md_content)
    
    return md_content


def get_model_config(
    layout_engine: str,
    ocr_engine: str,
    formula_engine: str,
    table_engine: str,
):
    """Generate model configuration"""
    from rapidocr import EngineType as OCREngineType
    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
    from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType
    from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType
    from rapid_doc.model.formula.rapid_formula_self.utils.typings import EngineType as FormulaEngineType
    from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

    # Layout configuration
    layout_config = {"model_type": LayoutModelType.PP_DOCLAYOUT_L}
    if layout_engine == "dxengine":
        layout_config["engine_type"] = LayoutEngineType.DXENGINE
        layout_config["model_dir_or_path"] = str(dxnn_models_dir / "pp_doclayout_l_part1.dxnn")
        layout_config["sub_model_path"] = str(onnx_models_dir / "pp_doclayout_l_part2.onnx")
    else:
        layout_config["engine_type"] = LayoutEngineType.ONNXRUNTIME
        layout_config["model_dir_or_path"] = str(onnx_models_dir / "pp_doclayout_l.onnx")

    # OCR configuration
    ocr_config = {}
    if ocr_engine == "dxengine":
        ocr_config["engine_type"] = "dxengine"
        ocr_config["Det.model_path"] = str(dxnn_models_dir / "ch_PP-OCRv5_server_det.dxnn")
        ocr_config["Rec.model_path"] = str(dxnn_models_dir / "ch_PP-OCRv5_rec_server_infer.dxnn")
        ocr_config["char_dict_path"] = str(project_root / "value_compare" / "recognition" / "character_dict_from_onnx.txt")
    elif ocr_engine == "paddle":
        ocr_config["Det.engine_type"] = OCREngineType.PADDLE
        ocr_config["Rec.engine_type"] = OCREngineType.PADDLE
    else:
        ocr_config["Det.engine_type"] = OCREngineType.ONNXRUNTIME
        ocr_config["Rec.engine_type"] = OCREngineType.ONNXRUNTIME
        ocr_config["Det.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_server_det.onnx")
        ocr_config["Rec.model_path"] = str(onnx_models_dir / "ch_PP-OCRv5_rec_server_infer.onnx")

    ocr_config.update({
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
        "save_debug_images": False,
    })

    # Formula configuration
    formula_config = {"model_type": FormulaModelType.PP_FORMULANET_PLUS_M}
    if formula_engine == "dxengine":
        formula_config["engine_type"] = "dxengine"
        formula_config["model_dir_or_path"] = str(dxnn_models_dir / "pp_formulanet_plus_m.dxnn")
    else:
        formula_config["engine_type"] = FormulaEngineType.ONNXRUNTIME
        formula_config["model_dir_or_path"] = str(onnx_models_dir / "pp_formulanet_plus_m.onnx")

    # Table configuration
    table_config = {"model_type": TableModelType.UNET}
    if table_engine == "dxengine":
        table_config["engine_type"] = "dxengine"
        table_config["unet.model_dir_or_path"] = str(dxnn_models_dir / "unet.dxnn")
    elif table_engine == "torch":
        table_config["engine_type"] = "torch"
    else:
        table_config["engine_type"] = "onnxruntime"
        table_config["unet.model_dir_or_path"] = str(onnx_models_dir / "unet.onnx")

    checkbox_config = {"checkbox_enable": False}
    image_config = {
        "extract_original_image": False,
        "extract_original_image_iou_thresh": 0.5,
    }

    return layout_config, ocr_config, formula_config, table_config, checkbox_config, image_config


def update_engine_settings(preset: str) -> Tuple[str, str, str, str]:
    """
    Update individual engine settings based on preset selection.
    
    Args:
        preset: "onnxruntime" or "deepx-npu"
    
    Returns:
        (layout_engine, ocr_engine, formula_engine, table_engine)
    """
    if preset == "onnxruntime":
        return "onnxruntime", "onnxruntime", "onnxruntime", "onnxruntime"
    else:  # deepx-npu
        return "dxengine", "dxengine", "onnxruntime", "dxengine"


def parse_document(
    file_paths: List[str],
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    layout_engine: str,
    ocr_engine: str,
    formula_engine: str,
    table_engine: str,
    use_async_pipeline: bool,
    progress=gr.Progress(),
) -> Tuple[str, str, str, List[str], List[List[str]]]:
    """
    Document parsing function - supports multiple files
    
    Returns:
        (markdown_content, info_text, layout_pdf_path, image_list, performance_rows)
    """
    try:
        from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
        from rapid_doc.data.data_reader_writer import FileBasedDataWriter
        from rapid_doc.utils.draw_bbox import draw_layout_bbox
        from rapid_doc.utils.enum_class import MakeMode
        from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

        if not file_paths:
            return "", "❌ Please upload a file or folder.", None, [], []

        progress(0.05, desc="Scanning files...")
        
        # Collect all valid files from the uploaded paths
        valid_files = []
        pdf_suffixes = [".pdf"]
        image_suffixes = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
        supported_suffixes = pdf_suffixes + image_suffixes
        
        for file_path in file_paths:
            path = Path(file_path)
            if path.is_file():
                if path.suffix.lower() in supported_suffixes:
                    valid_files.append(str(path))
        
        if not valid_files:
            return "", "❌ No valid PDF or image files found.", None, [], []
        
        logger.info(f"📂 Found {len(valid_files)} file(s) to process")
        
        # Process all files
        all_md_contents = []
        all_extracted_images = []
        all_perf_logs = []  # 모든 파일의 성능 로그 수집
        total_pages_all = 0
        total_time_all = 0
        
        for file_idx, file_path in enumerate(valid_files):
            progress((0.1 + (file_idx / len(valid_files)) * 0.8), 
                    desc=f"Processing file {file_idx + 1}/{len(valid_files)}...")
            
            # File information
            file_name = Path(file_path).stem
            file_suffix = Path(file_path).suffix.lower()
            
            logger.info("=" * 80)
            logger.info(f"📄 Processing [{file_idx + 1}/{len(valid_files)}]: {file_name}{file_suffix}")
            logger.info("=" * 80)
            logger.info("=" * 80)
            logger.info(f"📄 Processing [{file_idx + 1}/{len(valid_files)}]: {file_name}{file_suffix}")
            logger.info("=" * 80)
        
            # Model configuration (only once for first file)
            if file_idx == 0:
                layout_config, ocr_config, formula_config, table_config, checkbox_config, image_config = get_model_config(
                    layout_engine, ocr_engine, formula_engine, table_engine
                )
            
            # Read file
            pdf_bytes = read_fn(file_path)
            
            # Use full document
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)

            logger.info(f"Formula recognition: {'Enabled' if formula_enable else 'Disabled'}")
            logger.info(f"Table recognition: {'Enabled' if table_enable else 'Disabled'}")
            
            # Model inference with log capture
            start_time = time.time()
            
            # Add handler to capture logs
            import io
            log_stream = io.StringIO()
            log_handler = logger.add(log_stream, format="{message}", level="INFO")
            
            infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list, *_ = pipeline_doc_analyze(
                [new_pdf_bytes],
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                layout_config=layout_config,
                ocr_config=ocr_config,
                formula_config=formula_config,
                table_config=table_config,
                checkbox_config=checkbox_config,
                use_async_pipeline=use_async_pipeline,
                hybrid=DEFAULT_HYBRID,
            )
            
            # Remove log handler and extract log content
            logger.remove(log_handler)
            perf_logs = log_stream.getvalue()
            log_stream.close()
            all_perf_logs.append(perf_logs)
            
            # Process results
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_dict = all_page_dicts[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            
            # Set output directory
            local_image_dir, local_md_dir = prepare_env(str(DEFAULT_OUTPUT_DIR), file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            # Generate Middle JSON
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_dict, image_writer, _lang, _ocr_enable,
                formula_enable, ocr_config=ocr_config, image_config=image_config
            )

            pdf_info = middle_json["pdf_info"]
            
            # Draw layout bbox
            layout_pdf_path = os.path.join(local_md_dir, f"{file_name}_layout.pdf")
            draw_layout_bbox(pdf_info, new_pdf_bytes, local_md_dir, f"{file_name}_layout.pdf")
            
            # Generate Markdown
            md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, local_image_dir)
            
            # Convert image paths to Gradio-compatible format
            md_content = convert_md_images_for_gradio(md_content, local_image_dir)
            
            # Save Markdown
            md_writer.write_string(f"{file_name}.md", md_content)
            
            file_time = time.time() - start_time
            file_pages = len(images_list)
            
            total_time_all += file_time
            total_pages_all += file_pages
            
            # Add separator between files
            all_md_contents.append(f"## 📄 {file_name}{file_suffix}\n\n")
            all_md_contents.append(md_content)
            all_md_contents.append(f"\n\n---\n\n")
            
            # Collect extracted images
            import glob
            if os.path.exists(local_image_dir):
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    image_files = glob.glob(os.path.join(local_image_dir, ext))
                    all_extracted_images.extend(sorted(image_files))
            
            logger.info(f"✅ [{file_idx + 1}/{len(valid_files)}] Complete: {file_time:.2f}s, {file_pages} pages")
        
        progress(0.95, desc="Generating summary...")
        
        # Parse and format performance statistics from all logs
        combined_perf_logs = "\n".join(all_perf_logs)
        perf_rows, perf_total_time = extract_performance_summary(combined_perf_logs)
        
        # Debugging: Check if log is empty
        if not perf_rows:
            logger.warning("Failed to parse performance logs.")
            logger.debug(f"Captured log length: {len(combined_perf_logs)} characters")
            # Output log sample (first 500 characters)
            if combined_perf_logs:
                logger.debug(f"Log sample:\n{combined_perf_logs[:500]}")
        
        # Generate combined info text
        info_text = f"""✅ Batch Parsing Complete!

� Total Files: {len(valid_files)} files
📊 Total Pages: {total_pages_all} pages
⏱️ Total Time: {total_time_all:.2f}s
⚡ Average Speed: {total_time_all/total_pages_all:.3f} s/it | {total_pages_all/total_time_all:.2f} it/s

💾 Output Directory: {DEFAULT_OUTPUT_DIR}

"""
        
        logger.info("=" * 80)
        logger.info(f"🎉 Batch parsing complete: {len(valid_files)} files, {total_pages_all} pages in {total_time_all:.2f}s")
        logger.info("=" * 80)
        
        # Combine all markdown
        combined_md = "".join(all_md_contents)
        
        progress(1.0, desc="Complete!")
        
        # Return last layout PDF for preview
        return combined_md, info_text, layout_pdf_path if valid_files else None, all_extracted_images, perf_rows

    except Exception as e:
        logger.exception(e)
        error_text = f"❌ Error occurred:\n{str(e)}"
        return "", error_text, None, [], []


# Create Gradio interface
async def _set_language_cookie(request, call_next):
    response = await call_next(request)
    # Force UI language to English on every response
    if response is not None:
        response.set_cookie("language", "en", path="/", max_age=30 * 24 * 3600)
    return response


def create_ui():
    custom_css = """
    #md-preview,
    #md-preview > div {
        overflow: auto;
    }

    #right-pane {
        max-width: 900px;
        min-width: 720px;
    }

    #right-pane .tabitem {
        max-height: 720px;
        overflow: auto;
    }
    
    #md-preview img {
        max-width: 100%;
        height: auto;
    }
    
    #perf-mode-radio label {
        font-size: 0.85em !important;
    }
    
    #perf-mode-radio .wrap {
        gap: 0.3em !important;
    }
    """

    with gr.Blocks(
        title="RapidDoc - DX Engine", 
        theme=gr.themes.Soft(), 
        css=custom_css,
    ) as demo:
        # Header row with title and performance table side by side
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                # 🚀 RapidDoc - Document Parsing (DX Engine)
                
                **High-Performance Document Parsing System for Closed Environments**
                
                Upload PDF or image files to extract text, tables, formulas, and more.
                
                """)
            
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 📊 Performance Summary")
                        perf_display = gr.Markdown(
                            value="**No performance data available yet.**\n\nRun a parsing task to see performance metrics.",
                            elem_id="perf-summary"
                        )
                    
                    with gr.Column(scale=1):
                        perf_mode = gr.Radio(
                            choices=[
                                ("All", "all"),
                                ("⏱️ Time", "time"),
                                ("📦 Items", "items"),
                                ("⚡ Throughput", "throughput")
                            ],
                            value="all",
                            label="",
                            container=False,
                            elem_id="perf-mode-radio"
                        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload - support multiple files
                file_input = gr.File(
                    label="📁 File Upload (PDF or Image) - Multiple files supported",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
                    file_count="multiple",
                )
                
                
                # Engine preset selection
                with gr.Group():
                    gr.Markdown("### 🔧 Engine Selection")
                    engine_preset = gr.Radio(
                        choices=[
                            ("⚡ DeepX NPU (Recommended)", "deepx-npu"),
                            ("ONNX Runtime (CPU)", "onnxruntime")
                        ],
                        value="deepx-npu",
                        label="Engine Preset",
                        info="onnxruntime: All models use ONNX Runtime | deepx-npu: All models use DX Engine except Formula"
                    )
                
                # Hidden components to store individual engine settings
                layout_engine = gr.State(value="dxengine")
                ocr_engine = gr.State(value="dxengine")
                formula_engine = gr.State(value="onnxruntime")
                table_engine = gr.State(value="dxengine")
                
                
                # Basic settings
                with gr.Group():
                    gr.Markdown("### ⚙️ Basic Settings")
                    parse_method = gr.Radio(
                        choices=["auto", "ocr", "txt"],
                        value="auto",
                        label="Parsing Method",
                        info="auto: Auto select | ocr: Force OCR | txt: Text extraction only"
                    )
                formula_enable = gr.State(value=True)
                table_enable = gr.State(value=True)
                use_async_pipeline = gr.State(value=DEFAULT_PIPELINE_MODE)
                
                parse_btn = gr.Button("🚀 Start Parsing", variant="primary", size="lg")

                # Parsing info under action button
                with gr.Accordion("ℹ️ Parsing Information", open=True):
                    info_output = gr.Textbox(
                        label="info box",
                        lines=13,
                        max_lines=40,
                        show_copy_button=True
                    )
            
            with gr.Column(scale=2, elem_id="right-pane"):
                # Separate results by tabs
                with gr.Tabs():
                    with gr.Tab("📖️ Markdown Preview"):
                        md_preview = gr.Markdown(
                            label="Rendered Markdown",
                            value="",
                            elem_id="md-preview"
                        )
                    
                    with gr.Tab("📝 Markdown Source"):
                        md_output = gr.Textbox(
                            label="Markdown Text (for copy)",
                            lines=100,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("🖼️ Extracted Images"):
                        image_gallery = gr.Gallery(
                            label="Images Extracted from Document",
                            columns=3,
                            height="auto",
                            object_fit="contain"
                        )
                    
                    with gr.Tab("🎨 Layout Visualization"):
                        layout_output = gr.File(
                            label="Layout PDF (Downloadable)",
                        )
        
        # Connect events
        # Update engine settings when preset changes
        engine_preset.change(
            fn=update_engine_settings,
            inputs=[engine_preset],
            outputs=[layout_engine, ocr_engine, formula_engine, table_engine]
        )
        
        # Store performance rows in state
        perf_rows_state = gr.State(value=[])
        
        def parse_and_display(file_paths, parse_method, formula_enable, table_enable,
                            layout_engine, ocr_engine, formula_engine, table_engine,
                            use_async_pipeline, perf_mode, progress=gr.Progress()):
            """Wrapper to parse document and format performance display"""
            md_content, info_text, layout_pdf, images, perf_rows = parse_document(
                file_paths, parse_method, formula_enable, table_enable,
                layout_engine, ocr_engine, formula_engine, table_engine,
                use_async_pipeline, progress
            )
            perf_md = format_performance_markdown(perf_rows, perf_mode)
            return md_content, info_text, layout_pdf, images, perf_md, perf_rows
        
        def update_perf_display(perf_rows, perf_mode):
            """Update performance display when mode changes"""
            return format_performance_markdown(perf_rows, perf_mode)
        
        parse_btn.click(
            fn=parse_and_display,
            inputs=[
                file_input,
                parse_method,
                formula_enable,
                table_enable,
                layout_engine,
                ocr_engine,
                formula_engine,
                table_engine,
                use_async_pipeline,
                perf_mode,
            ],
            outputs=[md_output, info_output, layout_output, image_gallery, perf_display, perf_rows_state]
        ).then(
            fn=lambda md: md,
            inputs=[md_output],
            outputs=[md_preview]
        )
        
        # Update performance display when mode changes
        perf_mode.change(
            fn=update_perf_display,
            inputs=[perf_rows_state, perf_mode],
            outputs=[perf_display]
        )
        
    
    return demo


if __name__ == "__main__":
    
    logger.info("=" * 80)
    logger.info("RapidDoc Gradio UI Starting...")
    logger.info("Mode: Closed Environment (Network Blocked)")
    logger.info("=" * 80)
    
    demo = create_ui()
    demo.app.add_middleware(BaseHTTPMiddleware, dispatch=_set_language_cookie)
    
    # Allow access to output directory and all subdirectories
    allowed_paths_list = [
        str(DEFAULT_OUTPUT_DIR),
        str(DEFAULT_OUTPUT_DIR.absolute()),
    ]
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=allowed_paths_list
    )
