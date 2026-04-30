import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pypdfium2 import PdfDocument
import onnxruntime as ort
from dx_engine import InferenceEngine, InferenceOption
from rapid_doc.utils.device_utils import get_dxnn_devices

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.post_process import PPPostProcess
from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType
from rapid_doc.utils.pdf_reader import page_to_image
from rapid_doc.utils.enum_class import CategoryId
    

class DxnnPreprocessor:
    def __init__(self, model_type: ModelType = ModelType.PP_DOCLAYOUT_L):
        """
        Args:
            model_type: 모델 타입 (기본값: PP_DOCLAYOUT_L)
        """
        self.model_type = model_type
        
        # 모델 타입에 따른 이미지 크기 설정
        if model_type == ModelType.PP_DOCLAYOUT_PLUS_L:
            self.img_size = (800, 800)
        elif model_type == ModelType.PP_DOCLAYOUT_S:
            self.img_size = (480, 480)
        else:  # PP_DOCLAYOUT_L, PP_DOCLAYOUT_M
            self.img_size = (640, 640)
        
        self.preprocessor = PPPreProcess(img_size=self.img_size, model_type=model_type)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """
        이미지 리스트를 전처리합니다.
        
        Args:
            images: BGR 이미지 리스트 (cv2 형식)

        Returns:
            np.ndarray: 전처리된 이미지 배열
        """
        processed = self.preprocessor.resize(images)
        processed = np.expand_dims(processed, axis=0).astype(np.uint8)
        return processed


def load_pdf_page(pdf_path: str, page_num: int = 0, dpi: int = 200) -> np.ndarray:
    """
    PDF 파일의 특정 페이지를 이미지로 로드합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        page_num: 페이지 번호 (0부터 시작)
        dpi: 렌더링 DPI (기본값: 200)
        
    Returns:
        np.ndarray: BGR 이미지 (cv2 형식)
    """
    pdf_doc = PdfDocument(pdf_path)
    
    if page_num >= len(pdf_doc):
        raise ValueError(f"페이지 번호가 범위를 벗어났습니다. 총 페이지 수: {len(pdf_doc)}, 요청: {page_num}")
    
    total_pages = len(pdf_doc)
    page = pdf_doc[page_num]
    pil_image, scale = page_to_image(page, dpi=dpi)
    
    # PIL Image -> numpy array (RGB) -> BGR
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    print(f"PDF 페이지 로드: {pdf_path}, 페이지 {page_num + 1}/{total_pages}, DPI: {dpi}, Scale: {scale:.2f}")
    print(f"이미지 shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
    
    # 리소스 정리
    page.close()
    pdf_doc.close()
    
    return img_bgr


def visualize_layout_result(img_bgr: np.ndarray, layout_res: list, output_path: str):
    """
    Layout detection 결과를 이미지에 시각화합니다.
    (onnx_layout_doc.py와 동일한 시각화 로직)
    
    Args:
        img_bgr: 원본 BGR 이미지
        layout_res: batch_predict와 동일한 형식의 결과 (category_id, poly, score)
        output_path: 저장할 이미지 경로
    """
    # 이미지 복사
    vis_img = img_bgr.copy()
    
    # Category ID별 색상 매핑 (BGR 형식) - CategoryId enum 값에 맞춤
    category_colors = {
        0: (0, 255, 0),      # Title - 초록색
        1: (0, 0, 255),      # Text - 빨간색
        2: (128, 128, 128),  # Abandon - 회색
        3: (255, 0, 0),      # ImageBody - 파란색
        4: (255, 255, 0),    # ImageCaption - 청록색
        5: (255, 0, 255),    # TableBody - 자홍색
        6: (0, 255, 255),    # TableCaption - 노란색
        7: (200, 200, 0),    # TableFootnote - 진한 청록색
        8: (128, 0, 0),      # InterlineEquation_Layout - 진한 파란색
        9: (0, 0, 128),      # InterlineEquationNumber_Layout - 진한 빨간색
        13: (255, 128, 0),   # InlineEquation - 주황색
        14: (128, 255, 0),   # InterlineEquation_YOLO - 연두색
        15: (0, 128, 255),   # OcrText - 주황-파랑
        16: (128, 0, 128),   # LowScoreText - 보라색
    }
    
    # Category ID별 이름 매핑 - CategoryId enum 값에 맞춤
    category_names = {
        0: "Title",
        1: "Text",
        2: "Abandon",
        3: "ImageBody",
        4: "ImageCaption",
        5: "TableBody",
        6: "TableCaption",
        7: "TableFootnote",
        8: "InterlineEquation_Layout",
        9: "InterlineEquationNumber_Layout",
        13: "InlineEquation",
        14: "InterlineEquation_YOLO",
        15: "OcrText",
        16: "LowScoreText",
    }
    
    # 각 detection 그리기
    for detection in layout_res:
        category_id = detection['category_id']
        score = detection['score']
        
        # poly 또는 bbox 사용
        if 'poly' in detection:
            poly = detection['poly']
            # poly: [x1, y1, x2, y2, x3, y3, x4, y4]
            points = np.array(poly).reshape((-1, 2)).astype(np.int32)
        elif 'bbox' in detection:
            bbox = detection['bbox']
            # bbox: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox
            points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.int32)
        else:
            continue
        
        # 색상 선택
        color = category_colors.get(category_id, (255, 255, 255))
        
        # 다각형 그리기
        cv2.polylines(vis_img, [points], isClosed=True, color=color, thickness=2)
        
        # 라벨 그리기
        category_name = category_names.get(category_id, f"Cat_{category_id}")
        label = f"{category_name} {score:.2f}"
        
        # 텍스트 배경
        x_min, y_min = points[0]
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_img, (x_min, y_min - text_height - baseline - 5), 
                     (x_min + text_width, y_min), color, -1)
        
        # 텍스트
        cv2.putText(vis_img, label, (x_min, y_min - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 이미지 저장
    cv2.imwrite(output_path, vis_img)
    print(f"시각화 결과 저장: {output_path}")


def convert_to_rapid_layout_format(layout_res: list) -> list:
    """
    PPPostProcess 출력을 RapidLayoutModel.batch_predict 형식으로 변환합니다.
    
    Args:
        layout_res: PPPostProcess의 출력 (cls_id, label, score, coordinate)
        
    Returns:
        RapidLayoutModel.batch_predict와 동일한 형식의 결과
    """
    # PP-DocLayout-L 레이블을 CategoryId로 매핑 (RapidLayoutModel의 category_dict와 동일)
    category_dict = {
        "paragraph_title": CategoryId.Title,
        "image": CategoryId.ImageBody,
        "text": CategoryId.Text,
        "number": CategoryId.Abandon,
        "abstract": CategoryId.Text,
        "content": CategoryId.Text,
        "figure_title": CategoryId.ImageCaption,
        "formula": CategoryId.InterlineEquation_YOLO,
        "table": CategoryId.TableBody,
        "table_title": CategoryId.TableCaption,
        "reference": CategoryId.Text,
        "doc_title": CategoryId.Title,
        "footnote": CategoryId.Abandon,
        "header": CategoryId.Abandon,
        "algorithm": CategoryId.Text,
        "footer": CategoryId.Abandon,
        "seal": CategoryId.Abandon,
        "chart_title": CategoryId.ImageCaption,
        "chart": CategoryId.ImageBody,
        "formula_number": CategoryId.InterlineEquationNumber_Layout,
        "header_image": CategoryId.Abandon,
        "footer_image": CategoryId.Abandon,
        "aside_text": CategoryId.Text,
    }
    
    converted_res = []
    for detection in layout_res:
        label = detection['label']
        score = detection['score']
        coordinate = detection['coordinate']
        
        # CategoryId로 변환
        category_id = category_dict.get(label, CategoryId.Abandon)
        
        # coordinate를 poly 형식으로 변환
        if len(coordinate) == 4:
            # bbox: [xmin, ymin, xmax, ymax] -> poly: [x1, y1, x2, y2, x3, y3, x4, y4]
            xmin, ymin, xmax, ymax = coordinate
            poly = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        elif len(coordinate) == 8:
            # 이미 poly 형식
            poly = coordinate
        else:
            continue
        
        converted_res.append({
            "category_id": category_id,
            "poly": poly,
            "score": score,
        })
    
    return converted_res


def run_dxnn(part1_path: str, part2_path: str, pdf_path: str, page_num: int = 0, 
             output_dir: str = "./value_compare/output"):
    """
    DXNN 모델로 Layout Detection을 수행합니다.
    
    Args:
        part1_path: DXNN 모델 파일 경로 (part1)
        part2_path: ONNX 모델 파일 경로 (part2)
        pdf_path: 입력 PDF 파일 경로
        page_num: 처리할 PDF 페이지 번호 (0부터 시작, 기본값: 0)
        output_dir: 출력 디렉토리 경로 (기본값: "./value_compare/output")
    """

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 프로바이더 설정 (CPU)
    providers = ['CPUExecutionProvider']

    io = InferenceOption()
    io.devices = get_dxnn_devices()
    io.bound_option = InferenceOption.BOUND_OPTION.NPU_ALL
    
    dxnn_session = InferenceEngine(part1_path, io)   
    
    ort_session = ort.InferenceSession(
        part2_path,
        sess_options=sess_options,
        providers=providers
    )

    # PDF 로드
    input_image = load_pdf_page(pdf_path, page_num=page_num, dpi=200)
    original_height, original_width = input_image.shape[:2]
    
    # 전처리
    preprocessor = DxnnPreprocessor(model_type=ModelType.PP_DOCLAYOUT_L)
    input_tensor = preprocessor(input_image)
    
    # 리사이즈된 크기
    resized_height, resized_width = 640, 640
    
    # scale_factor 계산 (원본 -> 리사이즈)
    scale_h = resized_height / original_height
    scale_w = resized_width / original_width
    
    # im_shape: 리사이즈된 이미지 크기
    im_shape = np.array([[resized_height, resized_width]], dtype=np.float32)
    
    # scale_factor: 원본 대비 스케일 비율
    scale_factor = np.array([[scale_h, scale_w]], dtype=np.float32)
    
    print(f"원본 이미지 크기: ({original_width}, {original_height})")
    print(f"리사이즈 크기: ({resized_width}, {resized_height})")
    print(f"Scale factor: ({scale_w:.4f}, {scale_h:.4f})")

    # DXNN Runtime 추론 (Part 1)
    out = dxnn_session.run([input_tensor])
    ort_feed = {
            "p2o.pd_op.concat.12.0": out[0],
            "p2o.pd_op.layer_norm.20.0": out[1],
            "im_shape": im_shape,
            "scale_factor": scale_factor
        }

    ort_outputs = ort_session.run(None, ort_feed)
    
    print(f"\n추론 완료!")
    print(f"DXNN 출력 개수: {len(out)}")
    for i, o in enumerate(out):
        print(f"  Output {i}: shape={o.shape}, dtype={o.dtype}")
    
    print(f"\nONNX 출력 개수: {len(ort_outputs)}")
    for i, o in enumerate(ort_outputs):
        print(f"  Output {i}: shape={o.shape}, dtype={o.dtype}")
    
    # 후처리
    print("\n" + "=" * 80)
    print("후처리 시작")
    print("=" * 80)
    
    # PP-DocLayout-L의 레이블 (23개 카테고리)
    labels = [
        "paragraph_title", "image", "text", "number", "abstract", "content",
        "figure_title", "formula", "table", "table_title", "reference",
        "doc_title", "footnote", "header", "algorithm", "footer", "seal",
        "chart_title", "chart", "formula_number", "header_image",
        "footer_image", "aside_text"
    ]
    
    # conf_thresh 설정 (PP_DOCLAYOUT_L_Threshold 값 사용)
    conf_thresh = {
        0: 0.3,    # paragraph_title
        1: 0.5,    # image
        2: 0.4,    # text
        3: 0.5,    # number
        4: 0.5,    # abstract
        5: 0.5,    # content
        6: 0.5,    # figure_title
        7: 0.3,    # formula         
        8: 0.5,    # table
        9: 0.5,    # table_title
        10: 0.5,   # reference
        11: 0.5,   # doc_title
        12: 0.5,   # footnote
        13: 0.5,   # header
        14: 0.5,   # algorithm
        15: 0.5,   # footer
        16: 0.45,  # seal             
        17: 0.5,   # chart_title
        18: 0.5,   # chart
        19: 0.5,   # formula_number
        20: 0.5,   # header_image
        21: 0.5,   # footer_image
        22: 0.5    # aside_text
    }
    
    postprocessor = PPPostProcess(labels=labels, conf_thres=conf_thresh, iou_thres=0.5)
    
    # ort_outputs[0]이 detection 결과
    boxes = ort_outputs[0]  # shape: (N, 6) - [cls_id, score, xmin, ymin, xmax, ymax]
    
    # 원본 이미지 크기를 사용 (후처리에서 좌표가 원본 크기로 복원됨)
    img_size = (original_width, original_height)  # (width, height)
    print(f"후처리용 이미지 크기: {img_size}")
    print(f"Detection boxes shape: {boxes.shape}")
    
    # 후처리 적용
    layout_res = postprocessor(boxes, img_size)
    
    print(f"\n후처리 완료!")
    print(f"Detection 개수: {len(layout_res)}")
    
    # 결과를 RapidLayoutModel 형식으로 변환
    print("\n" + "=" * 80)
    print("결과 변환 (RapidLayoutModel 형식)")
    print("=" * 80)
    
    converted_layout_res = convert_to_rapid_layout_format(layout_res)
    
    print(f"변환된 Detection 개수: {len(converted_layout_res)}")
    
    # 결과 출력
    print("\nDetection 결과 (RapidLayoutModel 형식):")
    for i, detection in enumerate(converted_layout_res):
        print(f"\nDetection {i+1}:")
        print(f"  category_id: {detection['category_id']}")
        print(f"  poly: {detection['poly']}")
        print(f"  score: {detection['score']:.4f}")
    
    # 시각화
    print("\n" + "=" * 80)
    print("결과 시각화")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # PDF 파일명과 페이지 번호를 사용하여 출력 파일명 생성
    pdf_basename = Path(pdf_path).stem
    vis_filename = f"dxnn_layout_vis_{pdf_basename}_page{page_num + 1}.jpg"
    vis_path = os.path.join(output_dir, vis_filename)
    
    visualize_layout_result(input_image, converted_layout_res, vis_path)
    
    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DXNN Layout Detection')
    parser.add_argument('--part1', type=str, default='./dxnn_models/pp_doclayout_l_part1.dxnn',
                        help='DXNN 모델 파일 경로 (part1) (기본값: ./dxnn_models/pp_doclayout_l_part1.dxnn)')
    parser.add_argument('--part2', type=str, default='./onnx_models/pp_doclayout_l_part2.onnx',
                        help='ONNX 모델 파일 경로 (part2) (기본값: ./onnx_models/pp_doclayout_l_part2.onnx)')
    parser.add_argument('--pdf', type=str, default='./test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf',
                        help='입력 PDF 파일 경로 (기본값: ./test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf)')
    parser.add_argument('--page', type=int, default=0,
                        help='처리할 PDF 페이지 번호 (0부터 시작, 기본값: 0)')
    parser.add_argument('--output', type=str, default='./value_compare/output',
                        help='출력 디렉토리 경로 (기본값: ./value_compare/output)')
    
    args = parser.parse_args()
    
    run_dxnn(
        part1_path=args.part1,
        part2_path=args.part2,
        pdf_path=args.pdf,
        page_num=args.page,
        output_dir=args.output
    )