"""
Layout 모델의 batch_predict 출력만 확인하는 스크립트

사용법:
    python value_compare/layout_doc.py --pdf ./test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf --model_type pp_doclayout_l --save_output --visualize
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from pypdfium2 import PdfDocument

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rapid_doc.model.layout.rapid_layout import RapidLayoutModel
from rapid_doc.model.layout.rapid_layout_self import ModelType
from rapid_doc.utils.pdf_reader import page_to_image


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
    
    Args:
        img_bgr: 원본 BGR 이미지
        layout_res: batch_predict 결과 (한 페이지)
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


def main():
    parser = argparse.ArgumentParser(
        description='Layout 모델의 batch_predict 출력을 확인합니다.'
    )
    parser.add_argument(
        '--pdf',
        type=str,
        required=True,
        help='입력 PDF 파일 경로'
    )
    parser.add_argument(
        '--page',
        type=int,
        default=0,
        help='PDF 페이지 번호 (0부터 시작, 기본값: 0)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='PDF 렌더링 DPI (기본값: 200)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='pp_doclayout_l',
        choices=['pp_doclayout_plus_l', 'pp_doclayout_l', 'pp_doclayout_m', 'pp_doclayout_s'],
        help='모델 타입 (기본값: pp_doclayout_l)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='모델 파일 경로 (.onnx 또는 .dxnn 파일)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='배치 크기 (기본값: 1)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='출력 디렉토리 (기본값: value_compare/output)'
    )
    parser.add_argument(
        '--save_output',
        action='store_true',
        help='출력 결과를 JSON 파일로 저장'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='결과를 이미지로 시각화'
    )
    
    args = parser.parse_args()
    
    # PDF 경로 검증
    if not os.path.exists(args.pdf):
        print(f"오류: PDF 파일을 찾을 수 없습니다: {args.pdf}")
        return
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'value_compare', 'output')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 타입 변환
    model_type_map = {
        'pp_doclayout_plus_l': ModelType.PP_DOCLAYOUT_PLUS_L,
        'pp_doclayout_l': ModelType.PP_DOCLAYOUT_L,
        'pp_doclayout_m': ModelType.PP_DOCLAYOUT_M,
        'pp_doclayout_s': ModelType.PP_DOCLAYOUT_S,
    }
    model_type = model_type_map[args.model_type]
    
    print("=" * 80)
    print("Layout Model batch_predict 출력 확인")
    print("=" * 80)
    
    # Layout 모델 초기화
    print(f"\n모델 초기화: {args.model_type}")
    layout_config = {
        "model_type": model_type,
    }
    
    # 모델 경로가 지정된 경우 추가
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"오류: 모델 파일을 찾을 수 없습니다: {args.model_path}")
            return
        layout_config["model_dir_or_path"] = args.model_path
        print(f"모델 경로: {args.model_path}")
    
    layout_model = RapidLayoutModel(layout_config=layout_config)
    
    # PDF에서 이미지 로드
    print(f"\nPDF 로드: {args.pdf}")
    print(f"페이지: {args.page + 1}, DPI: {args.dpi}")
    img_bgr = load_pdf_page(args.pdf, page_num=args.page, dpi=args.dpi)
    
    # batch_predict 실행
    print("\n" + "=" * 80)
    print("Layout Model batch_predict 실행")
    print("=" * 80)
    np_images = [img_bgr]
    
    print(f"입력 이미지 개수: {len(np_images)}")
    print(f"배치 크기: {args.batch_size}")
    
    images_layout_res = layout_model.batch_predict(np_images, args.batch_size)
    
    print(f"\n결과 페이지 수: {len(images_layout_res)}")
    if len(images_layout_res) > 0:
        print(f"첫 번째 페이지의 detection 개수: {len(images_layout_res[0])}")
        
        # 첫 번째 페이지의 결과 출력
        print("\n첫 번째 페이지 Detection 결과:")
        for i, detection in enumerate(images_layout_res[0]):
            print(f"\nDetection {i+1}:")
            # 모든 키 출력
            for key, value in detection.items():
                if key == 'score':
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    # 결과 저장
    if args.save_output:
        print("\n" + "=" * 80)
        print("결과 저장")
        print("=" * 80)
        
        # 파일명 생성
        pdf_name = Path(args.pdf).stem
        output_filename = f"layout_output_{args.model_type}_{pdf_name}_page{args.page + 1}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # JSON으로 저장 (numpy 타입을 Python 타입으로 변환)
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            return obj
        
        serializable_res = convert_to_serializable(images_layout_res)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_res, f, indent=2, ensure_ascii=False)
        
        print(f"저장 완료: {output_path}")
    
    # 시각화
    if args.visualize:
        print("\n" + "=" * 80)
        print("결과 시각화")
        print("=" * 80)
        
        # 파일명 생성
        pdf_name = Path(args.pdf).stem
        vis_filename = f"layout_vis_{args.model_type}_{pdf_name}_page{args.page + 1}.jpg"
        vis_path = os.path.join(output_dir, vis_filename)
        
        # 시각화 실행
        if len(images_layout_res) > 0:
            visualize_layout_result(img_bgr, images_layout_res[0], vis_path)
        else:
            print("시각화할 detection 결과가 없습니다.")
    
    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)


if __name__ == '__main__':
    main()
