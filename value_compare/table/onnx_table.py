"""
UNET 테이블 모델의 inference 결과를 확인하는 스크립트

사용법:
    python value_compare/table/onnx_table.py --image ./test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf --model_path ./onnx_models/unet.onnx --save_output --visualize
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rapid_doc.model.table.rapid_table_self import (
    RapidTable,
    RapidTableInput,
    ModelType
)


def load_image(image_path: str) -> np.ndarray:
    """
    이미지 파일을 로드합니다.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        np.ndarray: BGR 이미지 (cv2 형식)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    print(f"이미지 로드: {image_path}")
    print(f"이미지 shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
    
    return img_bgr


def visualize_table_result(img_bgr: np.ndarray, cell_bboxes: np.ndarray, logic_points, output_path: str):
    """
    UNET 테이블 인식 결과를 이미지에 시각화합니다.
    
    Args:
        img_bgr: 원본 BGR 이미지
        cell_bboxes: 셀 바운딩 박스 (N, 8) - [x1, y1, x2, y2, x3, y3, x4, y4]
        logic_points: 논리적 위치 정보 (N, 4) - [row_start, row_end, col_start, col_end] (list or ndarray)
        output_path: 저장할 이미지 경로
    """
    # 이미지 복사
    vis_img = img_bgr.copy()
    
    if cell_bboxes is None or len(cell_bboxes) == 0:
        print("시각화할 셀이 없습니다.")
        cv2.imwrite(output_path, vis_img)
        return
    
    # 각 셀 그리기
    for i, bbox in enumerate(cell_bboxes):
        # bbox: [x1, y1, x2, y2, x3, y3, x4, y4]
        points = bbox.reshape((-1, 2)).astype(np.int32)
        
        # 다각형 그리기 (파란색)
        cv2.polylines(vis_img, [points], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # 논리적 위치 정보가 있으면 표시
        if logic_points is not None and i < len(logic_points):
            row_start, row_end, col_start, col_end = logic_points[i]
            label = f"R{int(row_start)}-{int(row_end)},C{int(col_start)}-{int(col_end)}"
            
            # 텍스트 배경
            x_min, y_min = points[0]
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis_img, (x_min, y_min - text_height - baseline - 5), 
                         (x_min + text_width, y_min), (255, 0, 0), -1)
            
            # 텍스트
            cv2.putText(vis_img, label, (x_min, y_min - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 이미지 저장
    cv2.imwrite(output_path, vis_img)
    print(f"시각화 결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='UNET 테이블 모델의 inference 결과를 확인합니다.'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='입력 이미지 파일 경로'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='UNET 모델 파일 경로 (.onnx 파일, 기본값: 자동 다운로드)'
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
    
    # 이미지 경로 검증
    if not os.path.exists(args.image):
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {args.image}")
        return
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        output_dir = os.path.join(project_root, 'value_compare', 'output')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("UNET 테이블 모델 Inference 결과 확인")
    print("=" * 80)
    
    # UNET 모델 초기화
    print(f"\nUNET 모델 초기화")
    config = RapidTableInput(
        model_type=ModelType.UNET,
        use_ocr=False,  # OCR 없이 셀 검출만 수행
        model_dir_or_path=args.model_path,
    )
    
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"오류: 모델 파일을 찾을 수 없습니다: {args.model_path}")
            return
        print(f"모델 경로: {args.model_path}")
    else:
        print("모델 경로: 자동 다운로드")
    
    table_model = RapidTable(config)
    
    # 이미지 로드
    print(f"\n이미지 로드: {args.image}")
    img_bgr = load_image(args.image)
    
    # UNET inference 실행
    print("\n" + "=" * 80)
    print("UNET Inference 실행")
    print("=" * 80)
    
    # OCR 결과를 빈 배열로 전달 (OCR 없이 셀 검출만 수행)
    # RapidTable은 ocr_results를 (boxes, texts, scores) 튜플로 기대합니다
    ocr_results = ([], [], [])
    
    result = table_model(img_bgr, ocr_results=ocr_results)
    
    print(f"\n결과:")
    print(f"  HTML: {'생성됨' if result.pred_html else '없음'}")
    print(f"  Cell Bboxes: {result.cell_bboxes.shape if result.cell_bboxes is not None else 'None'}")
    print(f"  Logic Points: {result.logic_points if result.logic_points is not None else 'None'}")
    print(f"  Elapsed: {result.elapse:.4f}초")
    
    if result.cell_bboxes is not None:
        print(f"\n검출된 셀 개수: {len(result.cell_bboxes)}")
        
        # 처음 3개 셀 정보 출력
        num_to_show = min(3, len(result.cell_bboxes))
        for i in range(num_to_show):
            print(f"\n셀 {i+1}:")
            print(f"  Bbox: {result.cell_bboxes[i]}")
            if result.logic_points is not None and i < len(result.logic_points):
                logic_pt = result.logic_points[i] if isinstance(result.logic_points, list) else result.logic_points[i].tolist()
                print(f"  Logic: {logic_pt} (row_start, row_end, col_start, col_end)")
    
    # 결과 저장
    if args.save_output:
        print("\n" + "=" * 80)
        print("결과 저장")
        print("=" * 80)
        
        # 파일명 생성
        image_name = Path(args.image).stem
        output_filename = f"unet_output_{image_name}.json"
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
        
        output_data = {
            'pred_html': result.pred_html,
            'cell_bboxes': convert_to_serializable(result.cell_bboxes),
            'logic_points': convert_to_serializable(result.logic_points),
            'elapse': result.elapse,
            'num_cells': len(result.cell_bboxes) if result.cell_bboxes is not None else 0,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"저장 완료: {output_path}")
    
    # 시각화
    if args.visualize:
        print("\n" + "=" * 80)
        print("결과 시각화")
        print("=" * 80)
        
        # 파일명 생성
        image_name = Path(args.image).stem
        vis_filename = f"unet_vis_{image_name}.jpg"
        vis_path = os.path.join(output_dir, vis_filename)
        
        # 시각화 실행
        visualize_table_result(img_bgr, result.cell_bboxes, result.logic_points, vis_path)
    
    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)


if __name__ == '__main__':
    main()
