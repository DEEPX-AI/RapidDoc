import numpy as np
import cv2
import argparse
import json
import os
import copy
import math
import sys
from pathlib import Path
from dx_engine import InferenceEngine, InferenceOption
from rapid_doc.utils.device_utils import get_dxnn_devices
from skimage import measure

# 프로젝트 루트를 sys.path에 추가하여 rapid_doc 모듈 import 가능하게
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ONNX의 table line 관련 유틸리티 import
from rapid_doc.model.table.rapid_table_self.wired_table_rec.utils.utils_table_line_rec import (
    get_table_line,
    final_adjust_lines,
    min_area_rect_box,
    draw_lines,
    adjust_lines,
)
from rapid_doc.model.table.rapid_table_self.wired_table_rec.utils.utils_table_recover import (
    sorted_ocr_boxes,
    box_4_2_poly_to_box_4_1,
)


def resize_with_padding(img: np.ndarray, target_size: int = 768, pad_color: tuple = (255, 255, 255)):
    """
    Aspect ratio를 유지하면서 패딩으로 정사각형으로 만듭니다.
    
    Args:
        img: 입력 이미지 (H, W, C)
        target_size: 목표 크기 (정사각형)
        pad_color: 패딩 색상 (B, G, R) - 기본값은 흰색
        
    Returns:
        tuple: (padded_img, scale, pad_top, pad_left, original_h, original_w)
            - padded_img: 패딩이 추가된 이미지 (target_size, target_size, C)
            - scale: 리사이즈 비율
            - pad_top: 위쪽 패딩 크기
            - pad_left: 왼쪽 패딩 크기
            - original_h: 원본 이미지 높이
            - original_w: 원본 이미지 너비
    """
    h, w = img.shape[:2]
    
    # 긴 쪽을 기준으로 스케일 계산
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 리사이즈 (축소 시 INTER_AREA, 확대 시 INTER_CUBIC 사용)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # 패딩 계산 (중앙 정렬)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # 패딩 추가
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )
    
    return padded, scale, pad_top, pad_left, h, w


def unpad_coordinates(coords, scale, pad_top, pad_left, clip_to_original=True, original_h=None, original_w=None):
    """
    패딩된 이미지의 좌표를 원본 이미지 좌표로 변환합니다.
    
    Args:
        coords: 좌표 배열 (N, 8) - [x1, y1, x2, y2, x3, y3, x4, y4] 또는 (N, 4) - [x1, y1, x2, y2]
        scale: resize_with_padding에서 반환된 스케일
        pad_top: 위쪽 패딩 크기
        pad_left: 왼쪽 패딩 크기
        clip_to_original: 원본 이미지 범위로 클리핑 여부
        original_h: 원본 이미지 높이 (clip_to_original=True일 때 필요)
        original_w: 원본 이미지 너비 (clip_to_original=True일 때 필요)
        
    Returns:
        np.ndarray: 원본 이미지 좌표계로 변환된 좌표
    """
    if coords is None or len(coords) == 0:
        return coords
    
    coords = np.array(coords, dtype=np.float32)
    original_shape = coords.shape
    
    # 1차원 배열을 2차원으로 변환
    if len(coords.shape) == 1:
        coords = coords.reshape(1, -1)
    
    # x, y 좌표 분리
    if coords.shape[1] == 8:
        # [x1, y1, x2, y2, x3, y3, x4, y4]
        x_coords = coords[:, [0, 2, 4, 6]]
        y_coords = coords[:, [1, 3, 5, 7]]
    elif coords.shape[1] == 4:
        # [x1, y1, x2, y2]
        x_coords = coords[:, [0, 2]]
        y_coords = coords[:, [1, 3]]
    else:
        raise ValueError(f"Unsupported coordinate shape: {coords.shape}")
    
    # 패딩 제거 및 스케일 복원
    x_coords = (x_coords - pad_left) / scale
    y_coords = (y_coords - pad_top) / scale
    
    # 원본 이미지 범위로 클리핑
    if clip_to_original and original_h is not None and original_w is not None:
        x_coords = np.clip(x_coords, 0, original_w)
        y_coords = np.clip(y_coords, 0, original_h)
    
    # 좌표 재결합
    if coords.shape[1] == 8:
        result = np.stack([
            x_coords[:, 0], y_coords[:, 0],
            x_coords[:, 1], y_coords[:, 1],
            x_coords[:, 2], y_coords[:, 2],
            x_coords[:, 3], y_coords[:, 3]
        ], axis=1)
    else:  # shape[1] == 4
        result = np.stack([
            x_coords[:, 0], y_coords[:, 0],
            x_coords[:, 1], y_coords[:, 1]
        ], axis=1)
    
    # 원래 shape으로 복원
    return result.reshape(original_shape)


def preprocess(image: np.ndarray, target_size: int = 768) -> dict:
    """
    Padding resize를 사용한 이미지 전처리 함수.
    
    Args:
        image: 입력 이미지 (numpy ndarray)
        target_size: 목표 크기 (정사각형, 기본값: 768)
        
    Returns:
        dict: {
            'img': 전처리된 이미지,
            'scale': 리사이즈 비율,
            'pad_top': 위쪽 패딩,
            'pad_left': 왼쪽 패딩,
            'original_h': 원본 높이,
            'original_w': 원본 너비
        }
    """
    img, scale, pad_top, pad_left, h, w = resize_with_padding(
        image, 
        target_size=target_size
    )
    
    # 정규화 전에 float32로 변환
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    images = img[None, :]
    
    return {
        'img': images,
        'scale': scale,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'original_h': h,
        'original_w': w
    }


def extract_lines_from_mask(pred: np.ndarray, axis: int = 0, threshold: int = 5):
    """
    세그멘테이션 마스크에서 가로선 또는 세로선을 추출합니다.
    
    Args:
        pred: 세그멘테이션 마스크 (H, W)
        axis: 0=가로선, 1=세로선
        threshold: 선으로 판단할 최소 길이
        
    Returns:
        list: 선의 위치 리스트
    """
    # axis 방향으로 투영
    projection = np.sum(pred > 0, axis=axis)
    
    # 선이 있는 위치 찾기
    lines = []
    in_line = False
    start = 0
    
    for i, val in enumerate(projection):
        if val > threshold and not in_line:
            start = i
            in_line = True
        elif val <= threshold and in_line:
            lines.append((start, i))
            in_line = False
    
    if in_line:
        lines.append((start, len(projection)))
    
    return lines


def extract_cell_bboxes_from_lines(h_lines: list, v_lines: list, img_shape: tuple):
    """
    가로선과 세로선의 교차점으로부터 셀 바운딩 박스를 생성합니다.
    
    Args:
        h_lines: 가로선 리스트 [(y_start, y_end), ...]
        v_lines: 세로선 리스트 [(x_start, x_end), ...]
        img_shape: 이미지 shape (H, W)
        
    Returns:
        tuple: (cell_bboxes, logic_points)
            - cell_bboxes: (N, 8) array [x1, y1, x2, y2, x3, y3, x4, y4]
            - logic_points: (N, 4) array [row_start, row_end, col_start, col_end]
    """
    if len(h_lines) < 2 or len(v_lines) < 2:
        return np.array([]), np.array([])
    
    # 선의 중심점 계산
    h_positions = [(start + end) // 2 for start, end in h_lines]
    v_positions = [(start + end) // 2 for start, end in v_lines]
    
    cell_bboxes = []
    logic_points = []
    
    # 각 셀 생성
    for i in range(len(h_positions) - 1):
        for j in range(len(v_positions) - 1):
            y1, y2 = h_positions[i], h_positions[i + 1]
            x1, x2 = v_positions[j], v_positions[j + 1]
            
            # 4점 좌표 [x1, y1, x2, y1, x2, y2, x1, y2]
            bbox = [x1, y1, x2, y1, x2, y2, x1, y2]
            cell_bboxes.append(bbox)
            
            # 논리적 위치 [row_start, row_end, col_start, col_end]
            logic = [i, i + 1, j, j + 1]
            logic_points.append(logic)
    
    return np.array(cell_bboxes, dtype=np.float32), np.array(logic_points, dtype=np.int32)


def extract_connected_components(pred: np.ndarray, min_area: int = 100):
    """
    연결된 영역(connected components)을 찾아 바운딩 박스를 추출합니다.
    
    Args:
        pred: 세그멘테이션 마스크 (H, W)
        min_area: 최소 영역 크기
        
    Returns:
        list: 바운딩 박스 리스트 [(x1, y1, x2, y2), ...]
    """
    # 이진화
    binary = (pred > 0).astype(np.uint8) * 255
    
    # Connected components 찾기
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):  # 0은 배경
        x, y, w, h, area = stats[i]
        
        if area >= min_area:
            # (x, y, w, h) -> (x1, y1, x2, y2)
            bboxes.append([x, y, x + w, y + h])
    
    return bboxes


def postprocess_unet_onnx_style(pred: np.ndarray, original_shape: tuple, scale: float, pad_top: int, pad_left: int, 
                                 **kwargs):
    """
    UNET 모델 출력을 ONNX TSRUnet.postprocess 방식으로 후처리합니다.
    
    Args:
        pred: 모델 출력 (1, H, W) 또는 (H, W) - 세그멘테이션 마스크
        original_shape: 원본 이미지 shape (H, W, C)
        scale: resize_with_padding에서 반환된 스케일
        pad_top: 위쪽 패딩 크기
        pad_left: 왼쪽 패딩 크기
        **kwargs: TSRUnet.postprocess와 동일한 파라미터들
        
    Returns:
        dict: {
            'cell_bboxes': 셀 바운딩 박스 (N, 8),
            'logic_points': 논리적 위치 정보 (N, 4),
            'pred_mask': 원본 예측 마스크,
            'num_cells': 셀 개수
        }
    """
    # shape 정규화
    if len(pred.shape) == 4:
        pred = pred[0, 0]  # (1, 1, H, W) -> (H, W)
    elif len(pred.shape) == 3:
        pred = pred[0]  # (1, H, W) -> (H, W)
    
    # uint8로 변환
    pred_uint8 = (pred * 255).astype(np.uint8) if pred.max() <= 1.0 else pred.astype(np.uint8)
    
    # ONNX TSRUnet.postprocess와 동일한 파라미터
    row = kwargs.get("row", 50)
    col = kwargs.get("col", 30)
    h_lines_threshold = kwargs.get("h_lines_threshold", 100)
    v_lines_threshold = kwargs.get("v_lines_threshold", 15)
    angle = kwargs.get("angle", 50)
    enhance_box_line = kwargs.get("enhance_box_line", True)
    morph_close = kwargs.get("morph_close", enhance_box_line)
    more_h_lines = kwargs.get("more_h_lines", enhance_box_line)
    more_v_lines = kwargs.get("more_v_lines", enhance_box_line)
    extend_line = kwargs.get("extend_line", enhance_box_line)
    rotated_fix = kwargs.get("rotated_fix", True)
    
    # 클래스 분리: 0=배경, 1=가로선, 2=세로선
    hpred = copy.deepcopy(pred_uint8)
    vpred = copy.deepcopy(pred_uint8)
    whereh = np.where(hpred == 1)
    wherev = np.where(vpred == 2)
    hpred[wherev] = 0
    vpred[whereh] = 0
    
    # 원본 이미지 크기로 resize (패딩 제거)
    # 패딩된 영역 제거
    if pad_top > 0 or pad_left > 0:
        h_end = int((original_shape[0] * scale) + pad_top)
        w_end = int((original_shape[1] * scale) + pad_left)
        hpred = hpred[pad_top:h_end, pad_left:w_end]
        vpred = vpred[pad_top:h_end, pad_left:w_end]
    
    # Morphological operations용 kernel 크기 계산
    # ONNX와 동일하게 패딩 제거 후의 실제 크기로 계산해야 함
    h, w = hpred.shape  # 패딩 제거 후의 shape 사용!
    hors_k = int(math.sqrt(w) * 1.2)
    vert_k = int(math.sqrt(h) * 1.2)
    
    # 원본 크기로 resize
    hpred = cv2.resize(hpred, (original_shape[1], original_shape[0]))
    vpred = cv2.resize(vpred, (original_shape[1], original_shape[0]))
    
    # Morphological operations
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
    vpred = cv2.morphologyEx(vpred, cv2.MORPH_CLOSE, vkernel, iterations=1)
    if morph_close:
        hpred = cv2.morphologyEx(hpred, cv2.MORPH_CLOSE, hkernel, iterations=1)
    
    # 선 추출
    colboxes = get_table_line(vpred, axis=1, lineW=col)  # 세로선
    rowboxes = get_table_line(hpred, axis=0, lineW=row)  # 가로선
    
    rboxes_row_, rboxes_col_ = [], []
    if more_h_lines:
        rboxes_row_ = adjust_lines(rowboxes, alph=h_lines_threshold, angle=angle)
    if more_v_lines:
        rboxes_col_ = adjust_lines(colboxes, alph=v_lines_threshold, angle=angle)
    rowboxes += rboxes_row_
    colboxes += rboxes_col_
    
    if extend_line:
        rowboxes, colboxes = final_adjust_lines(rowboxes, colboxes)
    
    # 선 그리기
    line_img = np.zeros(original_shape[:2], dtype="uint8")
    line_img = draw_lines(line_img, rowboxes + colboxes, color=255, lineW=2)
    
    # Connected components로 셀 영역 찾기
    labels = measure.label(line_img < 255, connectivity=2)
    regions = measure.regionprops(labels)
    cell_bboxes = min_area_rect_box(
        regions,
        False,
        original_shape[1],
        original_shape[0],
        filtersmall=True,
        adjust_box=False,
    )
    cell_bboxes = np.array(cell_bboxes)
    
    # 셀 정렬
    if len(cell_bboxes) > 0:
        cell_bboxes = cell_bboxes.reshape(cell_bboxes.shape[0], 4, 2)
        cell_bboxes[:, 3, :], cell_bboxes[:, 1, :] = (
            cell_bboxes[:, 1, :].copy(),
            cell_bboxes[:, 3, :].copy(),
        )
        _, idx = sorted_ocr_boxes(
            [box_4_2_poly_to_box_4_1(poly_box) for poly_box in cell_bboxes],
            threhold=0.4,
        )
        cell_bboxes = cell_bboxes[idx]
        cell_bboxes = cell_bboxes.reshape(-1, 8)
        
        # Logic points 생성 (간단한 방식: 각 셀을 순서대로 번호 부여)
        # ONNX와 동일하게 [row_idx, row_idx, 0, 0] 형식 사용
        logic_points = [[i, i, 0, 0] for i in range(len(cell_bboxes))]
    else:
        cell_bboxes = np.array([])
        logic_points = None
    
    return {
        'cell_bboxes': cell_bboxes,
        'logic_points': logic_points,
        'pred_mask': pred_uint8,
        'num_cells': len(cell_bboxes) if len(cell_bboxes) > 0 else 0,
        'rowboxes': rowboxes,
        'colboxes': colboxes,
    }


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

def run_dxnn_model(model_path: str, input_data: np.ndarray, target_size: int = 768):
    """
    DXNN 모델을 로드하고 추론을 수행합니다.
    
    Args:
        model_path: DXNN 모델 파일 경로
        input_data: 입력 데이터 (numpy ndarray)
        target_size: 목표 크기 (정사각형)
        
    Returns:
        dict: {
            'output': 모델 출력,
            'preprocess_info': 전처리 정보
        }
    """
    # 전처리
    preprocess_result = preprocess(input_data, target_size=target_size)
    
    io = InferenceOption()
    io.devices = get_dxnn_devices()
    io.bound_option = InferenceOption.BOUND_OPTION.NPU_ALL

    # DX Engine 추론
    session = InferenceEngine(model_path, io)
    output = session.run([preprocess_result['img']])
    
    return {
        'output': output,
        'scale': preprocess_result['scale'],
        'pad_top': preprocess_result['pad_top'],
        'pad_left': preprocess_result['pad_left'],
        'original_h': preprocess_result['original_h'],
        'original_w': preprocess_result['original_w']
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DX Engine UNET 테이블 모델의 inference 결과를 확인합니다.'
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
        required=True,
        help='UNET 모델 파일 경로 (.dxnn 파일)'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        default=768,
        help='목표 크기 (정사각형, 기본값: 768)'
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
    parser.add_argument(
        '--save_pred_mask',
        action='store_true',
        help='예측 마스크를 이미지로 저장'
    )
    parser.add_argument(
        '--postprocess',
        action='store_true',
        help='후처리를 수행하여 셀 바운딩 박스 추출'
    )
    parser.add_argument(
        '--postprocess_method',
        type=str,
        default='lines',
        choices=['lines', 'components'],
        help='후처리 방법: lines (가로/세로선 기반, 권장) 또는 components (연결 영역 기반)'
    )
    
    args = parser.parse_args()
    
    # 이미지 경로 검증
    if not os.path.exists(args.image):
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {args.image}")
        exit(1)
    
    # 모델 경로 검증
    if not os.path.exists(args.model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {args.model_path}")
        exit(1)
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        output_dir = os.path.join(Path(__file__).parent.parent.parent, 'value_compare', 'output')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DX Engine UNET 테이블 모델 Inference 결과 확인")
    print("=" * 80)
    
    # 이미지 로드
    img = cv2.imread(args.image)
    if img is None:
        print(f"오류: 이미지를 로드할 수 없습니다: {args.image}")
        exit(1)
    
    print(f"입력 이미지: {args.image}")
    print(f"원본 크기: {img.shape[1]}x{img.shape[0]} (W x H)")
    print(f"모델 경로: {args.model_path}")
    print(f"목표 크기: {args.target_size}x{args.target_size}")
    print()
    
    # UNET inference 실행
    print("=" * 80)
    print("DX Engine Inference 실행")
    print("=" * 80)
    
    import time
    start_time = time.time()
    result = run_dxnn_model(args.model_path, img, target_size=args.target_size)
    elapsed_time = time.time() - start_time
    
    output = result['output']
    print(f"\n결과:")
    print(f"  모델 출력 shape: {output[0].shape if isinstance(output, list) else output.shape}")
    print(f"  Scale: {result['scale']:.4f}")
    print(f"  Padding (top, left): ({result['pad_top']}, {result['pad_left']})")
    print(f"  원본 크기: {result['original_w']}x{result['original_h']}")
    print(f"  Elapsed: {elapsed_time:.4f}초")
    print()
    
    # 예측 마스크 저장
    pred_mask = output[0] if isinstance(output, list) else output
    if args.save_pred_mask and output is not None:
        if len(pred_mask.shape) == 4:
            pred_mask_vis = pred_mask[0, 0]
        elif len(pred_mask.shape) == 3:
            pred_mask_vis = pred_mask[0]
        else:
            pred_mask_vis = pred_mask
        
        # uint8로 변환 (시각화용)
        if pred_mask_vis.max() <= 1.0:
            pred_mask_vis = (pred_mask_vis * 255).astype(np.uint8)
        else:
            pred_mask_vis = pred_mask_vis.astype(np.uint8)
        
        image_name = Path(args.image).stem
        mask_filename = f"unet_pred_mask_{image_name}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, pred_mask_vis)
        print(f"예측 마스크 저장: {mask_path}")
        print()
    
    # 후처리 수행
    postprocess_result = None
    if args.postprocess or args.visualize:
        print("=" * 80)
        print("후처리 실행")
        print("=" * 80)
        print(f"방법: {args.postprocess_method}")
        
        postprocess_start = time.time()
        postprocess_result = postprocess_unet_onnx_style(
            pred_mask,
            img.shape,
            result['scale'],
            result['pad_top'],
            result['pad_left']
        )
        postprocess_time = time.time() - postprocess_start
        
        print(f"\n후처리 결과:")
        print(f"  검출된 셀 개수: {postprocess_result['num_cells']}")
        if 'rowboxes' in postprocess_result:
            print(f"  가로선 개수: {len(postprocess_result['rowboxes'])}")
            print(f"  세로선 개수: {len(postprocess_result['colboxes'])}")
        print(f"  Cell Bboxes shape: {postprocess_result['cell_bboxes'].shape if len(postprocess_result['cell_bboxes']) > 0 else 'None'}")
        if postprocess_result['logic_points'] is not None:
            lp = postprocess_result['logic_points']
            lp_len = len(lp) if isinstance(lp, list) else lp.shape[0]
            print(f"  Logic Points: {lp_len}개")
        print(f"  후처리 시간: {postprocess_time:.4f}초")
        print()
        
        # 처음 3개 셀 정보 출력
        if postprocess_result['num_cells'] > 0:
            num_to_show = min(3, postprocess_result['num_cells'])
            print(f"처음 {num_to_show}개 셀 정보:")
            for i in range(num_to_show):
                print(f"\n셀 {i+1}:")
                print(f"  Bbox: {postprocess_result['cell_bboxes'][i]}")
                if postprocess_result['logic_points'] is not None:
                    logic_pt = postprocess_result['logic_points'][i]
                    print(f"  Logic: {logic_pt} (row_start, row_end, col_start, col_end)")
            print()
    
    # 결과 저장
    if args.save_output:
        print("=" * 80)
        print("결과 저장")
        print("=" * 80)
        
        image_name = Path(args.image).stem
        output_filename = f"dxnn_unet_output_{image_name}.json"
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
            'model_output_shape': str(output[0].shape if isinstance(output, list) else output.shape),
            'scale': result['scale'],
            'pad_top': result['pad_top'],
            'pad_left': result['pad_left'],
            'original_h': result['original_h'],
            'original_w': result['original_w'],
            'elapsed': elapsed_time,
        }
        
        # 후처리 결과 추가
        if postprocess_result is not None:
            output_data['postprocess'] = {
                'method': args.postprocess_method,
                'num_cells': postprocess_result['num_cells'],
                'cell_bboxes': convert_to_serializable(postprocess_result['cell_bboxes']),
                'logic_points': convert_to_serializable(postprocess_result['logic_points']),
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"저장 완료: {output_path}")
        print()
    
    # 시각화
    if args.visualize:
        print("=" * 80)
        print("결과 시각화")
        print("=" * 80)
        
        if postprocess_result is not None and postprocess_result['num_cells'] > 0:
            image_name = Path(args.image).stem
            vis_filename = f"dxnn_unet_vis_{image_name}.jpg"
            vis_path = os.path.join(output_dir, vis_filename)
            
            # 시각화 실행
            visualize_table_result(
                img, 
                postprocess_result['cell_bboxes'], 
                postprocess_result['logic_points'], 
                vis_path
            )
        else:
            print("시각화할 셀이 없습니다. --postprocess 옵션을 사용하세요.")
            print()
    
    print("=" * 80)
    print("완료!")
    print("=" * 80)
    print("\n디버깅을 위한 변수:")
    print("  - img: 원본 이미지")
    print("  - result: inference 결과 (dict)")
    print("  - output: 모델 출력")
    