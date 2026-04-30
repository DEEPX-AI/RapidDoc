"""
OCR Detection Model Output Visualization Script

이 스크립트는 ONNX 형식의 OCR Detection 모델의 출력을 시각화합니다.
- Probability map 시각화
- Detection boxes 시각화
- 입력 이미지와 비교 시각화

사용법:
    python value_compare/detect/onnx_detect.py \
        --model_path onnx_models/ch_PP-OCRv5_server_det.onnx \
        --image_path test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf \
        --output_dir value_compare/detect/output
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from loguru import logger
import pyclipper
from shapely.geometry import Polygon


class ONNXDetectionVisualizer:
    """ONNX Detection 모델 출력 시각화기"""
    
    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        box_thresh: float = 0.3,
        unclip_ratio: float = 1.8,
    ):
        """
        Args:
            model_path: ONNX 모델 경로
            input_size: 입력 이미지 크기 (640x640)
            box_thresh: 박스 임계값
            unclip_ratio: 박스 확장 비율
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        
        # 정규화 파라미터 (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # ONNX Runtime 세션 초기화
        self._init_session()
        
    def _init_session(self):
        """ONNX Runtime 세션 초기화"""
        logger.info(f"ONNX 모델 로딩: {self.model_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # 입력/출력 정보
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"✓ 모델 로드 완료")
        logger.info(f"  - Input: {self.input_name}")
        logger.info(f"  - Output: {self.output_name}")
        
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        이미지 전처리
        
        Args:
            img: 입력 이미지 (H, W, C)
            
        Returns:
            전처리된 이미지 (1, C, H, W), 원본 크기 (H, W)
        """
        # 원본 크기 저장
        ori_h, ori_w = img.shape[:2]
        
        # 1. 고정 크기로 리사이즈 (640x640)
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # 2. 정규화 (0-255 -> 0-1)
        img_float = img_resized.astype(np.float32) / 255.0
        img_float -= self.mean
        img_float /= self.std
        
        # 3. Transpose (H, W, C) -> (C, H, W)
        img_transposed = img_float.transpose(2, 0, 1)
        
        # 4. 배치 차원 추가 (C, H, W) -> (1, C, H, W)
        img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        
        return img_batch, (ori_h, ori_w)
    
    def _unclip(self, box: np.ndarray, unclip_ratio: float) -> Optional[np.ndarray]:
        """
        박스 확장 (unclip)
        
        Args:
            box: 입력 박스 좌표
            unclip_ratio: 확장 비율
            
        Returns:
            확장된 박스
        """
        try:
            poly = Polygon(box)
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = np.array(offset.Execute(distance))
            
            if len(expanded) == 0:
                return None
            
            return expanded[0]
        except Exception as e:
            logger.debug(f"Unclip 오류: {e}")
            return None
    
    def _get_mini_boxes(self, contour: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        최소 외접 사각형 및 면적 계산
        
        Args:
            contour: 윤곽선
            
        Returns:
            박스 좌표 (4, 2), 최소 변 길이
        """
        try:
            bounding_box = cv2.minAreaRect(contour)
            points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
            
            if points[1][1] > points[0][1]:
                index_1, index_4 = 0, 1
            else:
                index_1, index_4 = 1, 0
                
            if points[3][1] > points[2][1]:
                index_2, index_3 = 2, 3
            else:
                index_2, index_3 = 3, 2
            
            box = np.array([
                points[index_1],
                points[index_2],
                points[index_3],
                points[index_4]
            ]).astype(np.float32)
            
            # 최소 변 길이 계산
            side_1 = np.linalg.norm(box[0] - box[1])
            side_2 = np.linalg.norm(box[1] - box[2])
            min_side = min(side_1, side_2)
            
            return box, min_side
            
        except Exception as e:
            logger.debug(f"Mini boxes 오류: {e}")
            return None, 0
    
    def _box_score_fast(self, bitmap: np.ndarray, box: np.ndarray) -> float:
        """
        박스 내부의 평균 점수 계산
        
        Args:
            bitmap: Probability map
            box: 박스 좌표
            
        Returns:
            평균 점수
        """
        try:
            h, w = bitmap.shape[:2]
            box = box.copy()
            
            xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
            xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
            ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
            ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)
            
            mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            
            box[:, 0] = box[:, 0] - xmin
            box[:, 1] = box[:, 1] - ymin
            
            cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
            
            return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
            
        except Exception as e:
            logger.debug(f"Box score 오류: {e}")
            return 0.0
    
    def postprocess(
        self,
        preds: np.ndarray,
        ori_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        후처리: probability map에서 텍스트 박스 추출 (DB 표준 후처리)
        
        Args:
            preds: 모델 예측 결과 (1, 1, 640, 640)
            ori_shape: 원본 이미지 크기 (H, W)
            
        Returns:
            텍스트 박스 좌표 배열 (N, 4, 2) - 원본 이미지 좌표계
        """
        try:
            # (1, 1, H, W) -> (H, W)
            pred = preds[0, 0, :, :]
            bitmap = pred > self.box_thresh
            
            # Contour 찾기
            contours, _ = cv2.findContours(
                (bitmap * 255).astype(np.uint8),
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            boxes = []
            ori_h, ori_w = ori_shape
            
            # 스케일 비율 계산 (640x640 -> 원본 크기)
            ratio_h = ori_h / self.input_size
            ratio_w = ori_w / self.input_size
            
            for contour in contours:
                # 너무 작은 contour는 무시
                if contour.shape[0] < 4:
                    continue
                
                # 점수 계산
                score = self._box_score_fast(pred, contour.squeeze())
                if score < self.box_thresh:
                    continue
                
                # 최소 외접 사각형 구하기
                box, min_side = self._get_mini_boxes(contour.squeeze())
                if box is None or min_side < 3:
                    continue
                
                # Unclip (박스 확장)
                if self.unclip_ratio > 0:
                    box = self._unclip(box, self.unclip_ratio)
                    if box is None:
                        continue
                    
                    box, min_side = self._get_mini_boxes(box)
                    if box is None or min_side < 3:
                        continue
                
                # 640x640 좌표계를 원본 이미지 좌표계로 변환
                box[:, 0] = np.clip(box[:, 0] * ratio_w, 0, ori_w)
                box[:, 1] = np.clip(box[:, 1] * ratio_h, 0, ori_h)
                
                boxes.append(box)
            
            if len(boxes) == 0:
                return None
                
            return np.array(boxes, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"후처리 오류: {e}")
            return None
    
    def detect(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        """
        텍스트 검출 실행
        
        Args:
            img: 입력 이미지
            
        Returns:
            boxes: 검출된 텍스트 박스 (N, 4, 2)
            prob_map: Probability map (640, 640)
            elapse: 추론 시간
        """
        start_time = time.time()
        
        # 전처리
        preprocessed, ori_shape = self.preprocess(img)
        
        # 추론
        preds = self.session.run(
            [self.output_name],
            {self.input_name: preprocessed}
        )[0]
        
        # 후처리
        boxes = self.postprocess(preds, ori_shape)
        
        elapse = time.time() - start_time
        
        # Probability map 추출 (시각화용)
        prob_map = preds[0, 0, :, :]
        
        return boxes, prob_map, elapse
    
    def visualize(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray],
        prob_map: np.ndarray,
        output_path: Path
    ):
        """
        검출 결과 시각화
        
        Args:
            img: 원본 이미지
            boxes: 검출된 텍스트 박스
            prob_map: Probability map
            output_path: 출력 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # 1. 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Probability Map (Heatmap)
        im = axes[0, 1].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title('Probability Map (Model Output)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Thresholded Map
        thresholded = (prob_map > self.box_thresh).astype(np.uint8) * 255
        axes[1, 0].imshow(thresholded, cmap='gray')
        axes[1, 0].set_title(f'Thresholded Map (threshold={self.box_thresh})', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 4. Detection Result (Boxes on Original)
        img_with_boxes = img.copy()
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                box = box.astype(np.int32)
                cv2.polylines(img_with_boxes, [box], True, (0, 255, 0), 2)
        
        axes[1, 1].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Detection Result ({len(boxes) if boxes is not None else 0} boxes)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 시각화 저장: {output_path}")
    
    def save_individual_outputs(
        self,
        img: np.ndarray,
        boxes: Optional[np.ndarray],
        prob_map: np.ndarray,
        output_dir: Path,
        prefix: str = "output"
    ):
        """
        개별 출력 저장 (디버깅용)
        
        Args:
            img: 원본 이미지
            boxes: 검출된 텍스트 박스
            prob_map: Probability map
            output_dir: 출력 디렉토리
            prefix: 파일명 prefix
        """
        # 1. Probability map 저장 (numpy)
        prob_map_path = output_dir / f"{prefix}_prob_map.npy"
        np.save(str(prob_map_path), prob_map)
        logger.info(f"✓ Probability map 저장: {prob_map_path}")
        
        # 2. Probability map 이미지 저장
        prob_map_img = (prob_map * 255).astype(np.uint8)
        prob_map_img_path = output_dir / f"{prefix}_prob_map.png"
        cv2.imwrite(str(prob_map_img_path), prob_map_img)
        
        # 3. Boxes 저장 (numpy)
        if boxes is not None:
            boxes_path = output_dir / f"{prefix}_boxes.npy"
            np.save(str(boxes_path), boxes)
            logger.info(f"✓ Boxes 저장: {boxes_path} (shape: {boxes.shape})")
        
        # 4. Boxes 이미지 저장
        img_with_boxes = img.copy()
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                box = box.astype(np.int32)
                cv2.polylines(img_with_boxes, [box], True, (0, 255, 0), 2)
                
                # 박스 번호 표시
                center = box.mean(axis=0).astype(np.int32)
                cv2.putText(
                    img_with_boxes,
                    str(i),
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )
        
        boxes_img_path = output_dir / f"{prefix}_boxes.png"
        cv2.imwrite(str(boxes_img_path), img_with_boxes)
        logger.info(f"✓ Boxes 이미지 저장: {boxes_img_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OCR Detection Model Output Visualization"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="ONNX 모델 경로 (예: onnx_models/ch_PP-OCRv5_server_det.onnx)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="입력 이미지 경로"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="value_compare/detect/output",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--box_thresh",
        type=float,
        default=0.3,
        help="박스 임계값 (default: 0.3)"
    )
    parser.add_argument(
        "--unclip_ratio",
        type=float,
        default=1.8,
        help="박스 확장 비율 (default: 1.8)"
    )
    
    args = parser.parse_args()
    
    # 경로 설정
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 존재 확인
    if not model_path.exists():
        logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    if not image_path.exists():
        logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    logger.info("=" * 80)
    logger.info("OCR Detection Model Output Visualization")
    logger.info("=" * 80)
    logger.info(f"모델: {model_path}")
    logger.info(f"이미지: {image_path}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"Box threshold: {args.box_thresh}")
    logger.info(f"Unclip ratio: {args.unclip_ratio}")
    logger.info("=" * 80)
    
    # 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    
    logger.info(f"이미지 크기: {img.shape[1]}x{img.shape[0]}")
    
    # Visualizer 초기화
    visualizer = ONNXDetectionVisualizer(
        model_path=str(model_path),
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio
    )
    
    # 검출 실행
    logger.info("\n검출 시작...")
    boxes, prob_map, elapse = visualizer.detect(img)
    
    logger.info(f"✓ 검출 완료")
    logger.info(f"  - 추론 시간: {elapse:.4f}초")
    logger.info(f"  - 검출된 박스 수: {len(boxes) if boxes is not None else 0}")
    logger.info(f"  - Probability map shape: {prob_map.shape}")
    logger.info(f"  - Probability map range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    
    # 통합 시각화
    output_filename = image_path.stem
    visualization_path = output_dir / f"{output_filename}_visualization.png"
    visualizer.visualize(img, boxes, prob_map, visualization_path)
    
    # 개별 출력 저장
    logger.info("\n개별 출력 저장 중...")
    visualizer.save_individual_outputs(
        img, boxes, prob_map, output_dir, prefix=output_filename
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ 모든 작업 완료!")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
