"""
OCR Recognition Model Inference Script

이 스크립트는 ONNX 형식의 OCR Recognition 모델을 사용하여 텍스트 인식을 수행합니다.
- PP-OCRv5 Recognition 모델 전처리 (mean=0.5, std=0.5)
- CTC 디코딩
- 텍스트 인식 결과 시각화

사용법:
    python value_compare/recognition/onnx_recognition.py \
        --model_path onnx_models/ch_PP-OCRv5_rec_server_infer.onnx \
        --image_path test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf \
        --output_dir value_compare/recognition/output
"""

import argparse
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode


class ONNXRecognitionInference:
    """ONNX Recognition 모델 추론기"""
    
    def __init__(
        self,
        model_path: str,
        dict_path: Optional[str] = None,
        input_height: int = 48,
        input_width: int = 640,
    ):
        """
        Args:
            model_path: ONNX 모델 경로
            dict_path: 문자 사전 경로 (None이면 기본 사전 사용)
            input_height: 입력 이미지 높이 (기본값: 48)
            input_width: 입력 이미지 최대 너비 (기본값: 640)
        """
        self.model_path = Path(model_path)
        self.input_height = input_height
        self.input_width = input_width
        self.rec_image_shape = [3, input_height, input_width]  # C, H, W
        
        # 정규화 파라미터 (PP-OCRv5 Recognition 표준)
        self.mean = 0.5
        self.std = 0.5
        
        # ONNX Runtime 세션 초기화
        self._init_session()
        
        # 문자 사전 로드 (우선순위: 모델 메타데이터 > 사용자 지정 > RapidOCR)
        self.character_dict = self._load_character_dict_from_model() or self._load_character_dict(dict_path)
        logger.info(f"✓ 문자 사전 로드 완료: {len(self.character_dict)} 문자")
        
        # RapidOCR의 CTCLabelDecode 사용
        self.ctc_decoder = CTCLabelDecode(character=self.character_dict)
        
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
        
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        
        logger.info(f"✓ 모델 로드 완료")
        logger.info(f"  - Input: {self.input_name}, shape: {input_shape}")
        logger.info(f"  - Output: {self.output_name}, shape: {output_shape}")
        
    def _load_character_dict_from_model(self) -> Optional[List[str]]:
        """
        모델 메타데이터에서 문자 사전 로드
        
        Returns:
            문자 리스트 (메타데이터 없으면 None)
        """
        try:
            metadata = self.session.get_modelmeta()
            if 'character' in metadata.custom_metadata_map:
                character_str = metadata.custom_metadata_map['character']
                char_list = character_str.split('\n')
                logger.info(f"✓ 모델 메타데이터에서 문자 사전 로드: {len(char_list)} 문자")
                return char_list
            return None
        except Exception as e:
            logger.warning(f"모델 메타데이터 읽기 실패: {e}")
            return None
    
    def _load_character_dict(self, dict_path: Optional[str] = None) -> List[str]:
        """
        문자 사전 로드
        
        Args:
            dict_path: 문자 사전 파일 경로
            
        Returns:
            문자 리스트
        """
        try:
            if dict_path and Path(dict_path).exists():
                # 사용자 지정 사전 로드
                with open(dict_path, 'r', encoding='utf-8') as f:
                    char_list = [line.strip() for line in f]
                logger.info(f"✓ 사용자 사전 로드: {dict_path}")
                return char_list
            
            # RapidOCR 기본 사전 경로 시도
            try:
                import rapidocr
                rapidocr_path = Path(rapidocr.__file__).parent
                
                # PP-OCRv5 사전 우선 (더 큰 사전)
                ppocrv5_dict = rapidocr_path / "models" / "ppocrv5_dict.txt"
                if ppocrv5_dict.exists():
                    with open(ppocrv5_dict, 'r', encoding='utf-8') as f:
                        char_list = [line.rstrip('\n\r') for line in f if line.strip()]
                    logger.info(f"✓ RapidOCR PP-OCRv5 사전 로드: {ppocrv5_dict} ({len(char_list)} 문자)")
                    return char_list
                
                # 기존 v1 사전 fallback
                default_dict = rapidocr_path / "models" / "ppocr_keys_v1.txt"
                if default_dict.exists():
                    with open(default_dict, 'r', encoding='utf-8') as f:
                        char_list = [line.strip() for line in f]
                    logger.info(f"✓ RapidOCR 기본 사전 로드: {default_dict} ({len(char_list)} 문자)")
                    return char_list
            except ImportError:
                pass
            
            # 기본 영숫자 사전
            logger.warning("문자 사전을 찾을 수 없습니다. 기본 영숫자 사전 사용")
            return list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            
        except Exception as e:
            logger.error(f"문자 사전 로드 실패: {e}")
            return list("0123456789abcdefghijklmnopqrstuvwxyz")
    
    def preprocess(self, img: np.ndarray, max_wh_ratio: Optional[float] = None) -> np.ndarray:
        """
        단일 이미지 전처리
        
        전처리 과정 (PP-OCRv5 Recognition 표준):
        1. Resize: aspect ratio 유지하며 높이 48로 조정
        2. Normalize: [0, 255] → [0, 1]
        3. Standardize: (img - 0.5) / 0.5 → [-1, 1]
        4. Transpose: (H, W, C) → (C, H, W)
        5. Padding: 오른쪽에 0으로 패딩 (동적 너비)
        
        Args:
            img: 입력 이미지 (H, W, C)
            max_wh_ratio: 최대 가로/세로 비율 (None이면 자동 계산)
            
        Returns:
            전처리된 이미지 (C, H, W)
        """
        imgC, imgH, imgW_base = self.rec_image_shape
        
        # 1. 최대 비율 계산 (동적 너비 조정)
        if max_wh_ratio is None:
            h, w = img.shape[:2]
            max_wh_ratio = max(imgW_base / imgH, w / float(h))
        
        # 동적 최대 너비 계산
        imgW = int(imgH * max_wh_ratio)
        
        # 2. Resize (aspect ratio 유지)
        h, w = img.shape[:2]
        ratio = w / float(h)
        
        # 새로운 너비 계산
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        
        # 3. 정규화 및 표준화
        # [0, 255] → [0, 1] → [-1, 1]
        resized_image = resized_image.astype(np.float32)
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0  # (C, H, W)
        resized_image -= self.mean  # mean subtraction
        resized_image /= self.std   # std division
        
        # 4. 패딩 (오른쪽에 0으로 패딩)
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        
        return padding_im
    
    def preprocess_batch(self, img_list: List[np.ndarray]) -> np.ndarray:
        """
        배치 이미지 전처리 (RapidOCR 방식 - 동적 너비 조정)
        
        Args:
            img_list: 입력 이미지 리스트
            
        Returns:
            전처리된 배치 이미지 (B, C, H, W)
        """
        imgC, imgH, imgW_base = self.rec_image_shape
        
        # 1. 배치 내 최대 Width/Height 비율 계산 (RapidOCR 방식)
        max_wh_ratio = imgW_base / imgH  # 기본값
        wh_ratio_list = []
        for img in img_list:
            h, w = img.shape[:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            wh_ratio_list.append(wh_ratio)
        
        # 2. 동적 최대 너비 계산
        imgW_dynamic = int(imgH * max_wh_ratio)
        
        logger.debug(f"배치 동적 너비 조정: 기본 {imgW_base}px → 동적 {imgW_dynamic}px (max_wh_ratio={max_wh_ratio:.2f})")
        
        batch_images = []
        
        for img in img_list:
            # 개별 이미지 전처리 (동적 너비 사용)
            norm_img = self.preprocess(img, max_wh_ratio=max_wh_ratio)
            batch_images.append(norm_img[np.newaxis, :])
        
        # 배치로 concat
        batch = np.concatenate(batch_images, axis=0).astype(np.float32)
        
        return batch
    
    def inference(self, img_input: np.ndarray) -> np.ndarray:
        """
        모델 추론
        
        Args:
            img_input: 전처리된 이미지 (B, C, H, W) 또는 (C, H, W)
            
        Returns:
            모델 출력 (B, T, num_classes)
        """
        # 배치 차원 추가 (필요한 경우)
        if len(img_input.shape) == 3:
            img_input = np.expand_dims(img_input, axis=0)
        
        start_time = time.perf_counter()
        
        # ONNX Runtime 추론
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: img_input}
        )
        
        inference_time = time.perf_counter() - start_time
        logger.debug(f"추론 시간: {inference_time*1000:.2f}ms")
        
        return outputs[0]
    
    def postprocess(self, preds: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        후처리: CTC 디코딩 (RapidOCR의 CTCLabelDecode 사용)
        
        Args:
            preds: 모델 출력 (B, T, num_classes)
            
        Returns:
            (텍스트 리스트, 신뢰도 리스트)
        """
        if preds is None or len(preds) == 0:
            return [], []
        
        # RapidOCR의 CTCLabelDecode 사용
        # return_word_box=False이므로 word_results는 빈 리스트
        line_results, _ = self.ctc_decoder(preds, return_word_box=False)
        
        # line_results: List[Tuple[str, float]] = [(text, confidence), ...]
        texts = [text for text, _ in line_results]
        scores = [score for _, score in line_results]
        
        return texts, scores
    
    def __call__(self, img_list: List[np.ndarray]) -> Tuple[List[str], List[float], float]:
        """
        텍스트 인식 실행
        
        Args:
            img_list: 입력 이미지 리스트
            
        Returns:
            (텍스트 리스트, 신뢰도 리스트, 경과 시간)
        """
        start_time = time.perf_counter()
        
        if not img_list:
            return [], [], 0.0
        
        # 1. 전처리
        batch_input = self.preprocess_batch(img_list)
        logger.debug(f"전처리 완료: {batch_input.shape}")
        
        # 2. 추론
        preds = self.inference(batch_input)
        logger.debug(f"추론 완료: {preds.shape}")
        
        # 3. 후처리
        texts, scores = self.postprocess(preds)
        
        elapsed = time.perf_counter() - start_time
        
        return texts, scores, elapsed


def visualize_results(
    img_list: List[np.ndarray],
    texts: List[str],
    scores: List[float],
    output_dir: Path
):
    """
    인식 결과 시각화
    
    Args:
        img_list: 입력 이미지 리스트
        texts: 인식된 텍스트 리스트
        scores: 신뢰도 리스트
        output_dir: 출력 디렉토리
    """
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (img, text, score) in enumerate(zip(img_list, texts, scores)):
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        
        # 이미지 표시
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        ax.imshow(img_rgb)
        ax.set_title(f'Text: "{text}" (score: {score:.3f})', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        
        # 저장
        output_path = output_dir / f"recognition_result_{idx:03d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='OCR Recognition Model Inference')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='ONNX 모델 경로 (예: onnx_models/ch_PP-OCRv5_rec_server_infer.onnx)'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='입력 이미지 경로 (텍스트 라인 이미지)'
    )
    parser.add_argument(
        '--dict_path',
        type=str,
        default=None,
        help='문자 사전 경로 (선택사항)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='value_compare/recognition/output',
        help='출력 디렉토리'
    )
    parser.add_argument(
        '--input_height',
        type=int,
        default=48,
        help='입력 이미지 높이 (기본값: 48)'
    )
    parser.add_argument(
        '--input_width',
        type=int,
        default=640,
        help='입력 이미지 최대 너비 (기본값: 640)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='결과 시각화 여부'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # 초기화
    # =========================================================================
    logger.info("=" * 80)
    logger.info("OCR Recognition Model Inference")
    logger.info("=" * 80)
    
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        logger.error(f"모델을 찾을 수 없습니다: {model_path}")
        return
    
    if not image_path.exists():
        logger.error(f"이미지를 찾을 수 없습니다: {image_path}")
        return
    
    # 인식기 초기화
    recognizer = ONNXRecognitionInference(
        model_path=str(model_path),
        dict_path=args.dict_path,
        input_height=args.input_height,
        input_width=args.input_width
    )
    
    # =========================================================================
    # 이미지 로드
    # =========================================================================
    logger.info(f"\n이미지 로딩: {image_path}")
    img = cv2.imread(str(image_path))
    
    if img is None:
        logger.error(f"이미지 로드 실패: {image_path}")
        return
    
    logger.info(f"✓ 이미지 크기: {img.shape}")
    
    # =========================================================================
    # 텍스트 인식
    # =========================================================================
    logger.info("\n텍스트 인식 시작...")
    
    texts, scores, elapsed = recognizer([img])
    
    logger.info(f"✓ 인식 완료 (소요 시간: {elapsed*1000:.2f}ms)")
    
    # =========================================================================
    # 결과 출력
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("인식 결과:")
    logger.info("=" * 80)
    
    for i, (text, score) in enumerate(zip(texts, scores)):
        logger.info(f"[{i}] Text: \"{text}\"")
        logger.info(f"    Score: {score:.4f}")
    
    # =========================================================================
    # 시각화 (선택사항)
    # =========================================================================
    if args.visualize:
        logger.info("\n결과 시각화 중...")
        visualize_results([img], texts, scores, output_dir)
        logger.info(f"✓ 시각화 완료: {output_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 완료")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
