#!/usr/bin/env python3
"""
OCR Detection 모델의 순수 ONNX Runtime 추론 성능 측정 스크립트

전체 파이프라인 없이 OCR Detection 모델의 순수 추론 속도만 측정합니다.
"""
import time
import numpy as np
from pathlib import Path
from loguru import logger


def benchmark_ocr_det_dxnn():
    
    # =========================================================================
    # 모델 경로 설정
    # =========================================================================
    project_root = Path(__file__).parent.parent.absolute()
    ocr_det_model_path = project_root / "dxnn_models" / "ch_PP-OCRv5_server_det.dxnn"
    
    if not ocr_det_model_path.exists():
        logger.error(f"OCR Detection 모델을 찾을 수 없습니다: {ocr_det_model_path}")
        return
    
    logger.info("=" * 80)
    logger.info("OCR Detection 모델 순수 DXNN 추론 성능 측정")
    logger.info("=" * 80)
    logger.info(f"모델 경로: {ocr_det_model_path}")
    
    # =========================================================================
    # ONNX Runtime 세션 생성
    # =========================================================================
    from dx_engine import InferenceEngine
    
    
    
    logger.info("DXNN Runtime 세션 생성 중...")
    session_start = time.perf_counter()
    session = InferenceEngine(str(ocr_det_model_path))
    session_time = time.perf_counter() - session_start
    logger.info(f"✓ 세션 생성 완료: {session_time:.3f}초")
    
    # 입력/출력 정보 확인
    input_name = 'x'
    input_shape = [1, 3, 640, 640]
    input_dtype = np.uint8
    logger.info(f"입력 이름: {input_name}")
    logger.info(f"입력 shape: {input_shape}")
    logger.info(f"입력 dtype: {input_dtype}")

    output_names = ["fetch_name_0"]
    logger.info(f"출력 개수: {len(output_names)}")
    for i, name in enumerate(output_names):
        output_shape = [1, 1, 640, 640]
        logger.info(f"  출력 {i}: {name} - shape: {output_shape}")
    
    # =========================================================================
    # 테스트 이미지 생성 (고정 크기: 640x640)
    # =========================================================================
    height, width = 640, 640
    
    logger.info("\n" + "=" * 80)
    logger.info(f"성능 측정 (이미지 크기: {height}x{width}, 배치: 1)")
    logger.info("=" * 80)
    
    # 랜덤 이미지 생성 (텍스트 영역 시뮬레이션)
    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # 전처리 (OCR Detection 모델 입력 형식에 맞게)
    if len(input_shape) == 4:
        if input_shape[1] == 3:  # NCHW 형식
            preprocessed = test_image.transpose(2, 0, 1)  # HWC -> CHW
            preprocessed = np.expand_dims(preprocessed, axis=0)  # CHW -> NCHW
        else:  # NHWC 형식
            preprocessed = np.expand_dims(test_image, axis=0)  # HWC -> NHWC
    else:
        preprocessed = test_image
    
    # DX Engine 입력은 C-contiguous 배열로 고정 (transpose 결과는 비연속 메모리일 수 있음)
    preprocessed = np.ascontiguousarray(preprocessed, dtype=np.uint8)
    logger.info(f"입력 contiguous: {preprocessed.flags['C_CONTIGUOUS']}")
    
    # =========================================================================
    # Warm-up (첫 실행은 느릴 수 있음)
    # =========================================================================
    logger.info("Warm-up 중...")
    for _ in range(3):
        _ = session.run(input_data=[preprocessed])
    
    # =========================================================================
    # 벤치마크 (여러 번 실행하여 평균 측정)
    # =========================================================================
    num_iterations = 10
    logger.info(f"벤치마크 시작 ({num_iterations}회 반복)...")
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = session.run(input_data=[preprocessed])
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    # 통계 계산
    times = np.array(times)
    mean_time = np.mean(times)
    
    logger.info(f"\n📊 성능 측정 결과:")
    logger.info(f"   {mean_time:.3f} s/it")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 벤치마크 완료")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        benchmark_ocr_det_dxnn()
    except KeyboardInterrupt:
        logger.info("\n벤치마크 중단됨")
    except Exception as e:
        logger.exception(f"벤치마크 실패: {e}")
