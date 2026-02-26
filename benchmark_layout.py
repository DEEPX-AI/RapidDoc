#!/usr/bin/env python3
"""
Layout 모델의 순수 ONNX Runtime 추론 성능 측정 스크립트

전체 파이프라인 없이 Layout 모델의 순수 추론 속도만 측정합니다.
"""
import time
import numpy as np
from pathlib import Path
from loguru import logger


def benchmark_layout_dxnn():
    
    # =========================================================================
    # 모델 경로 설정
    # =========================================================================
    project_root = Path(__file__).parent.parent.absolute()
    layout_part1_model_path = project_root / "dxnn_models" / "pp_doclayout_l_part1.dxnn"
    layout_part2_model_path = project_root / "onnx_models" / "pp_doclayout_l_part2.onnx"
    
    if not layout_part1_model_path.exists():
        logger.error(f"Layout 모델을 찾을 수 없습니다: {layout_part1_model_path}")
        return
    
    logger.info("=" * 80)
    logger.info("Layout 모델 순수 DXNN Runtime 추론 성능 측정")
    logger.info("=" * 80)
    logger.info(f"모델 경로: {layout_part1_model_path}")
    
    # =========================================================================
    # DXNN Runtime 세션 생성
    # =========================================================================
    import onnxruntime as ort
    from dx_engine import InferenceEngine
    
    # 세션 옵션 설정
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 프로바이더 설정 (CPU)
    providers = ['CPUExecutionProvider']
    logger.info("DXNN Runtime 세션 (Part1) 생성 중...")
    session_start = time.perf_counter()
    dxnn_session = InferenceEngine(str(layout_part1_model_path))   
    session_time = time.perf_counter() - session_start
    logger.info(f"✓ 세션 생성 완료: {session_time:.3f}초")

    logger.info("ONNX Runtime 세션 (Part2) 생성 중...")
    session_start = time.perf_counter()
    ort_session = ort.InferenceSession(
        str(layout_part2_model_path),
        sess_options=sess_options,
        providers=providers
    )
    session_time = time.perf_counter() - session_start
    logger.info(f"✓ 세션 생성 완료: {session_time:.3f}초")
    
    # =========================================================================
    # 테스트 이미지 생성 (고정 크기: 640x640)
    # =========================================================================
    height, width = 640, 640
    
    logger.info("\n" + "=" * 80)
    logger.info(f"성능 측정 (이미지 크기: {height}x{width}, 배치: 1)")
    logger.info("=" * 80)
    
    # 랜덤 이미지 생성 (문서 페이지 시뮬레이션)
    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # 전처리 (Layout 모델은 여러 입력 필요)
    # 입력 1: image (NCHW)
    image_input = test_image.transpose(2, 0, 1)  # HWC -> CHW
    image_input = np.expand_dims(image_input, axis=0)  # CHW -> NCHW
    image_input = np.ascontiguousarray(image_input, dtype=np.float32)
    
    # 입력 2: im_shape (원본 이미지 크기)
    im_shape = np.array([[height, width]], dtype=np.float32)
    
    # 입력 3: scale_factor (스케일 팩터, 변환 없으면 1.0)
    scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)
    
    logger.info(f"입력 contiguous: {image_input.flags['C_CONTIGUOUS']}")
    
    # =========================================================================
    # Warm-up (첫 실행은 느릴 수 있음)
    # =========================================================================
    logger.info("Warm-up 중...")
    for _ in range(3):
        out = dxnn_session.run([image_input])
        ort_feed = {
            "p2o.pd_op.concat.12.0": out[0],
            "p2o.pd_op.layer_norm.20.0": out[1],
            "im_shape": im_shape,
            "scale_factor": scale_factor
        }
        _ = ort_session.run(None, ort_feed)
        
    # =========================================================================
    # 벤치마크 (여러 번 실행하여 평균 측정)
    # =========================================================================
    num_iterations = 10
    logger.info(f"벤치마크 시작 ({num_iterations}회 반복)...")
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        out = dxnn_session.run([image_input])
        ort_feed = {
            "p2o.pd_op.concat.12.0": out[0],
            "p2o.pd_op.layer_norm.20.0": out[1],
            "im_shape": im_shape,
            "scale_factor": scale_factor
        }
        _ = ort_session.run(None, ort_feed)
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
        benchmark_layout_dxnn()
    except KeyboardInterrupt:
        logger.info("\n벤치마크 중단됨")
    except Exception as e:
        logger.exception(f"벤치마크 실패: {e}")
