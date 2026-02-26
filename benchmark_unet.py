#!/usr/bin/env python3
"""
UNET 모델의 순수 DX Engine 추론 성능 측정 스크립트

전체 파이프라인 없이 UNET 모델의 순수 추론 속도만 측정합니다.
"""
import time
import numpy as np
from pathlib import Path
from loguru import logger


def benchmark_unet_dxnn():
    """UNET 모델의 DX Engine 추론 성능 측정"""
    
    # =========================================================================
    # 모델 경로 설정
    # =========================================================================
    project_root = Path(__file__).parent.parent.absolute()
    unet_model_path = project_root / "dxnn_models" / "unet.dxnn"
    
    if not unet_model_path.exists():
        logger.error(f"UNET 모델을 찾을 수 없습니다: {unet_model_path}")
        return
    
    logger.info("=" * 80)
    logger.info("UNET 모델 순수 DXNN 추론 성능 측정")
    logger.info("=" * 80)
    logger.info(f"모델 경로: {unet_model_path}")
    
    # =========================================================================
    # DX Engine 세션 생성
    # =========================================================================
    from dx_engine import InferenceEngine

    logger.info("DXNN Runtime 세션 생성 중...")
    session_start = time.perf_counter()
    session = InferenceEngine(str(unet_model_path))
    session_time = time.perf_counter() - session_start
    logger.info(f"✓ 세션 생성 완료: {session_time:.3f}초")
    
    # 입력/출력 정보 확인 (모델 메타데이터 기반)
    input_infos = session.get_input_tensors_info()
    output_infos = session.get_output_tensors_info()
    if len(input_infos) != 1:
        raise ValueError(f"UNET benchmark는 단일 입력 모델만 지원합니다: input_count={len(input_infos)}")

    input_info = input_infos[0]
    input_name = input_info["name"]
    input_shape = [int(dim) for dim in input_info["shape"]]
    input_dtype = np.dtype(input_info["dtype"]).type
    logger.info(f"입력 이름: {input_name}")
    logger.info(f"입력 shape: {input_shape}")
    logger.info(f"입력 dtype: {input_dtype}")

    logger.info(f"출력 개수: {len(output_infos)}")
    for i, info in enumerate(output_infos):
        logger.info(f"  출력 {i}: {info['name']} - shape: {info['shape']}, dtype: {info['dtype']}")
    
    # =========================================================================
    # 테스트 이미지 생성 (모델 입력 shape 기반)
    # =========================================================================
    # 동적 축(-1/0)은 벤치마크용 기본값으로 대체
    resolved_shape = [1 if dim <= 0 else dim for dim in input_shape]
    if len(resolved_shape) != 4:
        raise ValueError(f"지원하지 않는 입력 rank: {resolved_shape} (expected 4D)")

    if resolved_shape[-1] in (1, 3, 4):
        input_layout = "NHWC"
        _, height, width, channels = resolved_shape
    elif resolved_shape[1] in (1, 3, 4):
        input_layout = "NCHW"
        _, channels, height, width = resolved_shape
    else:
        raise ValueError(f"입력 레이아웃을 판별할 수 없습니다: {resolved_shape}")
    
    logger.info("\n" + "=" * 80)
    logger.info(
        f"성능 측정 (layout: {input_layout}, shape: {resolved_shape}, "
        f"이미지 크기: {height}x{width}, 채널: {channels}, 배치: 1)"
    )
    logger.info("=" * 80)
    
    # 랜덤 이미지 생성
    test_image = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    
    # 전처리 (모델 입력 레이아웃에 맞게)
    if input_layout == "NCHW":
        preprocessed = test_image.transpose(2, 0, 1)  # HWC -> CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)  # CHW -> NCHW
    else:
        preprocessed = np.expand_dims(test_image, axis=0)  # HWC -> NHWC
    
    # DX Engine 입력은 C-contiguous 배열로 고정
    preprocessed = np.ascontiguousarray(preprocessed, dtype=input_dtype)
    
    logger.info(f"전처리 후 shape: {preprocessed.shape}")
    logger.info(f"전처리 후 dtype: {preprocessed.dtype}")
    logger.info(f"입력 contiguous: {preprocessed.flags['C_CONTIGUOUS']}")
    
    # =========================================================================
    # Warm-up (첫 실행은 느릴 수 있음)
    # =========================================================================
    try:
        logger.info("Warm-up 중...")
        for _ in range(3):
            _ = session.run(input_data=preprocessed)
        
        # =========================================================================
        # 벤치마크 (여러 번 실행하여 평균 측정)
        # =========================================================================
        num_iterations = 10
        logger.info(f"벤치마크 시작 ({num_iterations}회 반복)...")
        
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = session.run(input_data=preprocessed)
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
    finally:
        logger.info("세션 해제 중...")
        session.dispose()
        logger.info("✓ 세션 해제 완료")


if __name__ == "__main__":
    try:
        benchmark_unet_dxnn()
    except KeyboardInterrupt:
        logger.info("\n벤치마크 중단됨")
    except Exception as e:
        logger.exception(f"벤치마크 실패: {e}")
