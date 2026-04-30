# RapidDoc Offline Demo

폐쇄망 환경에서 로컬 모델만 사용하여 PDF를 파싱하는 데모 스크립트입니다.

## 📋 목차

- [특징](#특징)
- [설치](#설치)
- [사용법](#사용법)
  - [기본 실행](#기본-실행)
  - [커맨드라인 옵션](#커맨드라인-옵션)
  - [예제](#예제)
- [엔진 설정](#엔진-설정)
- [출력 결과](#출력-결과)
- [문제 해결](#문제-해결)

## ✨ 특징

- **완전 폐쇄망 지원**: 외부 네트워크 접근 없이 로컬 모델만 사용
- **다중 엔진 지원**: Layout, OCR, Formula, Table 각각 독립적으로 엔진 선택 가능
- **고성능 DX Engine**: DX Engine을 기본으로 사용하여 빠른 처리 속도
- **유연한 설정**: 커맨드라인 인자로 간편하게 옵션 변경

## 🔧 설치

### 1. 가상환경 활성화

```bash
cd /path/to/rapid_doc
source venv/bin/activate
```

### 2. 필요한 모델 파일 확인

#### DX Engine 사용 시
`dxnn_models/` 디렉토리에 다음 파일들이 필요합니다:
```
dxnn_models/
├── ch_PP-OCRv5_server_det.dxnn
├── ch_PP-OCRv5_rec_server_infer.dxnn
├── pp_doclayout_l_part1.dxnn
├── unet.dxnn
└── (기타 모델 파일들...)
```

#### ONNX Runtime 사용 시
`onnx_models/` 디렉토리에 다음 파일들이 필요합니다:
```
onnx_models/
├── ch_PP-OCRv5_server_det.onnx
├── ch_PP-OCRv5_rec_server_infer.onnx
├── pp_doclayout_l.onnx
├── pp_formulanet_plus_l.onnx
├── unet.onnx
└── (기타 모델 파일들...)
```

## 🚀 사용법

### 기본 실행

```bash
# 기본 설정으로 test_files 디렉토리의 PDF 파일 처리
python demo/demo_offline.py
```

### 커맨드라인 옵션

```bash
python demo/demo_offline.py --help
```

#### 옵션 목록

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-h, --help` | 도움말 표시 | - |
| `--input-dir DIR` | 입력 PDF 파일 디렉토리 | `test_files` |
| `--output-dir DIR` | 출력 디렉토리 | `demo/output-offline` |
| `--layout-engine ENGINE` | Layout 엔진 선택 | `dxengine` |
| `--ocr-engine ENGINE` | OCR 엔진 선택 | `dxengine` |
| `--formula-engine ENGINE` | Formula 엔진 선택 | `onnxruntime` |
| `--table-engine ENGINE` | Table 엔진 선택 | `dxengine` |
| `--no-formula` | 수식 인식 비활성화 | (활성화) |
| `--no-table` | 표 인식 비활성화 | (활성화) |
| `--max-files N` | 처리할 최대 파일 수 (0=전체) | `2` |

### 예제

#### 1. 기본 DX Engine으로 실행
```bash
python demo/demo_offline.py
```

#### 2. 모든 엔진을 ONNX Runtime으로 실행
```bash
python demo/demo_offline.py \
  --layout-engine onnxruntime \
  --ocr-engine onnxruntime \
  --formula-engine onnxruntime \
  --table-engine onnxruntime
```

#### 3. 특정 디렉토리의 모든 PDF 처리
```bash
python demo/demo_offline.py \
  --input-dir /path/to/pdfs \
  --output-dir /path/to/output \
  --max-files 0
```

#### 4. 수식 인식 없이 실행
```bash
python demo/demo_offline.py --no-formula
```

#### 5. Layout과 OCR만 DX Engine, 나머지는 ONNX Runtime
```bash
python demo/demo_offline.py \
  --layout-engine dxengine \
  --ocr-engine dxengine \
  --formula-engine onnxruntime \
  --table-engine onnxruntime
```

## ⚙️ 엔진 설정

### 지원 엔진

| 모듈 | 지원 엔진 |
|------|-----------|
| **Layout** | `onnxruntime`, `dxengine`, `openvino` |
| **OCR** | `onnxruntime`, `dxengine`, `openvino`, `torch`, `paddle` |
| **Formula** | `onnxruntime`, `dxengine`, `openvino` |
| **Table** | `onnxruntime`, `dxengine`, `torch` |

### 엔진별 특징

#### DX Engine (`dxengine`)
- ✅ **장점**: 가장 빠른 처리 속도, NPU 하드웨어 가속 지원
- ⚠️ **단점**: `.dxnn` 모델 파일 필요, DX Engine 라이브러리 필요
- 📌 **권장**: 고성능이 필요한 경우

#### ONNX Runtime (`onnxruntime`)
- ✅ **장점**: 범용성, 안정성, CPU에서도 잘 작동
- ⚠️ **단점**: DX Engine보다 느림
- 📌 **권장**: 표준 환경, 호환성이 중요한 경우

#### OpenVINO (`openvino`)
- ✅ **장점**: Intel CPU/GPU에서 최적화된 성능
- ⚠️ **단점**: openvino 패키지 별도 설치 필요
- 📌 **권장**: Intel 하드웨어 사용 시

#### PyTorch (`torch`)
- ✅ **장점**: 유연성, 디버깅 용이
- ⚠️ **단점**: 느린 속도, 메모리 사용량 높음
- 📌 **권장**: 개발 및 디버깅 용도

## 📊 출력 결과

### 디렉토리 구조

```
output-offline/
└── document_name.pdf/
    └── auto/
        ├── auto.md                    # 마크다운 결과
        ├── content_list.json          # 구조화된 컨텐츠
        ├── middle.json                # 중간 처리 결과
        ├── model.json                 # 모델 원본 출력
        ├── layout_bbox/               # Layout 바운딩 박스 이미지
        │   ├── page_0.png
        │   ├── page_1.png
        │   └── ...
        └── span_bbox/                 # Span 바운딩 박스 이미지
            ├── page_0.png
            ├── page_1.png
            └── ...
```

### 파일 설명

- **auto.md**: 추출된 텍스트를 마크다운 형식으로 변환한 결과
- **content_list.json**: 페이지별 구조화된 컨텐츠 (텍스트, 표, 수식 등)
- **middle.json**: 모델 출력과 최종 결과 사이의 중간 처리 데이터
- **model.json**: 모델의 원본 출력 (바운딩 박스, 신뢰도 등)
- **layout_bbox/**: 페이지별 레이아웃 영역 시각화 이미지
- **span_bbox/**: 페이지별 텍스트 영역 시각화 이미지

## 📈 성능 측정

실행 시 자동으로 성능 측정 결과가 출력됩니다:

```
================================================================================
📈 성능 측정 요약 (Performance Summary)
================================================================================
📊 Layout   [    dxengine] |    4.90초 (  3.5%) |   13it | 0.377 s/it |   2.65 it/s
📐 Formula  [ onnxruntime] |  129.86초 ( 93.7%) |   59it | 2.201 s/it |   0.45 it/s
📄 PDF-det  [    dxengine] |    0.21초 (  0.2%) |   13it | 0.016 s/it |  61.65 it/s
🔍 OCR-det  [    dxengine] |    0.93초 (  0.7%) |    9it | 0.103 s/it |   9.71 it/s
📋 Table    [    dxengine] |    2.71초 (  2.0%) |    5it | 0.542 s/it |   1.85 it/s
--------------------------------------------------------------------------------
🔥 총 처리 시간: 138.60초
================================================================================
```

각 모듈별로 사용된 엔진과 처리 시간, 처리 속도를 확인할 수 있습니다.

## 🔍 문제 해결

### 1. ModuleNotFoundError

**증상:**
```
ModuleNotFoundError: No module named 'rapid_doc'
```

**해결:**
```bash
# 가상환경이 활성화되었는지 확인
source venv/bin/activate

# rapid_doc 패키지 설치 확인
pip install -e .
```

### 2. 모델 파일이 없음

**증상:**
```
FileNotFoundError: Model file not found
```

**해결:**
- DX Engine 사용 시: `dxnn_models/` 디렉토리에 `.dxnn` 파일 확인
- ONNX Runtime 사용 시: `onnx_models/` 디렉토리에 `.onnx` 파일 확인
- 모델 다운로드가 필요한 경우 별도 문서 참조

### 3. NPU 메모리 부족

**증상:**
```
[DXRT][Error] Ran out of NPU memory
```

**해결:**
```bash
# 1. 처리할 파일 수 줄이기
python demo/demo_offline.py --max-files 1

# 2. 일부 모듈을 ONNX Runtime으로 변경
python demo/demo_offline.py --formula-engine onnxruntime
```

### 4. 네트워크 접근 시도

**증상:**
```
ConnectionError: Network access blocked in closed environment
```

**해결:**
- 이것은 정상적인 동작입니다 (폐쇄망 보호 기능)
- 모든 모델 파일이 로컬에 준비되어 있는지 확인

### 5. OpenVINO 엔진 사용 불가

**증상:**
```
ImportError: openvino not found
```

**해결:**
```bash
# OpenVINO 설치
pip install openvino
```

## 📝 참고 사항

### 테이블 인식

현재 `ModelType.UNET`을 사용하여 유선 테이블(표 형태가 명확한 테이블)만 인식합니다.
- 필요 모델: `unet.onnx` 또는 `unet.dxnn`
- 무선 테이블도 인식하려면: `paddle_cls.onnx` + `slanet_plus.onnx` 추가 필요

### 수식 인식

수식 인식은 복잡한 수학 공식을 LaTeX 형식으로 변환합니다.
- 처리 시간이 가장 오래 걸리는 모듈입니다 (전체 시간의 90% 이상)
- 수식이 없는 문서는 `--no-formula` 옵션으로 비활성화하여 시간 절약 가능

### 배치 처리

대량의 PDF 파일을 처리할 때는 `--max-files` 옵션을 조정하세요:
```bash
# 한 번에 10개씩 처리
python demo/demo_offline.py --max-files 10

# 디렉토리의 모든 파일 처리
python demo/demo_offline.py --max-files 0
```

## 🔗 관련 문서

- [RapidDoc API 가이드](README_API_OFFLINE.md) - FastAPI 서버 사용법
- [Gradio 데모](README_GRADIO.md) - 웹 UI 데모
- [메인 README](../README.md) - 전체 프로젝트 개요
