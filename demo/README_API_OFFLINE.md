# RapidDoc Offline API

폐쇄망 환경에서 DX Engine을 사용하는 RapidDoc FastAPI 서버입니다.

## 특징

- **폐쇄망 모드**: 외부 네트워크 접근 완전 차단
- **DX Engine 지원**: 고성능 DX Engine 기본 사용
- **다중 엔진 지원**: Layout, OCR, Formula, Table 각각 엔진 선택 가능
- **FastAPI 기반**: Swagger UI를 통한 간편한 API 테스트

## 기본 엔진 설정

```python
LAYOUT_ENGINE = "dxengine"       # Layout 모델
OCR_ENGINE = "dxengine"          # OCR 모델
FORMULA_ENGINE = "onnxruntime"   # Formula 모델
TABLE_ENGINE = "dxengine"        # Table 모델
```

## 실행 방법

### 1. 가상환경 활성화 및 서버 실행

```bash
# 가상환경 활성화
source venv/bin/activate

# 서버 실행
python demo/app_offline.py
```

또는

```bash
# uvicorn으로 실행
source venv/bin/activate
uvicorn demo.app_offline:app --host 0.0.0.0 --port 8888
```

### 2. API 문서 확인

브라우저에서 다음 주소로 접속:
- Swagger UI: http://localhost:8888/docs
- ReDoc: http://localhost:8888/redoc

### 3. Health Check

```bash
curl http://localhost:8888/health
```

응답 예시:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "api": "RapidDoc Offline API",
  "mode": "closed_environment",
  "default_engines": {
    "layout": "dxengine",
    "ocr": "dxengine",
    "formula": "onnxruntime",
    "table": "dxengine"
  }
}
```

## API 사용 방법

### Python으로 테스트

```bash
# 테스트 스크립트 실행
python demo/test_api_offline.py
```

### curl로 테스트

```bash
curl -X POST "http://localhost:8888/file_parse" \
  -F "files=@test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf" \
  -F "output_dir=./output-api" \
  -F "formula_enable=true" \
  -F "table_enable=true" \
  -F "layout_engine=dxengine" \
  -F "ocr_engine=dxengine" \
  -F "formula_engine=onnxruntime" \
  -F "table_engine=dxengine" \
  -F "return_md=true"
```

### Python requests로 사용

```python
import requests

url = "http://localhost:8888/file_parse"

with open("test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf", "rb") as f:
    files = {"files": ("example.pdf", f, "application/pdf")}
    
    data = {
        "output_dir": "./output-api",
        "formula_enable": "true",
        "table_enable": "true",
        "layout_engine": "dxengine",
        "ocr_engine": "dxengine",
        "formula_engine": "onnxruntime",
        "table_engine": "dxengine",
        "return_md": "true",
    }
    
    response = requests.post(url, files=files, data=data)
    result = response.json()
    
    # Markdown 결과 확인
    md_content = result["results"][0]["md_content"]
    print(md_content)
```

## API 파라미터

### 필수 파라미터
- `files`: 파싱할 파일 목록 (PDF, 이미지)

### 선택 파라미터
- `output_dir`: 출력 디렉토리 (기본값: "./output-offline")
- `clear_output_file`: 처리 후 파일 삭제 여부 (기본값: false)
- `lang_list`: 언어 목록 (기본값: ["ch"])
- `backend`: 백엔드 (기본값: "pipeline")
- `parse_method`: 파싱 방법 - auto/ocr/txt (기본값: "auto")
- `formula_enable`: 수식 인식 활성화 (기본값: true)
- `table_enable`: 표 인식 활성화 (기본값: true)

### 엔진 선택 파라미터
- `layout_engine`: Layout 엔진 - dxengine/onnxruntime/openvino (기본값: "dxengine")
- `ocr_engine`: OCR 엔진 - dxengine/onnxruntime/openvino/torch/paddle (기본값: "dxengine")
- `formula_engine`: Formula 엔진 - dxengine/onnxruntime/openvino (기본값: "onnxruntime")
- `table_engine`: Table 엔진 - dxengine/onnxruntime/torch (기본값: "dxengine")

### 출력 옵션
- `return_md`: Markdown 반환 (기본값: true)
- `return_middle_json`: Middle JSON 반환 (기본값: false)
- `return_model_output`: 모델 출력 반환 (기본값: false)
- `return_content_list`: 콘텐츠 목록 반환 (기본값: false)
- `return_images`: 이미지를 base64로 반환 (기본값: false)

### 페이지 범위
- `start_page_id`: 시작 페이지 (기본값: 0)
- `end_page_id`: 종료 페이지 (기본값: 99999)

## 응답 형식

```json
{
  "results": [
    {
      "filename": "example.pdf",
      "md_content": "# Document Title\n\n...",
      "backend": "pipeline",
      "engines": {
        "layout": "dxengine",
        "ocr": "dxengine",
        "formula": "onnxruntime",
        "table": "dxengine"
      }
    }
  ],
  "total_files": 1,
  "successful_files": 1,
  "mode": "closed_environment",
  "engines_used": {
    "layout": "dxengine",
    "ocr": "dxengine",
    "formula": "onnxruntime",
    "table": "dxengine"
  }
}
```

## 엔진 비교

### DX Engine
- 장점: 최고 성능, 최적화된 추론
- 단점: .dxnn 파일 필요
- 권장: Layout, OCR, Table

### ONNX Runtime
- 장점: 범용성, 안정성
- 단점: DX Engine 대비 느림
- 권장: Formula (dxnn 변환 전)

### OpenVINO
- 장점: Intel CPU 최적화
- 단점: 특정 하드웨어 의존
- 권장: Intel CPU 환경

## 문제 해결

### 1. 모델 파일이 없다는 오류
```bash
# 필요한 모델 파일 확인
ls dxnn_models/  # DX Engine 모델
ls onnx_models/  # ONNX 모델
```

### 2. 네트워크 에러
폐쇄망 모드에서는 의도적으로 네트워크를 차단합니다. 
모든 모델 파일이 로컬에 있어야 합니다.

### 3. 메모리 부족
큰 PDF 처리 시 메모리가 부족할 수 있습니다.
`start_page_id`와 `end_page_id`로 페이지 범위를 제한하세요.

## demo_offline.py와의 차이점

| 항목 | demo_offline.py | app_offline.py |
|------|----------------|----------------|
| 인터페이스 | CLI (명령행) | FastAPI (REST API) |
| 입력 방식 | 로컬 파일 경로 | HTTP 파일 업로드 |
| 출력 방식 | 로컬 파일 저장 | JSON 응답 |
| 사용 사례 | 배치 처리 | 웹 서비스 통합 |
| 설정 | 코드 수정 | API 파라미터 |

## 참고

- 원본 스크립트: [demo_offline.py](demo_offline.py)
- Docker API: [../docker/app.py](../docker/app.py)
