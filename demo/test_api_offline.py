#!/usr/bin/env python3
"""
RapidDoc Offline API 테스트 클라이언트

사용법:
    python demo/test_api_offline.py
"""
import requests
import json
from pathlib import Path

# API 서버 주소
API_URL = "http://localhost:8888"

def test_health_check():
    """Health check 테스트"""
    print("=" * 80)
    print("Testing Health Check...")
    print("=" * 80)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_file_parse(pdf_path: str):
    """파일 파싱 테스트"""
    print("=" * 80)
    print(f"Testing File Parse: {pdf_path}")
    print("=" * 80)
    
    # 파일 열기
    with open(pdf_path, "rb") as f:
        files = {
            "files": (Path(pdf_path).name, f, "application/pdf")
        }
        
        # 요청 데이터
        data = {
            "output_dir": "./output-api-test",
            "clear_output_file": "false",
            "lang_list": ["ch"],
            "backend": "pipeline",
            "parse_method": "auto",
            "formula_enable": "true",
            "table_enable": "true",
            "layout_engine": "dxengine",
            "ocr_engine": "dxengine",
            "formula_engine": "onnxruntime",
            "table_engine": "dxengine",
            "return_md": "true",
            "return_middle_json": "false",
            "return_model_output": "false",
            "return_content_list": "false",
            "return_images": "false",
            "start_page_id": "0",
            "end_page_id": "99999",
        }
        
        print("Sending request...")
        response = requests.post(f"{API_URL}/file_parse", files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nTotal files: {result['total_files']}")
            print(f"Successful files: {result['successful_files']}")
            print(f"Mode: {result['mode']}")
            print(f"Engines used: {json.dumps(result['engines_used'], indent=2)}")
            
            # 결과 출력
            for file_result in result['results']:
                print(f"\nFile: {file_result['filename']}")
                if 'error' in file_result:
                    print(f"  Error: {file_result['error']}")
                else:
                    if 'md_content' in file_result and file_result['md_content']:
                        md_length = len(file_result['md_content'])
                        print(f"  Markdown: {md_length} characters")
                        # 처음 500자만 출력
                        print(f"\n  Preview (first 500 chars):")
                        print("-" * 80)
                        print(file_result['md_content'][:500])
                        print("-" * 80)
        else:
            print(f"Error Response:\n{response.text}")
    
    print()


if __name__ == "__main__":
    # Health check 테스트
    test_health_check()
    
    # 파일 파싱 테스트
    # PDF 파일 경로를 지정하세요
    pdf_dir = Path(__file__).parent / "pdfs"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if pdf_files:
        # 첫 번째 PDF 파일로 테스트
        test_file_parse(str(pdf_files[0]))
    else:
        print("No PDF files found in test_files directory")
        print("Please add a PDF file to test the API")
