#!/usr/bin/env python3
"""
OCR Detection Model Input Sample 추출 스크립트

이 스크립트는 demo*_model.json 파일을 분석하여 OCR detection 모델의 
input sample을 추출합니다.

Usage:
    python extract_ocr_samples.py --json_path <path_to_model.json> --pdf_path <path_to_pdf> --output_dir <output_dir>
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import sys

# rapid_doc 모듈 import를 위한 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rapid_doc.utils.pdf_image_tools import load_images_from_pdf, ImageType


class OCRSampleExtractor:
    """OCR Detection Sample 추출기"""
    
    def __init__(self, json_path: str, pdf_path: str, output_dir: str):
        """
        Args:
            json_path: *_model.json 파일 경로
            pdf_path: 원본 PDF 파일 경로
            output_dir: 추출된 샘플을 저장할 디렉토리
        """
        self.json_path = Path(json_path)
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 데이터 로드
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)
        
        print(f"✅ JSON 파일 로드 완료: {self.json_path}")
        print(f"   총 페이지 수: {len(self.model_data)}")
    
    def load_pdf_images(self) -> List[np.ndarray]:
        """PDF에서 이미지 추출"""
        print(f"\n📄 PDF 이미지 로딩 중: {self.pdf_path}")
        
        # PDF를 바이트로 읽기
        with open(self.pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # PIL 이미지로 로드 (튜플 반환: images_list, pdf_doc)
        pil_images, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        
        # PIL을 numpy 배열로 변환
        np_images = []
        for img_dict in pil_images:
            pil_img = img_dict['img_pil']
            # PIL Image를 numpy 배열로 변환 (RGB -> BGR for OpenCV)
            np_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            np_images.append(np_img)
        
        print(f"✅ 로드된 페이지 수: {len(np_images)}")
        return np_images
    
    def analyze_categories(self) -> Dict[int, int]:
        """Category별 통계 분석"""
        category_counts = {}
        
        for page in self.model_data:
            for det in page.get('layout_dets', []):
                cat_id = det.get('category_id', -1)
                category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        return category_counts
    
    def extract_text_regions(self, target_categories: List[int] = None) -> List[Dict]:
        """
        텍스트 영역 추출
        
        Args:
            target_categories: 추출할 category_id 리스트
                             None인 경우 기본값: [0, 1, 4, 6, 7, 15, 16] (텍스트 관련)
        
        Returns:
            추출된 영역 정보 리스트
        """
        if target_categories is None:
            # 기본값: 텍스트 관련 카테고리
            # 0:Title, 1:Text, 4:ImageCaption, 6:TableCaption, 
            # 7:TableFootnote, 15:OcrText, 16:LowScoreText
            target_categories = [0, 1, 4, 6, 7, 15, 16]
        
        text_regions = []
        
        for page_idx, page in enumerate(self.model_data):
            page_no = page['page_info']['page_no']
            page_width = page['page_info']['width']
            page_height = page['page_info']['height']
            
            for det_idx, det in enumerate(page.get('layout_dets', [])):
                cat_id = det.get('category_id', -1)
                
                if cat_id in target_categories:
                    region_info = {
                        'page_no': page_no,
                        'page_idx': page_idx,
                        'det_idx': det_idx,
                        'category_id': cat_id,
                        'score': det.get('score', 0.0),
                        'poly': det.get('poly'),
                        'bbox': det.get('bbox'),
                        'text': det.get('text', ''),
                        'page_width': page_width,
                        'page_height': page_height,
                    }
                    text_regions.append(region_info)
        
        return text_regions
    
    def poly_to_bbox(self, poly: List[float]) -> Tuple[int, int, int, int]:
        """
        Poly 좌표를 bbox로 변환
        
        Args:
            poly: [x1, y1, x2, y2, x3, y3, x4, y4]
        
        Returns:
            (xmin, ymin, xmax, ymax)
        """
        if len(poly) == 8:
            xs = [poly[i] for i in range(0, 8, 2)]
            ys = [poly[i] for i in range(1, 8, 2)]
            return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        return None
    
    def crop_and_save_samples(
        self, 
        pdf_images: List[np.ndarray], 
        text_regions: List[Dict],
        max_samples: int = None,
        min_width: int = 10,
        min_height: int = 10
    ) -> List[str]:
        """
        텍스트 영역을 크롭하여 저장
        
        Args:
            pdf_images: PDF 페이지 이미지 리스트
            text_regions: 추출할 텍스트 영역 정보
            max_samples: 최대 샘플 수 (None이면 전체)
            min_width: 최소 너비
            min_height: 최소 높이
        
        Returns:
            저장된 파일 경로 리스트
        """
        saved_files = []
        category_names = {
            0: "Title",
            1: "Text",
            4: "ImageCaption",
            6: "TableCaption",
            7: "TableFootnote",
            15: "OcrText",
            16: "LowScoreText",
        }
        
        # CSV 파일로 메타데이터 저장
        metadata_path = self.output_dir / "samples_metadata.csv"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("filename,page_no,category_id,category_name,score,width,height,text\n")
        
        samples_to_process = text_regions[:max_samples] if max_samples else text_regions
        
        print(f"\n🖼️  이미지 크롭 및 저장 중...")
        print(f"   총 {len(samples_to_process)}개 샘플 처리")
        
        for idx, region in enumerate(samples_to_process):
            page_idx = region['page_idx']
            page_no = region['page_no']
            cat_id = region['category_id']
            cat_name = category_names.get(cat_id, f"Cat{cat_id}")
            
            # 페이지 이미지 가져오기
            if page_idx >= len(pdf_images):
                print(f"⚠️  경고: 페이지 {page_no}를 찾을 수 없습니다.")
                continue
            
            page_img = pdf_images[page_idx]
            
            # bbox 계산
            if region['bbox']:
                bbox = region['bbox']
                if len(bbox) == 4:
                    xmin, ymin, xmax, ymax = [int(x) for x in bbox]
                else:
                    continue
            elif region['poly']:
                bbox_tuple = self.poly_to_bbox(region['poly'])
                if bbox_tuple is None:
                    continue
                xmin, ymin, xmax, ymax = bbox_tuple
            else:
                continue
            
            # 좌표 검증 및 클리핑
            h, w = page_img.shape[:2]
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(xmin + 1, min(xmax, w))
            ymax = max(ymin + 1, min(ymax, h))
            
            crop_width = xmax - xmin
            crop_height = ymax - ymin
            
            # 최소 크기 체크
            if crop_width < min_width or crop_height < min_height:
                continue
            
            # 이미지 크롭
            cropped = page_img[ymin:ymax, xmin:xmax]
            
            # 파일명 생성
            filename = f"p{page_no:03d}_{cat_name}_idx{idx:04d}.png"
            save_path = self.output_dir / filename
            
            # 저장
            cv2.imwrite(str(save_path), cropped)
            saved_files.append(str(save_path))
            
            # 메타데이터 저장
            text_escaped = region['text'].replace('"', '""').replace('\n', '\\n')
            with open(metadata_path, 'a', encoding='utf-8') as f:
                f.write(f'"{filename}",{page_no},{cat_id},{cat_name},'
                       f'{region["score"]:.3f},{crop_width},{crop_height},"{text_escaped}"\n')
            
            if (idx + 1) % 100 == 0:
                print(f"   진행: {idx + 1}/{len(samples_to_process)}")
        
        print(f"✅ 저장 완료: {len(saved_files)}개 샘플")
        print(f"   메타데이터: {metadata_path}")
        
        return saved_files
    
    def create_summary_report(self, text_regions: List[Dict], saved_files: List[str]):
        """요약 보고서 생성"""
        report_path = self.output_dir / "extraction_report.txt"
        
        category_names = {
            0: "Title", 1: "Text", 4: "ImageCaption",
            6: "TableCaption", 7: "TableFootnote", 
            15: "OcrText", 16: "LowScoreText",
        }
        
        # Category별 통계
        cat_stats = {}
        for region in text_regions:
            cat_id = region['category_id']
            cat_stats[cat_id] = cat_stats.get(cat_id, 0) + 1
        
        # 페이지별 통계
        page_stats = {}
        for region in text_regions:
            page_no = region['page_no']
            page_stats[page_no] = page_stats.get(page_no, 0) + 1
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("OCR Detection Sample 추출 보고서\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"입력 JSON: {self.json_path}\n")
            f.write(f"입력 PDF: {self.pdf_path}\n")
            f.write(f"출력 디렉토리: {self.output_dir}\n\n")
            
            f.write(f"총 추출된 영역 수: {len(text_regions)}\n")
            f.write(f"저장된 샘플 수: {len(saved_files)}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("Category별 통계:\n")
            f.write("-" * 80 + "\n")
            for cat_id in sorted(cat_stats.keys()):
                cat_name = category_names.get(cat_id, f"Unknown({cat_id})")
                count = cat_stats[cat_id]
                f.write(f"  [{cat_id:2d}] {cat_name:20s}: {count:5d}개\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("페이지별 통계:\n")
            f.write("-" * 80 + "\n")
            for page_no in sorted(page_stats.keys()):
                count = page_stats[page_no]
                f.write(f"  Page {page_no:3d}: {count:5d}개\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\n📊 요약 보고서 생성: {report_path}")
        
        # 콘솔에도 출력
        print("\n" + "=" * 80)
        print("추출 결과 요약")
        print("=" * 80)
        print(f"총 추출된 영역: {len(text_regions)}개")
        print(f"저장된 샘플: {len(saved_files)}개")
        print("\nCategory별 통계:")
        for cat_id in sorted(cat_stats.keys()):
            cat_name = category_names.get(cat_id, f"Unknown({cat_id})")
            count = cat_stats[cat_id]
            print(f"  [{cat_id:2d}] {cat_name:20s}: {count:5d}개")
    
    def run(
        self, 
        target_categories: List[int] = None,
        max_samples: int = None,
        min_width: int = 10,
        min_height: int = 10
    ):
        """
        전체 추출 프로세스 실행
        
        Args:
            target_categories: 추출할 category_id 리스트
            max_samples: 최대 샘플 수
            min_width: 최소 너비
            min_height: 최소 높이
        """
        print("\n" + "=" * 80)
        print("OCR Detection Sample 추출 시작")
        print("=" * 80)
        
        # 1. Category 분석
        print("\n📊 Category 분석 중...")
        all_categories = self.analyze_categories()
        print("전체 Category 통계:")
        for cat_id, count in sorted(all_categories.items()):
            print(f"  Category {cat_id:2d}: {count:5d}개")
        
        # 2. PDF 이미지 로드
        pdf_images = self.load_pdf_images()
        
        # 3. 텍스트 영역 추출
        print(f"\n🔍 텍스트 영역 추출 중...")
        if target_categories:
            print(f"   대상 Categories: {target_categories}")
        text_regions = self.extract_text_regions(target_categories)
        print(f"✅ 추출된 영역: {len(text_regions)}개")
        
        # 4. 샘플 크롭 및 저장
        saved_files = self.crop_and_save_samples(
            pdf_images, 
            text_regions,
            max_samples=max_samples,
            min_width=min_width,
            min_height=min_height
        )
        
        # 5. 보고서 생성
        self.create_summary_report(text_regions, saved_files)
        
        print("\n✅ 모든 작업 완료!")
        print(f"   출력 디렉토리: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="OCR Detection Model Input Sample 추출 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 사용 (모든 텍스트 관련 카테고리)
  python extract_ocr_samples.py \\
      --json_path demo/output-offline/demo1/auto/demo1_model.json \\
      --pdf_path test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf \\
      --output_dir output/ocr_samples

  # 특정 카테고리만 추출 (OcrText만)
  python extract_ocr_samples.py \\
      --json_path demo/output-offline/demo1/auto/demo1_model.json \\
      --pdf_path test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf \\
      --output_dir output/ocr_samples_only \\
      --categories 15

  # 최대 100개 샘플만 추출
  python extract_ocr_samples.py \\
      --json_path demo/output-offline/demo1/auto/demo1_model.json \\
      --pdf_path test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf \\
      --output_dir output/ocr_samples_100 \\
      --max_samples 100
        """
    )
    
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='*_model.json 파일 경로'
    )
    
    parser.add_argument(
        '--pdf_path',
        type=str,
        required=True,
        help='원본 PDF 파일 경로'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='추출된 샘플을 저장할 디렉토리'
    )
    
    parser.add_argument(
        '--categories',
        type=int,
        nargs='+',
        default=None,
        help='추출할 category_id 리스트 (기본값: 0,1,4,6,7,15,16 - 텍스트 관련)'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='최대 샘플 수 (기본값: 전체)'
    )
    
    parser.add_argument(
        '--min_width',
        type=int,
        default=10,
        help='최소 너비 (픽셀, 기본값: 10)'
    )
    
    parser.add_argument(
        '--min_height',
        type=int,
        default=10,
        help='최소 높이 (픽셀, 기본값: 10)'
    )
    
    args = parser.parse_args()
    
    # 추출기 실행
    extractor = OCRSampleExtractor(
        json_path=args.json_path,
        pdf_path=args.pdf_path,
        output_dir=args.output_dir
    )
    
    extractor.run(
        target_categories=args.categories,
        max_samples=args.max_samples,
        min_width=args.min_width,
        min_height=args.min_height
    )


if __name__ == '__main__':
    main()
