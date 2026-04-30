#!/usr/bin/env python3
"""
OCR Detection Model Input 추출 스크립트

Layout detection 결과(Title, Text 등)를 입력으로 하고,
그 안의 OcrText(category_id=15) 라인들을 ground truth로 사용합니다.

Usage:
    python extract_ocr_detection_input.py --json_path <path_to_model.json> --pdf_path <path_to_pdf> --output_dir <output_dir>
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

from rapid_doc.utils.pdf_image_tools import load_images_from_pdf
from rapid_doc.utils.enum_class import ImageType


def poly_to_bbox(poly: List[float]) -> Tuple[int, int, int, int]:
    """Poly 좌표를 bbox로 변환"""
    if len(poly) == 8:
        xs = [poly[i] for i in range(0, 8, 2)]
        ys = [poly[i] for i in range(1, 8, 2)]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    return None


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """두 bbox의 IoU 계산"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 교집합 영역
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def is_inside(inner_box: Tuple, outer_box: Tuple, threshold: float = 0.8) -> bool:
    """inner_box가 outer_box 안에 포함되는지 확인"""
    iou = calculate_iou(inner_box, outer_box)
    if iou == 0:
        return False
    
    # inner_box의 면적 대비 교집합 비율
    x1_min, y1_min, x1_max, y1_max = inner_box
    x2_min, y2_min, x2_max, y2_max = outer_box
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    inner_area = (x1_max - x1_min) * (y1_max - y1_min)
    
    return (inter_area / inner_area) >= threshold if inner_area > 0 else False


class OCRDetectionInputExtractor:
    """OCR Detection Model Input 추출기"""
    
    def __init__(self, json_path: str, pdf_path: str, output_dir: str):
        self.json_path = Path(json_path)
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 하위 디렉토리 생성
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.vis_dir = self.output_dir / "visualizations"
        
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)
        
        # JSON 데이터 로드
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)
        
        print(f"✅ JSON 파일 로드 완료: {self.json_path}")
        print(f"   총 페이지 수: {len(self.model_data)}")
    
    def load_pdf_images(self) -> List[np.ndarray]:
        """PDF에서 이미지 추출"""
        print(f"\n📄 PDF 이미지 로딩 중: {self.pdf_path}")
        
        if isinstance(self.pdf_path, str):
            with open(self.pdf_path, 'rb') as f:
                pdf_bytes = f.read()
        else:
            with open(str(self.pdf_path), 'rb') as f:
                pdf_bytes = f.read()
        
        images_list, pdf_doc = load_images_from_pdf(
            pdf_bytes,
            dpi=200,
            image_type=ImageType.PIL
        )
        
        # PIL to numpy
        np_images = []
        for img_dict in images_list:
            pil_img = img_dict['img_pil']
            np_img = np.array(pil_img)
            if len(np_img.shape) == 2:  # grayscale
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
            elif np_img.shape[2] == 4:  # RGBA
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            np_images.append(np_img)
        
        print(f"✅ 로드된 페이지 수: {len(np_images)}")
        return np_images
    
    def extract_text_blocks_with_lines(self) -> List[Dict]:
        """
        텍스트 블록(Title, Text)과 그 안의 OcrText 라인들을 추출
        
        Returns:
            [
                {
                    'page_no': int,
                    'block_category': int,  # 0: Title, 1: Text
                    'block_bbox': (xmin, ymin, xmax, ymax),
                    'block_poly': [...],
                    'text_lines': [
                        {
                            'bbox': (xmin, ymin, xmax, ymax),
                            'poly': [...],
                            'text': str,
                            'score': float
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        # 텍스트 블록 카테고리: Title(0), Text(1), ImageCaption(4), TableCaption(6)
        text_block_categories = [0, 1, 4, 6, 7]
        ocr_text_category = 15
        
        results = []
        
        for page_idx, page in enumerate(self.model_data):
            page_no = page['page_info']['page_no']
            layout_dets = page.get('layout_dets', [])
            
            # 1. 텍스트 블록 찾기
            text_blocks = []
            for det in layout_dets:
                cat_id = det.get('category_id', -1)
                if cat_id in text_block_categories:
                    bbox = det.get('bbox')
                    poly = det.get('poly')
                    
                    if bbox and len(bbox) == 4:
                        block_bbox = tuple(int(x) for x in bbox)
                    elif poly:
                        block_bbox = poly_to_bbox(poly)
                    else:
                        continue
                    
                    text_blocks.append({
                        'category_id': cat_id,
                        'bbox': block_bbox,
                        'poly': poly,
                        'score': det.get('score', 0.0)
                    })
            
            # 2. OcrText 라인 찾기
            ocr_lines = []
            for det in layout_dets:
                cat_id = det.get('category_id', -1)
                if cat_id == ocr_text_category:
                    bbox = det.get('bbox')
                    poly = det.get('poly')
                    
                    if bbox and len(bbox) == 4:
                        line_bbox = tuple(int(x) for x in bbox)
                    elif poly:
                        line_bbox = poly_to_bbox(poly)
                    else:
                        continue
                    
                    ocr_lines.append({
                        'bbox': line_bbox,
                        'poly': poly,
                        'text': det.get('text', ''),
                        'score': det.get('score', 0.0)
                    })
            
            # 3. 각 텍스트 블록에 포함된 OcrText 라인 매칭
            for block in text_blocks:
                block_bbox = block['bbox']
                matched_lines = []
                
                for line in ocr_lines:
                    line_bbox = line['bbox']
                    # 라인이 블록 안에 포함되는지 확인
                    if is_inside(line_bbox, block_bbox, threshold=0.7):
                        matched_lines.append(line)
                
                # 라인이 있는 블록만 추가
                if matched_lines:
                    results.append({
                        'page_no': page_no,
                        'page_idx': page_idx,
                        'block_category': block['category_id'],
                        'block_bbox': block_bbox,
                        'block_poly': block['poly'],
                        'block_score': block['score'],
                        'text_lines': matched_lines
                    })
        
        return results
    
    def save_detection_sample(
        self,
        sample_idx: int,
        page_img: np.ndarray,
        sample_data: Dict
    ) -> str:
        """
        Detection 샘플 저장
        
        Args:
            sample_idx: 샘플 인덱스
            page_img: 페이지 이미지
            sample_data: 샘플 데이터
        
        Returns:
            저장된 파일명
        """
        page_no = sample_data['page_no']
        block_category = sample_data['block_category']
        block_bbox = sample_data['block_bbox']
        text_lines = sample_data['text_lines']
        
        category_names = {0: "Title", 1: "Text", 4: "ImgCap", 6: "TblCap", 7: "TblFoot"}
        cat_name = category_names.get(block_category, f"Cat{block_category}")
        
        # 파일명
        filename = f"p{page_no:03d}_{cat_name}_idx{sample_idx:04d}"
        
        # 1. 블록 이미지 크롭
        xmin, ymin, xmax, ymax = block_bbox
        h, w = page_img.shape[:2]
        
        xmin = max(0, min(xmin, w - 1))
        ymin = max(0, min(ymin, h - 1))
        xmax = max(xmin + 1, min(xmax, w))
        ymax = max(ymin + 1, min(ymax, h))
        
        cropped_img = page_img[ymin:ymax, xmin:xmax].copy()
        
        # 이미지 저장
        img_path = self.images_dir / f"{filename}.png"
        cv2.imwrite(str(img_path), cropped_img)
        
        # 2. 라벨 저장 (상대 좌표)
        label_path = self.labels_dir / f"{filename}.txt"
        with open(label_path, 'w', encoding='utf-8') as f:
            for line in text_lines:
                line_bbox = line['bbox']
                lxmin, lymin, lxmax, lymax = line_bbox
                
                # 블록 기준 상대 좌표로 변환
                rel_xmin = lxmin - xmin
                rel_ymin = lymin - ymin
                rel_xmax = lxmax - xmin
                rel_ymax = lymax - ymin
                
                # 정규화 (0~1)
                block_w = xmax - xmin
                block_h = ymax - ymin
                
                norm_xmin = rel_xmin / block_w
                norm_ymin = rel_ymin / block_h
                norm_xmax = rel_xmax / block_w
                norm_ymax = rel_ymax / block_h
                
                # YOLO 형식: class x_center y_center width height
                x_center = (norm_xmin + norm_xmax) / 2
                y_center = (norm_ymin + norm_ymax) / 2
                width = norm_xmax - norm_xmin
                height = norm_ymax - norm_ymin
                
                # class 0 = text line
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 3. 시각화 이미지 생성
        vis_img = cropped_img.copy()
        for line in text_lines:
            line_bbox = line['bbox']
            lxmin, lymin, lxmax, lymax = line_bbox
            
            # 블록 기준 상대 좌표
            rel_xmin = lxmin - xmin
            rel_ymin = lymin - ymin
            rel_xmax = lxmax - xmin
            rel_ymax = lymax - ymin
            
            # 사각형 그리기
            cv2.rectangle(vis_img, (rel_xmin, rel_ymin), (rel_xmax, rel_ymax), 
                         (0, 255, 0), 2)
            
            # 텍스트 표시 (선택적)
            text = line['text'][:20] if len(line['text']) > 20 else line['text']
            cv2.putText(vis_img, text, (rel_xmin, rel_ymin - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        vis_path = self.vis_dir / f"{filename}_vis.png"
        cv2.imwrite(str(vis_path), vis_img)
        
        return filename
    
    def run(self, max_samples: int = None):
        """전체 추출 프로세스 실행"""
        print("\n" + "=" * 80)
        print("OCR Detection Model Input 추출 시작")
        print("=" * 80)
        
        # 1. PDF 이미지 로드
        pdf_images = self.load_pdf_images()
        
        # 2. 텍스트 블록과 라인 추출
        print(f"\n🔍 텍스트 블록과 OcrText 라인 추출 중...")
        samples = self.extract_text_blocks_with_lines()
        print(f"✅ 추출된 텍스트 블록: {len(samples)}개")
        
        # 통계
        total_lines = sum(len(s['text_lines']) for s in samples)
        print(f"   총 텍스트 라인 수: {total_lines}개")
        
        # Category별 통계
        cat_stats = {}
        for sample in samples:
            cat_id = sample['block_category']
            cat_stats[cat_id] = cat_stats.get(cat_id, 0) + 1
        
        print("\n   블록 Category 통계:")
        category_names = {
            0: "Title", 1: "Text", 4: "ImageCaption",
            6: "TableCaption", 7: "TableFootnote"
        }
        for cat_id in sorted(cat_stats.keys()):
            cat_name = category_names.get(cat_id, f"Cat{cat_id}")
            print(f"     [{cat_id}] {cat_name:20s}: {cat_stats[cat_id]:4d}개")
        
        # 3. 샘플 저장
        print(f"\n💾 샘플 저장 중...")
        samples_to_save = samples[:max_samples] if max_samples else samples
        
        saved_count = 0
        metadata = []
        
        for idx, sample in enumerate(samples_to_save):
            page_idx = sample['page_idx']
            if page_idx >= len(pdf_images):
                continue
            
            page_img = pdf_images[page_idx]
            filename = self.save_detection_sample(idx, page_img, sample)
            
            metadata.append({
                'filename': filename,
                'page_no': sample['page_no'],
                'category': sample['block_category'],
                'num_lines': len(sample['text_lines']),
                'block_score': sample['block_score']
            })
            
            saved_count += 1
            
            if (idx + 1) % 50 == 0:
                print(f"   진행: {idx + 1}/{len(samples_to_save)}")
        
        print(f"✅ 저장 완료: {saved_count}개 샘플")
        
        # 4. 메타데이터 저장
        metadata_path = self.output_dir / "metadata.csv"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("filename,page_no,category,num_lines,block_score\n")
            for meta in metadata:
                f.write(f"{meta['filename']},{meta['page_no']},"
                       f"{meta['category']},{meta['num_lines']},{meta['block_score']:.3f}\n")
        
        # 5. 요약 보고서
        report_path = self.output_dir / "extraction_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("OCR Detection Model Input 추출 보고서\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"입력 JSON: {self.json_path}\n")
            f.write(f"입력 PDF: {self.pdf_path}\n")
            f.write(f"출력 디렉토리: {self.output_dir}\n\n")
            f.write(f"추출된 샘플 수: {saved_count}\n")
            f.write(f"총 텍스트 라인 수: {total_lines}\n\n")
            f.write("디렉토리 구조:\n")
            f.write(f"  - images/        : 입력 이미지 (텍스트 블록)\n")
            f.write(f"  - labels/        : 라벨 파일 (YOLO 형식)\n")
            f.write(f"  - visualizations/: 시각화 이미지\n\n")
            f.write("블록 Category 통계:\n")
            for cat_id in sorted(cat_stats.keys()):
                cat_name = category_names.get(cat_id, f"Cat{cat_id}")
                f.write(f"  [{cat_id}] {cat_name:20s}: {cat_stats[cat_id]:4d}개\n")
        
        print(f"\n📊 보고서 생성: {report_path}")
        print(f"\n✅ 모든 작업 완료!")
        print(f"\n출력 구조:")
        print(f"  {self.output_dir}/")
        print(f"  ├── images/          # 입력 이미지 (텍스트 블록)")
        print(f"  ├── labels/          # 라벨 파일 (YOLO 형식)")
        print(f"  ├── visualizations/  # 시각화")
        print(f"  ├── metadata.csv")
        print(f"  └── extraction_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description="OCR Detection Model Input 추출 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 사용
  python extract_ocr_detection_input.py \\
      --json_path demo/output-offline/demo1/auto/demo1_model.json \\
      --pdf_path test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf \\
      --output_dir output/ocr_detection_dataset

  # 최대 50개 샘플
  python extract_ocr_detection_input.py \\
      --json_path demo/output-offline/demo1/auto/demo1_model.json \\
      --pdf_path test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf \\
      --output_dir output/ocr_detection_50 \\
      --max_samples 50
        """
    )
    
    parser.add_argument('--json_path', type=str, required=True,
                       help='*_model.json 파일 경로')
    parser.add_argument('--pdf_path', type=str, required=True,
                       help='원본 PDF 파일 경로')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='출력 디렉토리')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='최대 샘플 수 (기본값: 전체)')
    
    args = parser.parse_args()
    
    extractor = OCRDetectionInputExtractor(
        json_path=args.json_path,
        pdf_path=args.pdf_path,
        output_dir=args.output_dir
    )
    
    extractor.run(max_samples=args.max_samples)


if __name__ == '__main__':
    main()
