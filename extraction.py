#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time
import os
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import argparse

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 기본 이미지 해상도 설정
IMAGE_RESOLUTION_SCALE = 2.0

def extract_images(input_file_path, output_dir="output", image_scale=IMAGE_RESOLUTION_SCALE):
    """
    Docling을 사용하여 PDF 파일의 이미지를 추출하고 PNG 파일로 저장합니다.
    
    Args:
        input_file_path (str): 입력 PDF 파일 경로
        output_dir (str): 출력 디렉토리 (기본값: "output")
        image_scale (float): 이미지 해상도 스케일 (기본값: 2.0)
    
    Returns:
        dict: 추출된 이미지 파일 정보
    """
    start_time = time.time()
    
    # 입력 파일 경로 확인
    input_path = Path(input_file_path)
    if not input_path.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return None
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 문서 파일명 (확장자 제외)
    doc_filename = input_path.stem
    
    logger.info(f"[시작] 파일 '{input_path.name}' 처리 시작")
    
    # PDF 파이프라인 옵션 설정
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = image_scale                  # 이미지 해상도 스케일 설정
    pipeline_options.generate_page_images = True                 # 페이지 이미지 생성 활성화
    pipeline_options.generate_picture_images = True              # 그림 이미지 생성 활성화
    pipeline_options.do_picture_description = True               # 그림 설명 생성 활성화
    pipeline_options.do_ocr = False                              # OCR 비활성화

    # DocumentConverter 인스턴스 생성
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    try:
        # 문서 변환
        logger.info(f"[변환] 문서 변환 중...")
        conv_res = doc_converter.convert(input_path)
        logger.info(f"[변환] 문서 변환 완료")

        # 이미지 정보 저장할 딕셔너리
        extracted_images = {
            "pages": [],       # 페이지 이미지
            "tables": [],      # 테이블 이미지
            "pictures": [],    # 그림 이미지
            "markdown": [],    # 마크다운 파일
            "html": []         # HTML 파일
        }
        
        # 1. 페이지 이미지 저장
        logger.info(f"[페이지] 페이지 이미지 저장 중...")
        page_count = 0
        if hasattr(conv_res.document, 'pages'):
            for page_no, page in conv_res.document.pages.items():
                # 페이지 번호 확인
                if hasattr(page, 'page_no'):
                    page_no = page.page_no
                
                # 이미지 파일명 생성
                page_image_filename = f"{doc_filename}-page-{page_no}.png"
                page_image_path = output_path / page_image_filename
                
                # 이미지 저장
                if hasattr(page, 'image') and hasattr(page.image, 'pil_image'):
                    with open(page_image_path, "wb") as fp:
                        page.image.pil_image.save(fp, format="PNG")
                    
                    page_count += 1
                    extracted_images["pages"].append({
                        "id": f"page_{page_no}",
                        "filename": str(page_image_path),
                        "type": "page"
                    })
                    
                    logger.info(f"  - 페이지 {page_no} 이미지 저장됨: {page_image_path}")
                else:
                    logger.warning(f"  - 페이지 {page_no}에 이미지 정보가 없음")
            
            logger.info(f"[페이지] {page_count}개 페이지 이미지 저장 완료")
        else:
            logger.warning("[페이지] 페이지 정보가 없습니다")
        
        # 2. 요소별 이미지 저장 (테이블, 그림)
        logger.info(f"[요소] 문서 요소 이미지 저장 중...")
        table_counter = 0
        picture_counter = 0
        
        if hasattr(conv_res.document, 'iterate_items'):
            for element, _level in conv_res.document.iterate_items():
                try:
                    # 테이블 요소 처리
                    if isinstance(element, TableItem):
                        table_counter += 1
                        
                        # 이미지 파일명 생성
                        table_image_filename = f"{doc_filename}-table-{table_counter}.png"
                        table_image_path = output_path / table_image_filename
                        
                        # get_image 메소드 호출
                        if hasattr(element, 'get_image') and callable(element.get_image):
                            with open(table_image_path, "wb") as fp:
                                element.get_image(conv_res.document).save(fp, "PNG")
                            
                            extracted_images["tables"].append({
                                "id": f"table_{table_counter}",
                                "filename": str(table_image_path),
                                "type": "table"
                            })
                            
                            logger.info(f"  - 테이블 {table_counter} 이미지 저장됨: {table_image_path}")
                        else:
                            logger.warning(f"  - 테이블 {table_counter}에 이미지 추출 메소드가 없음")
                    
                    # 그림 요소 처리
                    if isinstance(element, PictureItem):
                        picture_counter += 1
                        
                        # 이미지 파일명 생성
                        picture_image_filename = f"{doc_filename}-picture-{picture_counter}.png"
                        picture_image_path = output_path / picture_image_filename
                        
                        # get_image 메소드 호출
                        if hasattr(element, 'get_image') and callable(element.get_image):
                            with open(picture_image_path, "wb") as fp:
                                element.get_image(conv_res.document).save(fp, "PNG")
                            
                            # 캡션 정보 추출
                            caption = ""
                            if hasattr(element, 'caption_text') and callable(element.caption_text):
                                try:
                                    caption = element.caption_text(conv_res.document)
                                except:
                                    try:
                                        caption = element.caption_text()
                                    except:
                                        caption = f"Picture {picture_counter}"
                            elif hasattr(element, 'caption'):
                                caption = element.caption
                            else:
                                caption = f"Picture {picture_counter}"
                            
                            extracted_images["pictures"].append({
                                "id": f"picture_{picture_counter}",
                                "filename": str(picture_image_path),
                                "caption": caption,
                                "type": "picture"
                            })
                            
                            logger.info(f"  - 그림 {picture_counter} 이미지 저장됨: {picture_image_path}")
                        else:
                            logger.warning(f"  - 그림 {picture_counter}에 이미지 추출 메소드가 없음")
                except Exception as e:
                    logger.error(f"  - 요소 처리 중 오류 발생: {str(e)}")
            
            logger.info(f"[요소] {table_counter}개 테이블, {picture_counter}개 그림 이미지 저장 완료")
        else:
            logger.warning("[요소] iterate_items 메소드가 없습니다")
        
        # 3. 마크다운 파일 생성 (임베디드 이미지)
        if hasattr(conv_res.document, 'save_as_markdown'):
            try:
                md_embedded_filename = output_path / f"{doc_filename}-with-images.md"
                conv_res.document.save_as_markdown(md_embedded_filename, image_mode=ImageRefMode.EMBEDDED)
                
                extracted_images["markdown"].append({
                    "filename": str(md_embedded_filename),
                    "type": "embedded"
                })
                
                logger.info(f"[마크다운] 파일(임베디드 이미지) 생성: {md_embedded_filename}")
            except Exception as e:
                logger.error(f"[마크다운] 파일(임베디드) 생성 오류: {str(e)}")
            
            try:
                md_referenced_filename = output_path / f"{doc_filename}-with-image-refs.md"
                conv_res.document.save_as_markdown(md_referenced_filename, image_mode=ImageRefMode.REFERENCED)
                
                extracted_images["markdown"].append({
                    "filename": str(md_referenced_filename),
                    "type": "referenced"
                })
                
                logger.info(f"[마크다운] 파일(참조 이미지) 생성: {md_referenced_filename}")
            except Exception as e:
                logger.error(f"[마크다운] 파일(참조) 생성 오류: {str(e)}")
        else:
            logger.warning("[마크다운] save_as_markdown 메소드가 없습니다")
        
        # 4. HTML 파일 생성
        if hasattr(conv_res.document, 'save_as_html'):
            try:
                html_filename = output_path / f"{doc_filename}-with-image-refs.html"
                conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
                
                extracted_images["html"].append({
                    "filename": str(html_filename),
                    "type": "referenced"
                })
                
                logger.info(f"[HTML] 파일 생성: {html_filename}")
            except Exception as e:
                logger.error(f"[HTML] 파일 생성 오류: {str(e)}")
        else:
            logger.warning("[HTML] save_as_html 메소드가 없습니다")
        
        # 처리 시간 계산
        end_time = time.time() - start_time
        logger.info(f"[완료] 문서 처리 및 이미지 저장 완료: {end_time:.2f}초")
        
        # 결과 요약
        logger.info("\n===== 결과 요약 =====")
        logger.info(f"페이지 이미지: {len(extracted_images['pages'])}개")
        logger.info(f"테이블 이미지: {len(extracted_images['tables'])}개")
        logger.info(f"그림 이미지: {len(extracted_images['pictures'])}개")
        logger.info(f"마크다운 파일: {len(extracted_images['markdown'])}개")
        logger.info(f"HTML 파일: {len(extracted_images['html'])}개")
        logger.info("=====================\n")
        
        return extracted_images
    
    except Exception as e:
        logger.error(f"[오류] 문서 처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
def main():
    """
    명령줄에서 실행할 때 사용하는 메인 함수
    """
    parser = argparse.ArgumentParser(description='Docling을 사용하여 PDF 파일에서 이미지를 추출하고 저장합니다.')
    parser.add_argument('input_file', help='입력 PDF 파일 경로')
    parser.add_argument('-o', '--output-dir', default='output', help='출력 디렉토리 (기본값: output)')
    parser.add_argument('-s', '--scale', type=float, default=IMAGE_RESOLUTION_SCALE, help=f'이미지 해상도 스케일 (기본값: {IMAGE_RESOLUTION_SCALE})')
    
    args = parser.parse_args()
    
    result = extract_images(args.input_file, args.output_dir, args.scale)
    
    if result:
        print(f"\n추출 완료: 이미지가 '{args.output_dir}' 디렉토리에 저장되었습니다.")
        return 0
    else:
        print("\n오류: 이미지 추출에 실패했습니다.")
        return 1

if __name__ == "__main__":
    import sys
    import traceback
    
    try:
        sys.exit(main())
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 