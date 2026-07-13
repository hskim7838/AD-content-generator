# -*- coding: utf-8 -*-
"""
step3_compose_banners_simple.py
----------------------------------
[3단계] llm_output.json(모델별 카피 결과)을 제품 사진 위에 합성해 배너 이미지 생성.

경로는 config.py에서 가져옵니다.
"""

import json
import os

import config
from banner_utils import compose_banner_simple

if os.path.exists(config.LLM_OUTPUT_PATH):
    with open(config.LLM_OUTPUT_PATH, "r", encoding="utf-8") as f:
        copy_results = json.load(f)
    print(f" 성공: {config.LLM_OUTPUT_PATH} 파일로부터 데이터를 불러와 copy_results에 저장했습니다.")
else:
    copy_results = []
    print(f" ⚠ 오류: {config.LLM_OUTPUT_PATH} 파일이 존재하지 않습니다. step2를 먼저 실행하세요.")

os.makedirs(config.BANNER_OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=== 광고 배너 합성 시작 ===")
    for copy in copy_results:
        out_name = f"banner_{copy['model'].replace(' ', '_').replace('.', '')}.jpg"
        out_path = os.path.join(config.BANNER_OUTPUT_DIR, out_name)
        compose_banner_simple(config.PRODUCT_IMAGE_PATH, copy, out_path)
        print(f"  ✅ [{copy['model']}] 배너 생성 완료 -> {out_path}")
    print("=== 완료 ===")
