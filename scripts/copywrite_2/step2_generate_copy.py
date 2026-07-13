# -*- coding: utf-8 -*-
"""
step2_generate_copy.py
------------------------
[2단계] tiny.json + 매장정보 -> compare_models() 실행 -> llm_output.json 저장

경로는 config.py에서 가져옵니다 (이미지 경로를 바꿨다면 config.py만 수정하면 됨).
"""

import json
import os

from dotenv import load_dotenv

import config
from compare_copy_llms import compare_models, to_image_gen_params

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

if not os.path.exists(config.PRODUCT_ATTRS_PATH):
    raise FileNotFoundError(
        f"{config.PRODUCT_ATTRS_PATH} 가 없습니다. step1_prepare_tiny_json.py를 먼저 실행하세요."
    )

with open(config.PRODUCT_ATTRS_PATH, encoding="utf-8") as f:
    product_attrs = json.load(f)

df = compare_models(
    product_attrs,
    config.STORE_INFO,
    tone="친근한",
    product="크루아상 & 콜드브루 세트",
)

copy_results = df[["model", "tone", "title", "subtitle", "price", "cta"]].to_dict(orient="records")

with open(config.LLM_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(copy_results, f, ensure_ascii=False, indent=4)

print(f"GPT가 문구를 새로 생성하여 {config.LLM_OUTPUT_PATH} 파일로 저장했습니다!")

preview = to_image_gen_params(focusing_degree=0.8, background_intensity=0.3, preservation_degree=0.6)
print("이미지 생성 파라미터 미리보기:", preview)
