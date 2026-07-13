# -*- coding: utf-8 -*-
"""
config.py
-----------
경로를 한 곳에서만 관리하기 위한 공용 설정 파일.
여기 값만 바꾸면 step1~5 / notebook_main.py 전부에 동일하게 반영됩니다.

이전 버전은 각 step 파일에 PRODUCT_IMAGE_PATH 등을 따로 하드코딩해서,
step1만 고치고 step5는 안 고치면 기본값(./bakery.png)으로 계속 실행되는
문제가 있었습니다. 이 파일이 그 문제를 해결한 "단일 소스"입니다.
"""

import os

# ── 상품 정보 ─────────────────────────────────────────────────
PRODUCT_ID = "bakery_croissant_cookie_coldbrew"

# 원본 상품 사진 경로. 본인 로컬 환경에 맞게 이 한 줄만 수정하면
# step1~5 / notebook_main.py 전체에 반영됩니다.
PRODUCT_IMAGE_PATH = "/Users/apple/Desktop/AD-content-generator/bakery.png"

PRODUCT_CAPTION = (
    "A wooden plate holding a plain butter croissant, a chocolate-drizzled pastry, "
    "and a chocolate chip cookie, paired with a tall glass of iced cold brew coffee"
    "—warm sunlit cafe setup for a bakery breakfast set."
)

STORE_INFO_RAW = {"store_type": "", "store_name": "", "location": "", "highlight": ""}
STORE_INFO = {"name": "cafe masion", "price": "12,000원"}

# ── 산출 경로 (필요하면 수정, 기본값 그대로 써도 무방) ────────────────
BASE_DIR = "./tiny_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
JSON_DIR = os.path.join(BASE_DIR, "jsons")

# step1이 저장한 뒤 step2/step4/step5가 공통으로 읽는 경로
# (PRODUCT_IMAGE_PATH의 확장자를 그대로 따라감)
PRODUCT_ATTRS_PATH = os.path.join(JSON_DIR, f"{PRODUCT_ID}.json")

LLM_OUTPUT_PATH = "./llm_output.json"
OPTION_POOLS_PATH = "./option_pools.json"
RUN_CONFIG_PATH = "./run_config.json"

BANNER_OUTPUT_DIR = "./ad_banners"
RANDOM_BATCH_OUTPUT_DIR = "./real_random_banners"
